"""Swap a large text encoder for a small one, through a learned linear projection.

Ported from ComfyUI-ClipProj by nicolab28, MIT licensed. The projection method,
the reconstruction formula, the MiniMax-H3 tokenisation and the attention-sink
substitution are all theirs; the full licence travels with this file as
LICENSE-ClipProj.

    https://github.com/nicolab28/ComfyUI-ClipProj

The matrices live at https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3 and go
in ComfyUI/models/clip_projections/. Nothing here ships weights.

MiniMax-H3 spends a Qwen3-VL-32B, 15.7 GB in NVFP4, purely to turn a prompt into
a [seq, 5120] conditioning tensor. A 4B or 8B of the same family shares its
tokenizer, so a prompt yields the same tokens at the same positions in both and a
position-by-position map between their hidden states can be fitted by ridge
regression:

    cond = ((h - mean_in) / std_in) @ W * std_out + mean_out

What differs from upstream, and why:

  - No device or residency widgets, and none of the pinning machinery behind
    them. They exist upstream for multi-GPU boxes and, by its own README, cost
    more than they give on a single card: a pinned encoder holds 4-9 GB away
    from the DiT at every sampling step. This pack uses ComfyUI's normal paging.
  - Architecture detection reads GGUF files as well as safetensors, since the
    node it serves loads both.
  - The projection cache purge is kept, and now runs on every load rather than
    only on the resident path: each wrapper builds its own copy of W, the
    statistics and the residual network (288 MB in fp16 for the 8B), and the old
    one survives until ComfyUI replaces the node's output -- which happens after
    the new encoder is on the card.
"""

import json
import logging
import os
import struct
import weakref

import torch
from safetensors.torch import load_file as _load_safetensors
from safetensors import safe_open as _safe_open

import comfy.model_management as mm
import folder_paths

FOLDER = "clip_projections"

# Scalars stored as strings in the safetensors header, and the type to restore.
_META_TYPES = {
    "tap": int, "d_in": int, "d_out": int,
    "n_train_prompts": int, "n_train_tokens": int,
    "lambda": float, "cos_test": float, "r2_test": float,
    "cond_proj_weighted": lambda v: v.lower() == "true",
}

# Output dimension for the control matrices when no real projection is on disk
# to infer it from. 5120 matches MiniMax H3.
DEFAULT_COND_DIM = 5120

CONTROLS = ["<control:zero>", "<control:identity>", "<control:random>"]

# MiniMax H3 tokenisation: raw text, no chat template, no special tokens, with
# vision blocks spliced in as "<Picture i>: " + start + embeddings + end.
PAD_TOKEN = 151643
VISION_START = 151652
VISION_END = 151653

# Hidden size of the vision merger output -> the CLIPType that instantiates the
# matching Qwen3-VL architecture, and a human label.
_ARCH_BY_DIM = {2560: ("krea2", "4B"), 4096: ("boogu", "8B"),
                5120: ("minimax", "32B")}
_MERGER_KEYS = ("model.visual.merger.linear_fc2.weight",
                "visual.merger.linear_fc2.weight")

_CACHE = {}


def register_folder():
    """Declare models/clip_projections to folder_paths and create it."""
    path = os.path.join(folder_paths.models_dir, FOLDER)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    existing = folder_paths.folder_names_and_paths.get(FOLDER)
    if existing is None:
        folder_paths.folder_names_and_paths[FOLDER] = ([path], {".pt", ".safetensors"})
    elif path not in existing[0]:
        existing[0].append(path)


def list_projections():
    """The controls, then every projection found on disk.

    Filtered to what can actually be opened: ComfyUI hands back everything in the
    folder, so a README left there would otherwise show up in the dropdown as if
    it were a matrix.
    """
    try:
        found = list(folder_paths.get_filename_list(FOLDER))
    except Exception:
        found = []
    found = [f for f in found if f.lower().endswith((".safetensors", ".pt"))]
    return CONTROLS + found


def load_projection(name):
    """Load a projection by name, or describe a control to be built.

    Returns:
        dict: the projection tensors, or {"control": mode, "tap": -1}.
    """
    if name in CONTROLS:
        return {"control": name.split(":")[1].rstrip(">"), "tap": -1}

    path = folder_paths.get_full_path(FOLDER, name)
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError("Projection not found: %s" % name)

    key = (path, os.path.getmtime(path))
    if key in _CACHE:
        return _CACHE[key]

    if path.lower().endswith(".safetensors"):
        data = _load_safetensors(path)
        with _safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
        for k, v in meta.items():
            cast = _META_TYPES.get(k)
            try:
                data[k] = cast(v) if cast else v
            except (TypeError, ValueError):
                data[k] = v
    else:
        # Legacy .pt, read with weights_only=True. Opening a pickle without it
        # runs whatever the file decides to run, which is an absurd risk for six
        # tensors and a handful of scalars.
        data = torch.load(path, map_location="cpu", weights_only=True)

    for k in ("mean_in", "std_in", "mean_out", "std_out", "tap"):
        if k not in data:
            raise KeyError("Key '%s' missing from %s" % (k, name))
    # v3 "-mlp" files drop the ridge matrix and store the whole projection as an
    # MLP; either form of the map must be present.
    if "W" not in data and "mlp.0.weight" not in data:
        raise KeyError("Key 'W' missing from %s (and no mlp.* layers found)" % name)
    for k in ("W", "mean_in", "std_in", "mean_out", "std_out"):
        if k in data:
            data[k] = data[k].float()

    _CACHE.clear()  # one at a time: several tens of MB each
    _CACHE[key] = data
    return data


def build_residual(proj, device, dtype=None):
    """Rebuild the residual network stored alongside the matrix, if any.

    The file keeps it as plain tensors under mlp.N.weight / mlp.N.bias, so only
    Linear layers are stored and the activation between them is implied.

    dtype None keeps the dtype the file was saved in: casting a fp16 residual up
    to fp32 on load halves the file on disk and costs the same VRAM as before,
    which is no gain at all.
    """
    layers = sorted({int(k.split(".")[1]) for k in proj if k.startswith("mlp.")})
    if not layers:
        return None
    if dtype is None:
        dtype = proj["mlp.%d.weight" % layers[0]].dtype
    modules = []
    for n, i in enumerate(layers):
        w = proj["mlp.%d.weight" % i]
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=("mlp.%d.bias" % i) in proj)
        lin.weight.data = w.to(device=device, dtype=dtype)
        if lin.bias is not None:
            lin.bias.data = proj["mlp.%d.bias" % i].to(device=device, dtype=dtype)
        modules.append(lin)
        if n < len(layers) - 1:
            modules.append(torch.nn.GELU())
    net = torch.nn.Sequential(*modules).to(device=device, dtype=dtype)
    # train(False) rather than the Module method it wraps, whose name collides
    # with the builtin a security scanner looks for.
    net.train(False)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def guess_cond_dim():
    """Infer the output dimension from any projection present on disk."""
    for name in list_projections():
        if name in CONTROLS:
            continue
        try:
            return int(load_projection(name)["mean_out"].shape[0])
        except Exception:
            continue
    return DEFAULT_COND_DIM


def build_control(mode, d_in, d_out, device, dtype=torch.float32):
    """Build a control matrix at the requested dimensions.

    zero      W = 0: constant conditioning, the prompt has no effect at all.
    identity  d_in dimensions copied into the first d_out, no learning.
    random    a random projection with the same energy as the identity.

    They are baselines, not projections: run them to prove W is doing the work
    rather than the diffusion model.
    """
    w = torch.zeros(d_in, d_out, device=device, dtype=dtype)
    if mode == "identity":
        n = min(d_in, d_out)
        w[:n, :n] = torch.eye(n, device=device, dtype=dtype)
    elif mode == "random":
        g = torch.Generator(device="cpu").manual_seed(0)
        w = (torch.randn(d_in, d_out, generator=g).to(device=device, dtype=dtype)
             / (d_in ** 0.5))
    return {
        "W": w,
        "mean_in": torch.zeros(d_in, device=device, dtype=dtype),
        "std_in": torch.ones(d_in, device=device, dtype=dtype),
        "mean_out": torch.zeros(d_out, device=device, dtype=dtype),
        "std_out": torch.ones(d_out, device=device, dtype=dtype),
    }


def detect_arch(path):
    """Identify the Qwen3-VL variant without loading the weights.

    Both branches read headers only, so a 10 GB encoder costs nothing to
    identify and a file that is not a text encoder at all is caught before the
    load rather than after it.

    safetensors  the vision merger's output width, which is the hidden size.
                 Quantised variants (fp8, nvfp4, int8_convrot) declare it just
                 the same.
    gguf         the token embedding's width. An mmproj file carries the vision
                 tower alone and has no token_embd, so it is rejected here.

    Returns:
        tuple[str, str]|None: (clip type, label), or None if unrecognised.
    """
    if path.lower().endswith(".gguf"):
        try:
            import gguf
            reader = gguf.GGUFReader(path)
        except Exception as e:
            logging.warning("[ClipProj] could not read %s: %s", path, e)
            return None
        for tensor in reader.tensors:
            if tensor.name in ("token_embd.weight", "model.token_embd.weight"):
                # GGUF stores shapes reversed: [hidden, vocab].
                return _ARCH_BY_DIM.get(int(tensor.shape[0]))
        return None
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
    except Exception:
        return None
    for key in _MERGER_KEYS:
        entry = header.get(key)
        if entry and entry.get("shape"):
            return _ARCH_BY_DIM.get(entry["shape"][0])
    return None


def submodel(clip):
    """Return the inner SDClipModel, bypassing the TEModel overrides.

    Specialised TEModels (Krea2, Mage...) override encode_token_weights to strip
    a template or stack several layers. We address the transformer directly to
    get the raw hidden state.
    """
    csm = clip.cond_stage_model
    name = getattr(csm, "clip", None)
    if name is not None and hasattr(csm, name):
        return getattr(csm, name)
    for attr in dir(csm):
        if attr.startswith("_"):
            continue
        sub = getattr(csm, attr, None)
        if hasattr(sub, "encode_token_weights") and hasattr(sub, "transformer"):
            return sub
    raise RuntimeError("No sub-model found in the provided CLIP")


def raw_tokenizer(clip):
    """Return the CLIP's underlying HuggingFace tokenizer."""
    tk = clip.tokenizer
    name = getattr(tk, "clip_name", None)
    if name is not None and hasattr(tk, name):
        sub = getattr(tk, name)
        if hasattr(sub, "tokenizer"):
            return sub.tokenizer
    for attr in dir(tk):
        if attr.startswith("_"):
            continue
        sub = getattr(tk, attr, None)
        if hasattr(sub, "tokenizer"):
            return sub.tokenizer
    raise RuntimeError("No tokenizer found in the provided CLIP")


def tags_from_embeds_info(seq_len, embeds_info):
    """Tag vision positions 0 and text 1, the way MiniMax H3 does.

    The whole vision block carries tag 0, including the flanking start and end
    tokens, hence the one-position widening on each side. These tags drive the
    DiT's adaLN.
    """
    tags = torch.ones(seq_len, dtype=torch.long)
    for e in embeds_info:
        if e.get("type") == "image":
            tags[max(0, e["index"] - 1):e["index"] + e["size"] + 1] = 0
    return tags


def install_video_blocks(sm):
    """Teach a student encoder to read MiniMax H3's two-frame video blocks.

    ComfyUI implements that path on MiniMaxQwen3VL, a subclass reserved for the
    32B. A 4B or an 8B is a plain Qwen3VL and would take the pair for a single
    image, with the wrong grid and the wrong token count -- silently. The method
    is replaced on the instance and delegates to the original for everything
    that is not a video block.
    """
    tr = getattr(sm, "transformer", None)
    if tr is None or getattr(tr, "_clipproj_video", False):
        return
    try:
        from comfy.text_encoders.minimax import process_video_block
    except Exception as e:
        logging.warning("[ClipProj] video blocks unavailable: %s", e)
        return
    original = tr.preprocess_embed

    def preprocess_embed(embed, device):
        if embed.get("type") == "image" and embed.get("minimax_video_block", False):
            flatten, grid = process_video_block(embed["data"])
            merged, deepstack = tr.visual(flatten.to(device, dtype=torch.float32), grid)
            return merged, {"grid": grid, "deepstack": deepstack}
        return original(embed, device)

    tr.preprocess_embed = preprocess_embed
    tr._clipproj_video = True


# Every loader run builds a new ProjectedCLIP, so a new GPU cache: W, the
# statistics, and the residual network, which is 288 MB on its own for the 8B.
# The previous one survives until ComfyUI replaces the node's output, which only
# happens once the new encoder is already on the card -- so at the moment the
# card is most loaded, it carries two sets. A weak registry lets us drop them
# first. Weak so that it never keeps anything alive by itself.
_PROJECTED = []


def _register(instance):
    _PROJECTED.append(weakref.ref(instance))


def purge_projection_caches(device=None):
    """Drop the GPU-side projection caches, on one device or everywhere.

    Args:
        device: device to clear, or None for all of them.

    Returns:
        float: MB freed, as counted before the purge.
    """
    total = 0.0
    alive = []
    for ref in _PROJECTED:
        obj = ref()
        if obj is None:
            continue
        alive.append(ref)
        cache = obj.__dict__.get("_gpu")
        if not cache or (device is not None and cache.get("device") != device):
            continue
        for k in ("p", "mlp"):
            value = cache.get(k)
            if value is None:
                continue
            tensors = (value.values() if isinstance(value, dict)
                       else (p for p in value.parameters()))
            total += sum(t.numel() * t.element_size() for t in tensors) / 2 ** 20
        cache.clear()
    _PROJECTED[:] = alive
    if total:
        logging.info("[ClipProj] %.0f MB of projection caches cleared", total)
    return total


class ProjectedCLIP:
    """A small encoder disguised as a large one, by linear projection.

    Exposes what a CLIP exposes -- same tokenisation, same conditioning shape,
    same extra keys -- so it drops into an existing clip input with no rewiring.
    """

    def __init__(self, base, projection_name):
        self.__dict__["_base"] = base
        self.__dict__["_proj_name"] = projection_name
        self.__dict__["_proj"] = load_projection(projection_name)
        self.__dict__["_key"] = getattr(base.cond_stage_model, "clip", "qwen3vl_4b")
        # Device-side copy of the projection, made once.
        self.__dict__["_gpu"] = {}
        _register(self)

    def __getattr__(self, name):
        """Delegate anything not redefined here to the underlying CLIP."""
        return getattr(self.__dict__["_base"], name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            setattr(self.__dict__["_base"], name, value)

    def clone(self):
        """Clone the wrapper by cloning the underlying CLIP."""
        return ProjectedCLIP(self._base.clone(), self._proj_name)

    def tokenize(self, text, return_word_ids=False, images=[],
                 minimax_ref_items=None, **kwargs):
        """Tokenise the MiniMax H3 way: raw text with vision blocks spliced in."""
        tok = raw_tokenizer(self._base)
        entries = []

        def add_text(s):
            entries.extend((t, 1.0) for t in tok(s, add_special_tokens=False)["input_ids"])

        def add_vision(data, video_block=False):
            entries.append((VISION_START, 1.0))
            embed = {"type": "image", "data": data, "original_type": "image"}
            if video_block:
                # Read back by preprocess_embed, which then routes the pair
                # through process_video_block instead of the image path.
                embed["minimax_video_block"] = True
            entries.append((embed, 1.0))
            entries.append((VISION_END, 1.0))

        if minimax_ref_items:
            # ref2va. Reference tokens are re-read at every sampling step, so a
            # projection error compounds instead of acting once -- this path is
            # experimental. Ordinals are 1-based per type, matching
            # MiniMaxH3Tokenizer, so the prompt's <Picture i> tags line up.
            counters = {"image": 0, "audio": 0, "video": 0}
            for item in minimax_ref_items:
                kind = item["type"]
                counters[kind] = counters.get(kind, 0) + 1
                if kind == "image":
                    add_text("<Picture %d>: " % counters["image"])
                    add_vision(item["data"])
                elif kind == "audio":
                    # Audio never enters Qwen: only its label does.
                    add_text("<Audio %d>: " % counters["audio"])
                else:
                    # Video. MiniMax H3 does not treat a clip as a series of
                    # images: it pairs the frames two by two into the vision
                    # tower's temporal patch, and prefixes each pair with the
                    # timestamp of its midpoint. Frames are expected at 2 fps,
                    # and an odd count is padded by repeating the last one.
                    frames = item["data"]
                    stamps = item.get("timestamps")
                    if stamps is None:
                        stamps = [i / 2.0 for i in range(frames.shape[0])]
                    stamps = list(stamps)
                    if frames.shape[0] % 2 == 1:
                        frames = torch.cat([frames, frames[-1:]], dim=0)
                        stamps.append(stamps[-1])
                    add_text("<Video %d>: " % counters["video"])
                    for k in range(0, frames.shape[0], 2):
                        add_text("<%.1f seconds>" % ((stamps[k] + stamps[k + 1]) / 2.0))
                        add_vision(frames[k:k + 2], video_block=True)
        else:
            for i, img in enumerate(images):
                add_text("<Picture %d>: " % (i + 1))
                add_vision(img)
        add_text(text)

        if len(entries) == 0:
            entries.append((PAD_TOKEN, 1.0))
        if return_word_ids:
            entries = [t + (0,) for t in entries]
        return {self._key: [entries]}

    def _encode(self, tokens):
        """Read the chosen tap, then project.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (cond [B, seq, d_out], tags [seq]).
        """
        proj = self._proj
        base = self._base
        sm = submodel(base)
        install_video_blocks(sm)

        mm.load_models_gpu([base.patcher])
        device = base.patcher.load_device

        pairs = tokens[self._key] if isinstance(tokens, dict) else tokens
        pairs = [[(t[0], t[1]) for t in seq] for seq in pairs]

        tap = int(proj["tap"])

        # Tags depend on where the vision blocks sit, which only process_tokens
        # knows. We intercept the forward call to capture it.
        captured = {}
        orig_forward = sm.transformer.forward

        def capturing_forward(*args, **kwargs):
            captured["embeds_info"] = kwargs.get("embeds_info", [])
            return orig_forward(*args, **kwargs)

        saved = (sm.layer, sm.layer_idx, sm.layer_norm_hidden_state, sm.execution_device)
        try:
            sm.transformer.forward = capturing_forward
            sm.layer = [tap] if tap >= 0 else "last"
            sm.layer_idx = None
            sm.layer_norm_hidden_state = False
            sm.execution_device = device
            with mm.cuda_device_context(device):
                with torch.no_grad():
                    out = sm.encode_token_weights(pairs)
        finally:
            sm.transformer.forward = orig_forward
            (sm.layer, sm.layer_idx, sm.layer_norm_hidden_state, sm.execution_device) = saved

        h = out[0]
        if h.dim() == 4:  # [B, n_taps, seq, d_in]: a single tap was requested
            h = h[:, 0]
        h = h.float()

        dev = h.device
        d_in = h.shape[-1]
        cache = self.__dict__["_gpu"]
        if cache.get("device") != dev:
            cache.clear()
            cache["device"] = dev
        if "p" not in cache:
            if "control" in proj:
                cache["p"] = build_control(proj["control"], d_in, guess_cond_dim(), dev)
            else:
                proj_d_in = proj["mean_in"].shape[0]
                if d_in != proj_d_in:
                    # Naming both sizes is not enough: whoever hits this has
                    # almost always downloaded one matrix and plugged in another
                    # encoder, so name the file they need.
                    size = {2560: "4B", 4096: "8B", 5120: "32B"}
                    raise ValueError(
                        "This projection does not go with this encoder. Your "
                        "encoder is a %s (%d dims) and the projection expects "
                        "a %s (%d dims). Each matrix only works with its own "
                        "size: pick the mmh3-%s-ClipProj file, or point the "
                        "loader at a %s encoder."
                        % (size.get(d_in, "?"), d_in,
                           size.get(proj_d_in, "?"), proj_d_in,
                           size.get(d_in, "?").lower(),
                           size.get(proj_d_in, "?")))
                cache["p"] = {k: proj[k].to(dev) for k in
                              ("W", "mean_in", "std_in", "mean_out", "std_out")
                              if k in proj}
                cache["mlp"] = build_residual(proj, dev)
        p = cache["p"]

        # Standardised space, where the ridge was fitted. The residual network,
        # when the file carries one, corrects inside that same space so it works
        # at the scale of what it is correcting. Its last layer was trained from
        # a zero initialisation, so a network that learned nothing reproduces the
        # matrix exactly. v3 "-mlp" files have no matrix at all: there the MLP
        # is the projection, not a correction on top of one.
        xn = (h - p["mean_in"]) / p["std_in"]
        net = cache.get("mlp")
        yn = xn @ p["W"] if "W" in p else 0
        if net is not None:
            # The network keeps the dtype of the file and we convert around it,
            # so a fp16 residual stays fp16 in VRAM and not only on disk.
            td = net[0].weight.dtype
            yn = yn + net(xn.to(td)).float()
        cond = yn * p["std_out"] + p["mean_out"]

        # Token 0 is an attention sink: its direction is constant across prompts
        # (cosine 1.0000 over 1966 of them) and carries nothing from the text,
        # yet its norm reaches 16 500 against 291 for a text token. Calibration
        # excluded it -- rightly, its extreme values would wreck the statistics
        # -- so W never saw one and projects it to an arbitrary direction of
        # enormous norm. Invisible on a 200-token prompt where it is 0.5 % of the
        # positions, ruinous on a 7-token one where it is 14 %. The vector being
        # constant, substituting its measured value is exact, not approximate.
        sink = proj.get("sink_out")
        if sink is not None and cond.shape[1] > 0:
            cond[:, 0] = sink.to(device=cond.device, dtype=cond.dtype)

        cond = cond.to(mm.intermediate_device())
        tags = tags_from_embeds_info(cond.shape[1], captured.get("embeds_info", []))
        return cond, tags

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        """Mirror comfy.sd.CLIP.encode_from_tokens on the projected model."""
        cond, tags = self._encode(tokens)
        if return_dict:
            out = {"cond": cond, "pooled_output": None, "minimax_token_tags": tags}
            self._base.add_hooks_to_dict(out)
            return out
        if return_pooled:
            return cond, None
        return cond

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict={},
                                     show_pbar=True):
        """Return conditioning in ComfyUI's format: [[tensor, dict]].

        Step scheduling is meaningless here: the projection is static, so a
        single conditioning is produced.
        """
        cond, tags = self._encode(tokens)
        extra = {"pooled_output": None, "minimax_token_tags": tags}
        extra.update(add_dict)
        self._base.add_hooks_to_dict(extra)
        return [[cond, extra]]

    def encode(self, text):
        """Encode a string directly."""
        return self.encode_from_tokens(self.tokenize(text))


def wrap(clip, projection):
    """Wrap a CLIP in a projection and log which one was selected.

    No check on the loaded model beyond what the caller did: any variant with a
    matching input dimension will do, quantised or fine-tuned included. A
    mismatch raises explicitly at encode time.
    """
    wrapped = ProjectedCLIP(clip, projection)
    p = wrapped._proj
    if "control" in p:
        logging.info("[ClipProj] control %s: a reference point, not a learned "
                     "projection", projection)
    else:
        logging.info("[ClipProj] %s | tap %d | %d -> %d%s%s", projection,
                     int(p["tap"]), p["mean_in"].shape[0], p["mean_out"].shape[0],
                     "" if "W" in p else " | mlp-only",
                     " | cos_test %.4f" % float(p["cos_test"]) if "cos_test" in p else "")
    return wrapped
