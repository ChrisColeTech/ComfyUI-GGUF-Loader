# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import warnings
import logging
import torch
import gguf
import re
import os

from .ops import GGMLTensor
from .dequant import is_quantized, dequantize_tensor

# Image / DiT / UNet GGUF architecture allowlist.
# Keep in sync with the arch tags we write when quantizing custom models under
# D:\models\image-models (and image-models-dev). Unknown tags raise at load.
IMG_ARCH_LIST = {
    # Upstream / common
    "flux", "sd1", "sdxl", "sd3", "aura", "hidream", "cosmos",
    "ltxv", "hyvid", "wan", "lumina2", "qwen_image",
    # Custom DiT tags used by our quant pipelines
    "minimax_h3",  # MiniMax H3 files are content-detected from the ltx2 tag
    "zimage",      # Z-Image turbo DiT
    "ideogram4",   # Ideogram-4 DiT
    "flux2",       # Flux2 conversion scripts (alias; some files also use "flux")
    "flux1",       # Flux1 VAE / older flux tags seen in the wild
    "krea2",       # Krea-2 (some builds; files may also use qwen_image)
    "minimax_music3",  # MiniMax Music 3 DiT (same tag on the TE, see below)
    "diffusion_model",  # generic fallback used by agnostic convert scripts
}
TXT_ARCH_LIST = {
    "t5", "t5encoder", "llama",
    "qwen2", "qwen2vl", "qwen3", "qwen35", "qwen3vl",
    "gemma3",
    # LTX-2.5 unified multimodal TE (Gemma4UnifiedForConditionalGeneration)
    "gemma4", "gemma4_unified",
    # Other TE GGUFs under image-models / open PRs
    "mistral3",  # Ministral-3 (Flux2 TE) — PR #440 / #436
    "clip",      # CLIP TE (and some mmproj files tagged arch=clip)
    # MiniMax Music 3 AR text encoder. Same arch tag as the DiT (both halves of
    # the model are tagged minimax_music3); keys are the original comfy names so
    # this arch is intentionally absent from every SD_MAP below.
    "minimax_music3",
}
VIS_TYPE_LIST = {"clip-vision", "mmproj"}

MINIMAX_H3_TENSOR_SIGNATURE = {
    "video_patch_proj.weight",
    "audio_patch_proj.weight",
}

def image_architecture(arch_str, tensor_names):
    """Return a safe, content-derived image architecture identifier."""
    if arch_str == "ltx2" and MINIMAX_H3_TENSOR_SIGNATURE <= set(tensor_names):
        return "minimax_h3"
    return arch_str

def read_gguf_arch(path):
    """Read general.architecture from a GGUF file, or None if absent/invalid.

    Cheap header-only probe used to validate loader node settings before the
    (slow) full state-dict load.
    """
    reader = gguf.GGUFReader(path)
    try:
        return get_field(reader, "general.architecture", str)
    except TypeError:
        return None

# GGUF text-encoder architectures that only work with one specific CLIPLoader
# `type`. Loaded under any other type, every key misses the target model and
# comfy crashes at first forward with a cryptic `'NoneType' object has no
# attribute 'device'` (seen in the wild with the Qwen2.5-VL TE + default
# `stable_diffusion` type → SDXLClipModel). Fail at load with the fix instead.
TE_TYPE_REQUIREMENTS = {
    "qwen2vl": "qwen_image",  # Qwen-Image / Qwen-Image-Edit 2509 TE
}

# llama.cpp writes `general.architecture = "clip"` for mmproj vision projectors.
# They carry the vision tower belonging to a VL text encoder (`v.blk.*` keys),
# never a text encoder of their own, so they are merged into the TE's state dict
# rather than handed to comfy as a second text encoder.
VISION_PROJECTOR_ARCHES = {"clip"}

def is_vision_projector(path):
    """True for mmproj GGUF files (vision tower, no text encoder)."""
    if not str(path).endswith(".gguf"):
        return False
    return read_gguf_arch(path) in VISION_PROJECTOR_ARCHES

def validate_te_type(path, type_str):
    """Raise a clear error when a GGUF TE is loaded with the wrong CLIP type.

    No-op for non-GGUF files and architectures without a known requirement.
    """
    if not str(path).endswith(".gguf"):
        return
    arch = read_gguf_arch(path)
    if arch in VISION_PROJECTOR_ARCHES:
        return  # merged into the TE, carries no type of its own
    required = TE_TYPE_REQUIREMENTS.get(arch)
    if required is not None and type_str != required:
        raise ValueError(
            f"'{os.path.basename(path)}' is a {arch!r} text encoder: load it with"
            f" type '{required}' (got '{type_str or 'stable_diffusion'}')."
            f" The mmproj vision tower is picked up automatically from the same"
            f" folder when present."
        )

def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))

def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        # extra check here as this is used for checking arch string
        if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF {field_name} key: expected string, got {field.types!r}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]].item())
    else:
        raise TypeError(f"Unknown field type {field_type}")

def get_list_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        return tuple(str(field.parts[part_idx], encoding="utf-8") for part_idx in field.data)
    elif field_type in [int, float, bool]:
        return tuple(field_type(field.parts[part_idx][0]) for part_idx in field.data)
    else:
        raise TypeError(f"Unknown field type {field_type}")

GGUF_SCALAR_TYPES = {
    gguf.GGUFValueType.UINT8: int, gguf.GGUFValueType.INT8: int,
    gguf.GGUFValueType.UINT16: int, gguf.GGUFValueType.INT16: int,
    gguf.GGUFValueType.UINT32: int, gguf.GGUFValueType.INT32: int,
    gguf.GGUFValueType.UINT64: int, gguf.GGUFValueType.INT64: int,
    gguf.GGUFValueType.FLOAT32: float, gguf.GGUFValueType.FLOAT64: float,
    gguf.GGUFValueType.BOOL: bool,
}

def get_gguf_metadata(reader):
    """Extract all simple metadata fields like safetensors"""
    metadata = {}
    for field_name in reader.fields:
        try:
            field = reader.get_field(field_name)
            if len(field.types) != 1:  # Simple scalar fields only
                continue
            value = field.parts[field.data[-1]]
            if field.types[0] == gguf.GGUFValueType.STRING:
                metadata[field_name] = str(value, "utf-8")
                continue
            cast = GGUF_SCALAR_TYPES.get(field.types[0])
            if cast is not None:
                # These parts are 1-element 1-D numpy arrays. numpy>=2 refuses to
                # convert those with int()/float(), so read through .item() —
                # otherwise every numeric field is silently dropped. Writers also
                # use the unsigned types for counts, so cover the whole set.
                metadata[field_name] = cast(value.item())
        except Exception as e:
            logging.debug(f"Skipping GGUF metadata field {field_name!r}: {e}")
            continue
    return metadata

def gguf_sd_loader(path, handle_prefix="model.diffusion_model.", is_text_model=False):
    """
    Read state dict as fake tensors
    """
    reader = gguf.GGUFReader(path)

    # filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        # PR #470: some Flux-compat GGUFs (e.g. LongCat bfl_format) name QK-norm
        # params ".scale" instead of comfy RMSNorm's ".weight" → silent NaNs.
        if sd_key.endswith(".query_norm.scale") or sd_key.endswith(".key_norm.scale"):
            sd_key = sd_key[:-len(".scale")] + ".weight"
        tensors.append((sd_key, tensor))

    # detect and verify architecture
    compat = None
    arch_str = get_field(reader, "general.architecture", str)
    type_str = get_field(reader, "general.type", str)
    arch_str = image_architecture(arch_str, (key for key, _ in tensors))
    if arch_str in [None, "pig", "cow"]:
        if is_text_model:
            raise ValueError(f"This gguf file is incompatible with llama.cpp!\nConsider using safetensors or a compatible gguf file\n({path})")
        compat = "sd.cpp" if arch_str is None else arch_str
        # import here to avoid changes to convert.py breaking regular models
        from .tools.convert import detect_arch
        try:
            arch_str = detect_arch(set(val[0] for val in tensors)).arch
            arch_str = image_architecture(arch_str, (key for key, _ in tensors))
        except Exception as e:
            raise ValueError(f"This model is not currently supported - ({e})")
    elif arch_str not in TXT_ARCH_LIST and is_text_model:
        if type_str not in VIS_TYPE_LIST:
            raise ValueError(f"Unexpected text model architecture type in GGUF file: {arch_str!r}")
    elif arch_str not in IMG_ARCH_LIST and not is_text_model:
        raise ValueError(f"Unexpected architecture type in GGUF file: {arch_str!r}")

    if compat:
        logging.warning(f"Warning: This gguf model file is loaded in compatibility mode '{compat}' [arch:{arch_str}]")

    # main loading loop
    state_dict = {}
    qtype_dict = {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        # torch_tensor = torch.from_numpy(tensor.data) # mmap

        # NOTE: line above replaced with this block to avoid persistent numpy warning about mmap
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data) # mmap

        # MiniMax Music 3: the text encoder GGUF carries the whole tokenizer.json
        # as a raw byte blob (GGML type I8) under this key. It is data, not a
        # weight — comfy's MiniMaxMusic3Tokenizer does .numpy().tobytes() on it,
        # so it has to survive verbatim. Wrapping it in a GGMLTensor (or letting
        # the non-weight float32 branch below dequantize it) destroys the bytes.
        if sd_key == "tokenizer_json":
            state_dict[sd_key] = torch_tensor.clone()
            qtype_dict["I8/blob"] = qtype_dict.get("I8/blob", 0) + 1
            continue

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            raw_shape = tensor.shape if tensor.shape is not None else torch_tensor.shape
            shape = torch.Size(tuple(int(v) for v in reversed(raw_shape)))
            # Workaround for stable-diffusion.cpp SDXL detection.
            if compat == "sd.cpp" and arch_str == "sdxl":
                if any([tensor_name.endswith(x) for x in (".proj_in.weight", ".proj_out.weight")]):
                    while len(shape) > 2 and shape[-1] == 1:
                        shape = shape[:-1]

        # PR #392: lumina2 / NextDiT (Z-Image) pad tokens may be stored 1D.
        if arch_str in {"lumina2", "zimage"} and sd_key in ("x_pad_token", "cap_pad_token"):
            if len(shape) == 1:
                shape = torch.Size((1, shape[0]))

        # LTX-2: keyframes_abs_pos_embedding is a [1, dim] parameter that
        # quantizers drop the batch axis from; restore it or load_state_dict
        # rejects the tensor with a size mismatch.
        if arch_str in {"ltxv", "ltx2"} and sd_key.endswith("keyframes_abs_pos_embedding") and len(shape) == 1:
            shape = torch.Size((1, shape[0]))

        # add to state dict
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
        state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)

        # 1D tensors shouldn't be quantized, this is a fix for BF16
        if len(shape) <= 1 and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            state_dict[sd_key] = dequantize_tensor(state_dict[sd_key], dtype=torch.float32)
        # PR #467: GGMLLayer only intercepts weight/bias, so bare nn.Parameters
        # (e.g. LTX-2 learnable_registers) must already be real float tensors.
        # Buffers stored as F16 (e.g. MiniMax-H3 adaln_t_table) need the same treatment:
        # nothing casts them at the use site and they meet float32 math there.
        elif not sd_key.endswith((".weight", ".bias")) and state_dict[sd_key].dtype != torch.float32:
            state_dict[sd_key] = dequantize_tensor(state_dict[sd_key], dtype=torch.float32)

        # keep track of loaded tensor types
        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    # print loaded tensor type counts
    logging.info("gguf qtypes: " + ", ".join(f"{k} ({v})" for k, v in qtype_dict.items()))

    # mark largest tensor for vram estimation
    qsd = {k:v for k,v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    # extra info to return
    extra = {
        "arch_str": arch_str,
        "metadata": get_gguf_metadata(reader)
    }
    return (state_dict, extra)

# for remapping llama.cpp -> original key names
T5_SD_MAP = {
    "enc.": "encoder.",
    ".blk.": ".block.",
    "token_embd": "shared",
    "output_norm": "final_layer_norm",
    "attn_q": "layer.0.SelfAttention.q",
    "attn_k": "layer.0.SelfAttention.k",
    "attn_v": "layer.0.SelfAttention.v",
    "attn_o": "layer.0.SelfAttention.o",
    "attn_norm": "layer.0.layer_norm",
    "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
    "ffn_up": "layer.1.DenseReluDense.wi_1",
    "ffn_down": "layer.1.DenseReluDense.wo",
    "ffn_gate": "layer.1.DenseReluDense.wi_0",
    "ffn_norm": "layer.1.layer_norm",
}

LLAMA_SD_MAP = {
    "blk.": "model.layers.",
    "attn_norm": "input_layernorm",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_v_norm.": "self_attn.v_norm.",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output.weight": "lm_head.weight",
}

GEMMA3_SD_MAP = LLAMA_SD_MAP.copy()
GEMMA3_SD_MAP.update({
    "ffn_norm": "pre_feedforward_layernorm",
    "post_ffw_norm": "post_feedforward_layernorm",
    "post_attention_norm": "post_attention_layernorm",
})

CLIP_VISION_SD_MAP = {
    "mm.": "visual.merger.mlp.",
    "v.post_ln.": "visual.merger.ln_q.",
    "v.patch_embd": "visual.patch_embed.proj",
    "v.blk.": "visual.blocks.",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "attn_out.": "attn.proj.",
    "ln1.": "norm1.",
    "ln2.": "norm2.",
}

# Qwen3-VL deepstack mmproj (PR #473 MiniMax-H3 / Qwen3VL TE)
CLIP_VISION_QWEN3_MAP = {
    "v.blk": "model.visual.blocks",
    ".fc": ".linear_fc",
    "ck.8.": "st.0.",
    "ck.16.": "st.1.",
    "ck.24.": "st.2.",
    "ck.5.": "st.0.",
    "ck.11.": "st.1.",
    "ck.17.": "st.2.",
    "attn_out": "attn.proj",
    "ln1": "norm1",
    "ln2": "norm2",
    "attn_qkv": "attn.qkv",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
    "mm.0": "model.visual.merger.linear_fc1",
    "mm.2": "model.visual.merger.linear_fc2",
    "v.post_ln": "model.visual.merger.norm",
    "v.patch_embd": "model.visual.patch_embed.proj",
    "v.position_embd.weight": "visual.pos_embed.weight",
    "v.deepstast.": "model.visual.deepstack_merger_list.",
}

def sd_map_replace(raw_sd, key_map):
    sd = {}
    for k,v in raw_sd.items():
        for s,d in key_map.items():
            k = k.replace(s,d)
        sd[k] = v
    return sd

def llama_head_counts(metadata, arch_str, default_head=32, default_head_kv=8):
    """Read the real attention head counts for the Q/K un-permute.

    A wrong count does not raise: several head counts divide the same hidden
    size (28 and 32 both divide 3584), so the wrong permutation applies silently
    and the model emits fluent nonsense. Always prefer what the file says.
    """
    n_head = metadata.get(f"{arch_str}.attention.head_count")
    n_head_kv = metadata.get(f"{arch_str}.attention.head_count_kv")
    if n_head is None or n_head_kv is None:
        logging.warning(
            f"GGUF {arch_str} file has no attention head counts; assuming "
            f"{default_head}/{default_head_kv} for the Q/K un-permute.")
    return int(n_head or default_head), int(n_head_kv or default_head_kv)

def llama_permute(raw_sd, n_head, n_head_kv):
    # Reverse version of LlamaModel.permute in llama.cpp convert script
    sd = {}
    permute = lambda x,h: x.reshape(h, x.shape[0] // h // 2, 2, *x.shape[1:]).swapaxes(1, 2).reshape(x.shape)
    for k,v in raw_sd.items():
        if k.endswith(("q_proj.weight", "q_proj.bias")):
            v.data = permute(v.data, n_head)
        if k.endswith(("k_proj.weight", "k_proj.bias")):
            v.data = permute(v.data, n_head_kv)
        sd[k] = v
    return sd

def gemma3_norm_corrections(sd):
    # Reverse change from Gemma3Model modify_tensors in llama.cpp convert script
    norm_patterns = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "model.norm.weight"
    ]
    corrected = 0
    for key in list(sd.keys()):
        if any(p in key for p in norm_patterns):
            if is_quantized(sd[key]):
                sd[key] = dequantize_tensor(sd[key], dtype=torch.float32) - 1.0
            else:
                sd[key] = sd[key].float() - 1.0
            corrected += 1
    #logging.info(f"Gemma3: Applied -1 norm correction to {corrected} tensors")
    return sd

def strip_quant_suffix(name):
    pattern = r"[-_]?(?:ud-)?i?q[0-9]_[a-z0-9_\-]{1,8}$"
    match = re.search(pattern, name, re.IGNORECASE)
    if match:
        name = name[:match.start()]
    return name

def squash_name(name):
    """Filename reduced to [a-z0-9], for punctuation-insensitive matching.

    A TE and its mmproj are rarely punctuated the same way by whoever quantized
    them: 'qwen3vl_8b_Q4_K_M.gguf' ships beside 'qwen3vl8b-mmproj-f16.gguf'. A
    plain substring test misses that pair, so the vision tower never loads and
    comfy detects the file as plain Qwen3-8B instead of Qwen3-VL.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())

def gguf_mmproj_loader(path):
    # Reverse version of Qwen2VLVisionModel.modify_tensors
    logging.info("Attenpting to find mmproj file for text encoder...")

    # get name to match w/o quant suffix
    tenc_fname = os.path.basename(path)
    tenc = os.path.splitext(tenc_fname)[0].lower()
    tenc = strip_quant_suffix(tenc)

    # try and find matching mmproj
    target = []
    root = os.path.dirname(path)
    for fname in os.listdir(root):
        name, ext = os.path.splitext(fname)
        if ext.lower() != ".gguf":
            continue
        if "mmproj" not in name.lower():
            continue
        if squash_name(tenc) in squash_name(name):
            target.append(fname)

    if len(target) == 0:
        logging.warning(f"Can't find mmproj file for '{tenc_fname}' (matching:'{tenc}'); vision path will not work.")
        return {}
    if len(target) > 1:
        logging.info(f"Ambiguous mmproj for text encoder '{tenc_fname}', will use first match.")

    logging.info(f"Using mmproj '{target[0]}' for text encoder '{tenc_fname}'.")
    return load_mmproj_sd(os.path.join(root, target[0]))

def load_mmproj_sd(path):
    """Map an mmproj GGUF to the vision-tower keys comfy's VL TEs expect.

    Split out from `gguf_mmproj_loader` so a user who picks the mmproj in a CLIP
    slot explicitly gets the same state dict the sibling auto-detection builds.
    """
    vsd, _ = gguf_sd_loader(path, is_text_model=True)
    return map_mmproj_sd(vsd)

def map_mmproj_sd(vsd):
    """mmproj GGUF keys -> comfy vision-tower keys (`visual.*`)."""
    # concat 4D to 5D
    if "v.patch_embd.weight.1" in vsd:
        w1 = dequantize_tensor(vsd.pop("v.patch_embd.weight"), dtype=torch.float32)
        w2 = dequantize_tensor(vsd.pop("v.patch_embd.weight.1"), dtype=torch.float32)
        vsd["v.patch_embd.weight"] = torch.stack([w1, w2], dim=2)

    # Qwen3-VL deepstack mmproj (MiniMax-H3 TE path) — PR #473
    if any("deepstack" in key or "deepstast" in key for key in vsd):
        return sd_map_replace(vsd, CLIP_VISION_QWEN3_MAP)

    # Qwen2-VL / default clip-vision mmproj
    vsd = sd_map_replace(vsd, CLIP_VISION_SD_MAP)

    # handle split Q/K/V
    if "visual.blocks.0.attn_q.weight" in vsd:
        attns = {}
        # filter out attentions + group
        for k,v in vsd.items():
            if any(x in k for x in ["attn_q", "attn_k", "attn_v"]):
                k_attn, k_name = k.rsplit(".attn_", 1)
                k_attn += ".attn.qkv." + k_name.split(".")[-1]
                if k_attn not in attns:
                    attns[k_attn] = {}
                attns[k_attn][k_name] = dequantize_tensor(
                    v, dtype=(torch.bfloat16 if is_quantized(v) else torch.float16)
                )

        # recombine
        for k,v in attns.items():
            suffix = k.split(".")[-1]
            vsd[k] = torch.cat([
                v[f"q.{suffix}"],
                v[f"k.{suffix}"],
                v[f"v.{suffix}"],
            ], dim=0)
            # remove the consumed per-q/k/v tensors: leaving them in produces
            # hundreds of bogus "unexpected keys" in comfy's load report and
            # would break any strict consumer of the merged sd.
            for side in ("q", "k", "v"):
                block = k[: -len(".attn.qkv." + suffix)]
                vsd.pop(f"{block}.attn_{side}.{suffix}", None)
        del attns

    # comfy's Qwen2VLVisionTransformer positions via rope and has no pos_embed
    # parameter; the mmproj's position table would arrive as an unexpected key.
    vsd.pop("v.position_embd.weight", None)

    return vsd

def gguf_tokenizer_loader(path, temb_shape):
    # convert gguf tokenizer to spiece
    logging.info("Attempting to recreate sentencepiece tokenizer from GGUF file metadata...")
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError("Please make sure sentencepiece and protobuf are installed.\npip install sentencepiece protobuf")
    spm = model.ModelProto()

    reader = gguf.GGUFReader(path)

    if get_field(reader, "tokenizer.ggml.model", str) == "t5":
        if temb_shape == (256384, 4096): # probably UMT5
            spm.trainer_spec.model_type == 1 # Unigram (do we have a T5 w/ BPE?)
        else:
            raise NotImplementedError("Unknown model, can't set tokenizer!")
    else:
        raise NotImplementedError("Unknown model, can't set tokenizer!")

    spm.normalizer_spec.add_dummy_prefix = get_field(reader, "tokenizer.ggml.add_space_prefix", bool)
    spm.normalizer_spec.remove_extra_whitespaces = get_field(reader, "tokenizer.ggml.remove_extra_whitespaces", bool)

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    scores = get_list_field(reader, "tokenizer.ggml.scores", float)
    toktypes = get_list_field(reader, "tokenizer.ggml.token_type", int)

    for idx, (token, score, toktype) in enumerate(zip(tokens, scores, toktypes)):
        # # These aren't present in the original?
        # if toktype == 5 and idx >= temb_shape[0]%1000):
        #     continue

        piece = spm.SentencePiece()
        piece.piece = token
        piece.score = score
        piece.type = toktype
        spm.pieces.append(piece)

    # unsure if any of these are correct
    spm.trainer_spec.byte_fallback = True
    spm.trainer_spec.vocab_size = len(tokens) # split off unused?
    spm.trainer_spec.max_sentence_length = 4096
    spm.trainer_spec.eos_id = get_field(reader, "tokenizer.ggml.eos_token_id", int)
    spm.trainer_spec.pad_id = get_field(reader, "tokenizer.ggml.padding_token_id", int)

    logging.info(f"Created tokenizer with vocab size of {len(spm.pieces)}")
    del reader
    return torch.ByteTensor(list(spm.SerializeToString()))

def gguf_tekken_tokenizer_loader(path, temb_shape):
    # convert ggml (hf) tokenizer metadata to tekken/comfy data
    logging.info("Attempting to recreate tekken tokenizer from GGUF file metadata...")
    import json
    import base64
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    reader = gguf.GGUFReader(path)

    model_str = get_field(reader, "tokenizer.ggml.model", str)
    if model_str == "gpt2":
        # PR #440: Ministral-3 3B uses (131072, 3072); larger Mistral uses 5120
        if temb_shape in {(131072, 5120), (131072, 3072)}:
            data = {
                "config": {"num_vocab_tokens": 150000, "default_vocab_size": 131072},
                "vocab": [],
                "special_tokens": [],
            }
        else:
            raise NotImplementedError("Unknown model, can't set tokenizer!")
    else:
        raise NotImplementedError("Unknown model, can't set tokenizer!")

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    toktypes = get_list_field(reader, "tokenizer.ggml.token_type", int)

    decoder = {v: k for k, v in bytes_to_unicode().items()}
    for idx, (token, toktype) in enumerate(zip(tokens, toktypes)):
        if toktype == 3:
            data["special_tokens"].append(
                {'rank': idx, 'token_str': token, 'is_control': True}
            )
        else:
            tok = bytes([decoder[char] for char in token])
            data["vocab"].append({
                "rank": len(data["vocab"]),
                "token_bytes": base64.b64encode(tok).decode("ascii"),
                "token_str": tok.decode("utf-8", errors="replace") # ?
            })

    logging.info(f"Created tekken tokenizer with vocab size of {len(data['vocab'])} (+{len(data['special_tokens'])})")
    del reader
    return torch.ByteTensor(list(json.dumps(data).encode('utf-8')))

def gguf_gemma3_tokenizer_loader(path):
    #TODO: merge into gguf_tokenizer_loader
    logging.info("Attempting to recreate sentencepiece tokenizer from GGUF file metadata...")
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError("Please install sentencepiece and protobuf.\npip install sentencepiece protobuf")

    # Building the proto appends 262k pieces one at a time (~20s). The result
    # is deterministic for a given file, so cache the serialized model next to
    # the GGUF and reuse it.
    cache_path = path + ".spiece_cache.bin"
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as f:
                data = f.read()
            if len(data) > 1_000_000:  # sanity: a real Gemma-3 proto is ~5MB
                logging.info("Using cached sentencepiece tokenizer: %s", cache_path)
                return torch.ByteTensor(list(data))
            logging.warning("Ignoring suspiciously small tokenizer cache, rebuilding.")
        except OSError:
            pass  # unreadable cache -> rebuild below

    spm = model.ModelProto()
    reader = gguf.GGUFReader(path)

    spm.normalizer_spec.name = "identity"
    spm.normalizer_spec.add_dummy_prefix = False
    spm.trainer_spec.model_type = 2
    spm.trainer_spec.input_format = "tsv"
    spm.trainer_spec.byte_fallback = True
    spm.trainer_spec.max_sentence_length = 4192
    spm.trainer_spec.bos_piece = "<bos>"

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    scores = get_list_field(reader, "tokenizer.ggml.scores", float)
    toktype = get_list_field(reader, "tokenizer.ggml.token_type", int)
    
    if not tokens or not scores or not toktype:
        raise ValueError("Missing tokenizer metadata")
    
    for idx in range(len(tokens)):
        piece = spm.SentencePiece()
        piece.piece = tokens[idx]
        if idx == 3:  # UNK position
            piece.type = 2  # UNK Token
            piece.score = 0.0 # UNK Score
        else:
            piece.type = toktype[idx]
            piece.score = scores[idx]
        spm.pieces.append(piece)
    
    spm.trainer_spec.vocab_size = len(spm.pieces)
    logging.info(f"Created tokenizer with vocab size of {len(spm.pieces)}")

    del reader
    data = spm.SerializeToString()
    try:
        with open(cache_path, "wb") as f:
            f.write(data)
    except OSError:
        pass  # read-only model dir -> just don't cache
    return torch.ByteTensor(list(data))

def gguf_gemma4_tokenizer_loader(path):
    """The HF tokenizer.json document for a Gemma-4 text encoder GGUF.

    Comfy's Gemma-4 tokenizers consume the full `tokenizers`-library JSON
    (BPE vocab + merges, pretokenizer, added tokens). GGUF has no field for
    that document and it cannot be rebuilt from the sentencepiece-style
    metadata the way Gemma-3's can, so it rides in a sidecar file next to the
    GGUF: `<gguf name>.tokenizer.json` (e.g. extracted from the matching
    comfy safetensors' `tokenizer_json` entry, or the model's HF repo).
    """
    import json
    sidecar = path + ".tokenizer.json"
    if not os.path.isfile(sidecar):
        raise ValueError(
            f"'{os.path.basename(path)}' is a Gemma-4 text encoder: its "
            "tokenizer needs the HF tokenizer.json, which the GGUF format "
            "cannot carry. Put it next to the GGUF as "
            f"'{os.path.basename(path)}.tokenizer.json' — extract the "
            "'tokenizer_json' entry from the matching comfy safetensors "
            "checkpoint, or download tokenizer.json from the model's HF repo."
        )
    with open(sidecar, "rb") as f:
        data = f.read()
    doc = json.loads(data.decode("utf-8"))
    vocab = len(doc.get("model", {}).get("vocab", {}))
    reader = gguf.GGUFReader(path)
    gguf_tokens = len(get_list_field(reader, "tokenizer.ggml.tokens", str))
    del reader
    if vocab != gguf_tokens:
        raise ValueError(
            f"The tokenizer sidecar {os.path.basename(sidecar)} has {vocab} "
            f"vocab entries but the GGUF has {gguf_tokens} tokens — it belongs "
            "to a different model and would silently mistokenize every prompt."
        )
    logging.info("Using Gemma-4 tokenizer sidecar: %s (%d tokens)",
                 sidecar, vocab)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()

def _merge_gemma4_fixups(path, sd):
    """Merge tiny non-quantizable buffers from `<gguf>.fixup.safetensors`.

    Gemma-4 multiplies every block output by a per-layer learned scalar held
    in a `torch.empty` buffer (comfy/text_encoders/gemma4.py). Quantizers drop
    these 1-element tensors because they are not 2-D weights, and comfy only
    logs 'clip missing' — the model then runs on uninitialized garbage. The
    sidecar carries them (comfy-layout keys; extract from the source
    checkpoint), and a missing scalar without a sidecar is a hard error.
    """
    sidecar = path + ".fixup.safetensors"
    if os.path.isfile(sidecar):
        import safetensors.torch
        fix = safetensors.torch.load_file(sidecar)
        merged = [k for k in fix if k not in sd]
        for k in merged:
            sd[k] = fix[k].float()
        logging.info("Merged %d missing Gemma-4 tensors from %s", len(merged), sidecar)
    missing = sum(1 for k in sd if k.endswith(".layer_scalar"))
    if missing == 0:
        raise ValueError(
            f"'{os.path.basename(path)}' carries no model.layers.*.layer_scalar "
            "tensors. Gemma-4 scales every block output by these learned "
            "per-layer constants; without them the text encoder runs on "
            "uninitialized memory and every prompt becomes garbage. Extract "
            "them from the matching comfy safetensors checkpoint into "
            f"'{os.path.basename(path)}.fixup.safetensors' (keys "
            "model.layers.N.layer_scalar)."
        )
    return sd

def gguf_clip_loader(path):
    sd, extra = gguf_sd_loader(path, is_text_model=True)
    arch = extra.get("arch_str", None)
    if arch in VISION_PROJECTOR_ARCHES:
        # mmproj picked directly: hand back the mapped vision tower so the
        # caller can merge it into the TE it belongs to.
        return map_mmproj_sd(sd)
    if arch in {"t5", "t5encoder"}:
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape == (256384, 4096):
            # non-standard Comfy-Org tokenizer
            sd["spiece_model"] = gguf_tokenizer_loader(path, sd[temb_key].shape)
            # TODO: dequantizing token embed here is janky but otherwise we OOM due to tensor being massive.
            logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        sd = sd_map_replace(sd, T5_SD_MAP)
    elif arch in {"llama", "qwen2", "qwen2vl", "qwen3", "qwen35", "qwen3vl", "gemma3", "gemma4", "gemma4_unified", "mistral3"}:
        # TODO: pass model_options["vocab_size"] to loader somehow
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape[0] >= (64 * 1024):
            if arch in {"llama", "mistral3"} and sd[temb_key].shape in {(131072, 5120), (131072, 3072)}:
                # non-standard Comfy-Org tokenizer
                sd["tekken_model"] = gguf_tekken_tokenizer_loader(path, sd[temb_key].shape)
            elif arch == "gemma3":
                sd["spiece_model"] = gguf_gemma3_tokenizer_loader(path)
            # See note above for T5.
            logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        if arch in {"gemma3", "gemma4", "gemma4_unified"}:
            if arch != "gemma3" and "tokenizer_json" not in sd:
                # Gemma-4: comfy's tokenizer needs the HF tokenizer.json (see
                # gguf_gemma4_tokenizer_loader); without it SDTokenizer falls
                # back to a path string and crashes on .decode().
                sd["tokenizer_json"] = gguf_gemma4_tokenizer_loader(path)
            sd = sd_map_replace(sd, GEMMA3_SD_MAP)
            sd = gemma3_norm_corrections(sd)
            if arch != "gemma3":
                sd = _merge_gemma4_fixups(path, sd)
        else:
            sd = sd_map_replace(sd, LLAMA_SD_MAP)
        if arch in {"llama", "mistral3", "qwen2"}:
            # L3 / Mistral / qwen2-compat. Head counts come from the file: the
            # 32/8 that fits L3-8B silently corrupts anything shaped otherwise.
            sd = llama_permute(sd, *llama_head_counts(extra.get("metadata", {}), arch))
        if arch in {"qwen2vl", "qwen3vl"}:
            vsd = gguf_mmproj_loader(path)
            sd.update(vsd)
    else:
        pass
    return sd
