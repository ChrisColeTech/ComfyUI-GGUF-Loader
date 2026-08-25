# Apache-2.0. Adapted from KimberleyJensen/Mel-Band-Roformer-Vocal-Model and
# lucidrains/BS-RoFormer; mel filter bank follows librosa (ISC).
"""MelBandRoformer vocal/instrumental separation, checkpoint-compatible.

Only the inference path is ported. Module and parameter names match the public
``MelBandRoformer_fp16`` checkpoint exactly, including ``rotary_embed.freqs``,
so the state dict loads without remapping.
"""

from functools import partial

import numpy as np
import torch
from einops import pack, rearrange, reduce, repeat, unpack
from torch import nn
from torch.nn import functional as F

# ── librosa's slaney mel filter bank ──────────────────────────────────────
# Ported rather than taken from torchaudio: the bank decides *band membership*
# (`mel_filter_bank > 0`), so a rounding difference at a band edge would change
# the model's layer shapes, not just its numbers.


def _hz_to_mel(frequencies):
    frequencies = np.asanyarray(frequencies)
    f_sp = 200.0 / 3
    mels = frequencies / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    if frequencies.ndim:
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels):
    mels = np.asanyarray(mels)
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    if mels.ndim:
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return freqs


def librosa_mel_fn(*, sr, n_fft, n_mels=128, fmin=0.0, fmax=None, dtype=np.float32):
    fmax = float(sr) / 2 if fmax is None else fmax
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, np.newaxis]
    return weights


# ── rotary embedding ──────────────────────────────────────────────────────
# The checkpoint stores `rotary_embed.freqs`, so this keeps that parameter
# rather than recomputing the frequencies as a buffer.


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def rotate(self, t):
        seq_len = t.shape[-2]
        pos = torch.arange(seq_len, device=t.device, dtype=self.freqs.dtype)
        freqs = repeat(pos[:, None] * self.freqs[None, :], "... n -> ... (n r)", r=2)
        rotated = rearrange(t, "... (d r) -> ... d r", r=2)
        x1, x2 = rotated.unbind(dim=-1)
        rotated = rearrange(torch.stack((-x2, x1), dim=-1), "... d r -> ... (d r)")
        return (t * freqs.cos() + rotated * freqs.sin()).type(t.dtype)


# ── transformer ───────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim), nn.Linear(dim, inner), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(inner, dim), nn.Dropout(0.0))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, rotary_embed=None):
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.rotary_embed = rotary_embed
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(inner, dim, bias=False), nn.Dropout(0.0))

    def forward(self, x):
        x = self.norm(x)
        q, k, v = rearrange(self.to_qkv(x), "b n (qkv h d) -> qkv b h n d",
                            qkv=3, h=self.heads)
        if self.rotary_embed is not None:
            q, k = self.rotary_embed.rotate(q), self.rotary_embed.rotate(k)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out * rearrange(self.to_gates(x), "b n h -> b h n 1").sigmoid()
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class Transformer(nn.Module):
    def __init__(self, *, dim, depth, dim_head=64, heads=8, ff_mult=4,
                 norm_output=True, rotary_embed=None):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, rotary_embed=rotary_embed),
                FeedForward(dim=dim, mult=ff_mult),
            ]) for _ in range(depth)
        ])
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# ── band split / mask estimation ──────────────────────────────────────────


class BandSplit(nn.Module):
    def __init__(self, dim, dim_inputs):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList(
            [nn.Sequential(RMSNorm(d), nn.Linear(d, dim)) for d in dim_inputs])

    def forward(self, x):
        return torch.stack(
            [net(part) for part, net in zip(x.split(self.dim_inputs, dim=-1), self.to_features)],
            dim=-2)


def _mlp(dim_in, dim_out, dim_hidden, depth):
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)
    net = []
    for index, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        net.append(nn.Linear(a, b))
        if index != len(dims) - 2:
            net.append(nn.Tanh())
    return nn.Sequential(*net)


class MaskEstimator(nn.Module):
    def __init__(self, dim, dim_inputs, depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs
        hidden = dim * mlp_expansion_factor
        self.to_freqs = nn.ModuleList([
            nn.Sequential(_mlp(dim, d * 2, hidden, depth), nn.GLU(dim=-1))
            for d in dim_inputs])

    def forward(self, x):
        return torch.cat(
            [mlp(band) for band, mlp in zip(x.unbind(dim=-2), self.to_freqs)], dim=-1)


# ── model ─────────────────────────────────────────────────────────────────


class MelBandRoformer(nn.Module):
    def __init__(self, dim=384, *, depth=6, stereo=True, num_stems=1,
                 time_transformer_depth=1, freq_transformer_depth=1, num_bands=60,
                 dim_head=64, heads=8, sample_rate=44100, stft_n_fft=2048,
                 stft_hop_length=441, stft_win_length=2048, stft_normalized=False,
                 mask_estimator_depth=2, **_ignored):
        super().__init__()
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)
        kwargs = dict(dim=dim, heads=heads, dim_head=dim_head)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **kwargs),
                Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **kwargs),
            ]) for _ in range(depth)
        ])

        self.stft_window_fn = partial(torch.hann_window, stft_win_length)
        self.stft_kwargs = dict(n_fft=stft_n_fft, hop_length=stft_hop_length,
                                win_length=stft_win_length, normalized=stft_normalized)
        freqs = stft_n_fft // 2 + 1

        mel_filter_bank = torch.from_numpy(
            librosa_mel_fn(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands))
        # The bank omits the first and last bins; force them in so every
        # frequency belongs to at least one band.
        mel_filter_bank[0][0] = 1.0
        mel_filter_bank[-1, -1] = 1.0

        freqs_per_band = mel_filter_bank > 0
        if not freqs_per_band.any(dim=0).all():
            raise ValueError("every frequency must be covered by at least one band")

        freq_indices = repeat(torch.arange(freqs), "f -> b f", b=num_bands)[freqs_per_band]
        if stereo:
            freq_indices = repeat(freq_indices, "f -> f s", s=2) * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, "f s -> (f s)")

        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("freqs_per_band", freqs_per_band, persistent=False)
        num_freqs_per_band = reduce(freqs_per_band, "b f -> b", "sum")
        self.register_buffer("num_freqs_per_band", num_freqs_per_band, persistent=False)
        self.register_buffer("num_bands_per_freq",
                             reduce(freqs_per_band, "b f -> f", "sum"), persistent=False)

        dim_inputs = tuple(2 * f * self.audio_channels for f in num_freqs_per_band.tolist())
        self.band_split = BandSplit(dim=dim, dim_inputs=dim_inputs)
        self.mask_estimators = nn.ModuleList([
            MaskEstimator(dim=dim, dim_inputs=dim_inputs, depth=mask_estimator_depth)
            for _ in range(num_stems)])

    def forward(self, raw_audio):
        """``raw_audio`` [B, C, T] -> separated stem [B, C, T]."""
        device = raw_audio.device
        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")
        batch, channels, _ = raw_audio.shape
        if channels != self.audio_channels:
            raise ValueError(
                f"model expects {self.audio_channels} channel(s), got {channels}")

        packed, packed_shape = pack([raw_audio], "* t")
        window = self.stft_window_fn(device=device)
        stft_repr = torch.view_as_real(
            torch.stft(packed, **self.stft_kwargs, window=window, return_complex=True))
        stft_repr = unpack(stft_repr, packed_shape, "* f t c")[0]
        # Merge channels into the frequency axis so bands split across both.
        stft_repr = rearrange(stft_repr, "b s f t c -> b (f s) t c")

        batch_arange = torch.arange(batch, device=device)[..., None]
        x = stft_repr[batch_arange, self.freq_indices]
        x = self.band_split(rearrange(x, "b f t c -> b t (f c)"))

        for time_transformer, freq_transformer in self.layers:
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_transformer(x)
            x = unpack(x, ps, "* t d")[0]
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_transformer(x)
            x = unpack(x, ps, "* f d")[0]

        num_stems = len(self.mask_estimators)
        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)

        stft_repr = torch.view_as_complex(rearrange(stft_repr, "b f t c -> b 1 f t c"))
        masks = torch.view_as_complex(masks).type(stft_repr.dtype)

        # Bands overlap, so a frequency's mask is the mean of the bands covering it.
        scatter_indices = repeat(self.freq_indices, "f -> b n f t",
                                 b=batch, n=num_stems, t=stft_repr.shape[-1])
        expanded = repeat(stft_repr, "b 1 ... -> b n ...", n=num_stems)
        masks_summed = torch.zeros_like(expanded).scatter_add_(2, scatter_indices, masks)
        denom = repeat(self.num_bands_per_freq, "f -> (f r) 1", r=channels)
        stft_repr = stft_repr * (masks_summed / denom.clamp(min=1e-8))

        stft_repr = rearrange(stft_repr, "b n (f s) t -> (b n s) f t", s=self.audio_channels)
        recon = torch.istft(stft_repr, **self.stft_kwargs, window=window,
                            return_complex=False)
        recon = rearrange(recon, "(b n s) t -> b n s t",
                          b=batch, s=self.audio_channels, n=num_stems)
        return rearrange(recon, "b 1 s t -> b s t") if num_stems == 1 else recon
