import torch

from vendor.melband_arch import MelBandRoformer, librosa_mel_fn


def test_band_structure_matches_the_public_checkpoint():
    # The mel bank decides band *membership*, not just weights: `bank > 0` sets
    # every layer's input width. A rounding difference at a band edge would
    # change the architecture and break the state dict, so pin the structure to
    # what MelBandRoformer_fp16 actually loads against.
    model = MelBandRoformer()
    assert len(model.band_split.to_features) == 60
    assert len(model.mask_estimators[0].to_freqs) == 60
    assert model.freq_indices.shape == (3958,)
    assert int(model.num_freqs_per_band.sum()) == 1979
    assert int(model.num_bands_per_freq.sum()) == 1979
    assert model.num_freqs_per_band[:8].tolist() == [7, 6, 6, 6, 6, 6, 6, 6]
    assert sum(p.numel() for p in model.parameters()) == 228202852


def test_every_frequency_belongs_to_a_band():
    bank = torch.from_numpy(librosa_mel_fn(sr=44100, n_fft=2048, n_mels=60))
    bank[0][0] = 1.0
    bank[-1, -1] = 1.0
    assert bool((bank > 0).any(dim=0).all())


def test_forward_is_shape_preserving_and_finite():
    torch.manual_seed(0)
    model = MelBandRoformer(dim=16, depth=1, dim_head=8, heads=2, num_bands=8).eval()
    audio = torch.randn(1, 2, 8192) * 0.1
    with torch.inference_mode():
        out = model(audio)
    assert out.shape[:2] == audio.shape[:2]
    # istft drops the trailing partial hop, so allow a short tail.
    assert 0 < audio.shape[-1] - out.shape[-1] <= 2048
    assert torch.isfinite(out).all()
