import torch

from seedvc_arch import CAMPPlus, InterpolateRegulator, SConv1d, sequence_mask


def test_sequence_mask_and_regulator_shapes():
    lengths = torch.tensor([11, 7])
    assert sequence_mask(lengths).sum().item() == 18
    regulator = InterpolateRegulator().eval()
    output = regulator(torch.randn(2, 5, 768), lengths)[0]
    assert output.shape == (2, 11, 512)
    assert output[1, 7:].abs().sum().item() == 0


def test_padding_wrapper_preserves_length():
    layer = SConv1d(4, 8, 5, norm="weight_norm").eval()
    assert layer(torch.randn(2, 4, 3)).shape == (2, 8, 3)


def test_campplus_final_embedding_does_not_clamp_identity():
    model = CAMPPlus()
    assert list(model.dense.nonlinear._modules) == ["batchnorm"]
