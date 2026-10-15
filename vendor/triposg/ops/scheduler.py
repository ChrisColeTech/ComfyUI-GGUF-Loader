"""Pure-PyTorch rectified-flow scheduler used by TripoSG inference."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class SchedulerOutput:
    """Result of a scheduler step, compatible with Diffusers-style outputs."""

    prev_sample: torch.Tensor

    def to_tuple(self) -> tuple[torch.Tensor]:
        return (self.prev_sample,)

    def __getitem__(self, key: Union[int, slice, str]):
        if isinstance(key, str):
            if key != "prev_sample":
                raise KeyError(key)
            return self.prev_sample
        return self.to_tuple()[key]

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.to_tuple())

    def __len__(self) -> int:
        return 1


@dataclass(frozen=True)
class _SchedulerConfig(Mapping[str, object]):
    num_train_timesteps: int
    shift: float
    use_dynamic_shifting: bool

    def __getitem__(self, key: str) -> object:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(("num_train_timesteps", "shift", "use_dynamic_shifting"))

    def __len__(self) -> int:
        return 3


class RectifiedFlowScheduler:
    """TripoSG's rectified-flow Euler scheduler without Diffusers."""

    _compatibles: list[str] = []
    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
    ) -> None:
        self.config = _SchedulerConfig(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
        )

        # Match the reference's NumPy-to-float32 conversion before initial shifting.
        indices = torch.arange(num_train_timesteps, dtype=torch.float64)
        timesteps = ((1.0 - indices / num_train_timesteps) * num_train_timesteps).float()
        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            sigmas = self.time_shift(sigmas)

        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = sigmas.cpu()
        self.num_inference_steps: Optional[int] = None
        self._step_index: Optional[int] = None
        self._begin_index: Optional[int] = None

    @property
    def step_index(self) -> Optional[int]:
        return self._step_index

    @property
    def begin_index(self) -> Optional[int]:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.config.num_train_timesteps

    def _t_to_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        return timestep / self.config.num_train_timesteps

    def time_shift_dynamic(
        self, mu: float, sigma: float, t: torch.Tensor
    ) -> torch.Tensor:
        exp_mu = math.exp(mu)
        return exp_mu / (exp_mu + (1 / t - 1) ** sigma)

    def time_shift(self, t: torch.Tensor) -> torch.Tensor:
        shift = self.config.shift
        return shift * t / (1 + (shift - 1) * t)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        sigmas: Optional[Sequence[float] | torch.Tensor] = None,
        mu: Optional[float] = None,
    ) -> None:
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(
                " you have to pass a value for `mu` when "
                "`use_dynamic_shifting` is set to be `True`"
            )

        if sigmas is None:
            if num_inference_steps is None:
                raise ValueError("`num_inference_steps` must be provided when `sigmas` is None")
            indices = torch.arange(num_inference_steps, dtype=torch.float64)
            sigma_tensor = 1.0 - indices / num_inference_steps
            self.num_inference_steps = num_inference_steps
        else:
            # Conversion before shifting fixes the reference's Python-list arithmetic bug.
            sigma_tensor = torch.as_tensor(sigmas, dtype=torch.float64).clone()
            self.num_inference_steps = len(sigma_tensor)

        if self.config.use_dynamic_shifting:
            sigma_tensor = self.time_shift_dynamic(mu, 1.0, sigma_tensor)
        else:
            shift = self.config.shift
            sigma_tensor = shift * sigma_tensor / (1 + (shift - 1) * sigma_tensor)

        sigma_tensor = sigma_tensor.to(dtype=torch.float32, device=device)
        self.timesteps = self._sigma_to_t(sigma_tensor).to(device=device)
        self.sigmas = torch.cat(
            [sigma_tensor, torch.zeros(1, device=sigma_tensor.device)]
        )
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(
        self,
        timestep: Union[float, torch.Tensor],
        schedule_timesteps: Optional[torch.Tensor] = None,
    ) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        position = 1 if len(indices) > 1 else 0
        return indices[position].item()

    def _init_step_index(self, timestep: Union[float, torch.Tensor]) -> None:
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, tuple[torch.Tensor]]:
        del s_churn, s_tmin, s_tmax, s_noise, generator
        if isinstance(timestep, int) or (
            isinstance(timestep, torch.Tensor) and not timestep.dtype.is_floating_point
        ):
            raise ValueError(
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as "
                "timesteps to `EulerDiscreteScheduler.step()` is not supported. "
                "Make sure to pass one of the `scheduler.timesteps` as a timestep."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        prev_sample = sample + (sigma - sigma_next) * model_output
        prev_sample = prev_sample.to(model_output.dtype)
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    def scale_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sigmas = self._t_to_sigma(timesteps.to(dtype=torch.float32))
        while len(sigmas.shape) < len(original_samples.shape):
            sigmas = sigmas.unsqueeze(-1)
        return (1.0 - sigmas) * original_samples + sigmas * noise

    def __len__(self) -> int:
        return self.config.num_train_timesteps
