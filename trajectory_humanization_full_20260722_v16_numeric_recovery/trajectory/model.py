"""Action-specific conditional DDPM for human-like screen trajectories.

This module contains a genuine neural generator.  It does not select, replay,
warp, or perturb a stored trajectory template.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .constraints import ConstrainedTrajectory, constrain_and_decode
from .data import (
    ACTIONS,
    FEATURE_DIM,
    KEYCODE_VOCAB_SIZE,
    MAX_POINTERS,
    ORIENTATION_IDS,
    TrajectoryBatch,
)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    if timesteps.dim() != 1:
        raise ValueError("timesteps must have shape [B]")
    half = dim // 2
    if half == 0:
        return timesteps.float().unsqueeze(-1)
    exponent = -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32)
    exponent = exponent / max(half - 1, 1)
    phase = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
    embedding = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConditionEncoder(nn.Module):
    """Encode all caller-visible physical and semantic conditions."""

    def __init__(
        self,
        action: str,
        cond_dim: int,
        keycode_vocab: int = KEYCODE_VOCAB_SIZE,
        keycode_dim: int = 32,
        screen_scale_px: float = 1000.0,
    ):
        super().__init__()
        if action not in ACTIONS:
            raise ValueError("unsupported action: %r" % action)
        if keycode_vocab <= 1:
            raise ValueError("keycode_vocab must be > 1")
        self.action = action
        self.cond_dim = int(cond_dim)
        self.keycode_vocab = int(keycode_vocab)
        self.keycode_dim = int(keycode_dim)
        self.screen_scale_px = float(screen_scale_px)
        self.orientation_embedding = nn.Embedding(len(ORIENTATION_IDS), 8)
        self.keycode_embedding = nn.Embedding(keycode_vocab, keycode_dim)
        self.keycode_gru = nn.GRU(keycode_dim, keycode_dim, batch_first=True)
        # duration(1), start/end XY(8), pinch span(2), pinch angle sin/cos(4),
        # n_keys/n_letters(2), pointer count/relative lengths(3), and the two
        # pointers' global start/end lifetime fractions(4) = 24 values.
        numeric_dim = 24
        self.network = nn.Sequential(
            nn.Linear(numeric_dim + 8 + keycode_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def _orientation_indices(self, orientation_id: torch.Tensor) -> torch.Tensor:
        result = torch.full_like(orientation_id, -1)
        for index, value in enumerate(ORIENTATION_IDS):
            result = torch.where(orientation_id == value, torch.full_like(result, index), result)
        if torch.any(result < 0):
            raise ValueError("orientation_id must be one of %r" % (ORIENTATION_IDS,))
        return result

    def forward(self, batch: TrajectoryBatch) -> torch.Tensor:
        batch.validate()
        if batch.action != self.action:
            raise ValueError("model action %s cannot encode %s batch" % (self.action, batch.action))
        if torch.any(batch.keycodes < 0) or torch.any(batch.keycodes >= self.keycode_vocab):
            raise ValueError("keycode outside configured vocabulary [0,%d)" % self.keycode_vocab)

        orientation = self.orientation_embedding(self._orientation_indices(batch.orientation_id))
        key_input = self.keycode_embedding(batch.keycodes) * batch.keycode_mask.unsqueeze(-1).to(batch.features.dtype)
        key_sequence, _ = self.keycode_gru(key_input)
        key_lengths = batch.keycode_mask.sum(dim=1)
        last_index = torch.clamp(key_lengths - 1, min=0)
        key_embedding = key_sequence[torch.arange(key_sequence.shape[0], device=key_sequence.device), last_index]
        key_embedding = torch.where((key_lengths > 0).unsqueeze(-1), key_embedding, torch.zeros_like(key_embedding))

        lengths = batch.point_mask.sum(dim=-1).to(batch.features.dtype)
        max_length = torch.clamp(lengths.max(dim=1, keepdim=True).values, min=1.0)
        relative_lengths = lengths / max_length
        pointer_count = batch.pointer_mask.sum(dim=1, keepdim=True).to(batch.features.dtype) / float(MAX_POINTERS)
        duration_denominator = torch.clamp(batch.duration_ms, min=1.0).unsqueeze(-1)
        pointer_start_fraction = batch.pointer_start_offset_ms / duration_denominator
        pointer_end_fraction = batch.pointer_end_offset_ms / duration_denominator
        angle = batch.pinch_angle
        numeric = torch.cat(
            [
                torch.log1p(batch.duration_ms).unsqueeze(-1) / math.log(10001.0),
                batch.start_xy.reshape(batch.features.shape[0], -1) / self.screen_scale_px,
                batch.end_xy.reshape(batch.features.shape[0], -1) / self.screen_scale_px,
                torch.log1p(torch.clamp(batch.pinch_span, min=0.0)) / math.log(2001.0),
                torch.sin(angle),
                torch.cos(angle),
                torch.log1p(batch.n_keys.to(batch.features.dtype)).unsqueeze(-1) / math.log(101.0),
                torch.log1p(batch.n_letters.to(batch.features.dtype)).unsqueeze(-1) / math.log(101.0),
                pointer_count,
                relative_lengths,
                pointer_start_fraction,
                pointer_end_fraction,
            ],
            dim=-1,
        )
        return self.network(torch.cat([numeric, orientation, key_embedding], dim=-1))


class ReferenceSetEncoder(nn.Module):
    """Permutation-invariant style encoder over five masked real references."""

    def __init__(self, cond_dim: int, keycode_vocab: int = KEYCODE_VOCAB_SIZE, point_dim: int = 64, key_dim: int = 16):
        super().__init__()
        self.point_network = nn.Sequential(
            nn.Linear(FEATURE_DIM, point_dim), nn.SiLU(), nn.Linear(point_dim, point_dim), nn.SiLU()
        )
        self.key_embedding = nn.Embedding(keycode_vocab, key_dim)
        # Per-ref token: masked point embedding mean/std + contact fraction,
        # timeline occupancy and pointer count.
        self.reference_network = nn.Sequential(
            nn.Linear(point_dim * 2 + key_dim * 2 + 7, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        # DeepSets aggregation uses first and second moments, both invariant to
        # reference order.
        self.set_network = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

    def forward(self, batch: TrajectoryBatch) -> torch.Tensor:
        batch.validate(require_references=True)
        b, r, p, t, _ = batch.ref_features.shape
        feature_mask = batch.ref_feature_mask.to(batch.ref_features.dtype)
        point_valid = batch.ref_point_mask.unsqueeze(-1).to(batch.ref_features.dtype)
        point_embedding = self.point_network(batch.ref_features * feature_mask) * point_valid
        denominator = torch.clamp(point_valid.sum(dim=(2, 3)), min=1.0)
        mean = point_embedding.sum(dim=(2, 3)) / denominator
        second = (point_embedding ** 2).sum(dim=(2, 3)) / denominator
        std = torch.sqrt(torch.clamp(second - mean ** 2, min=0.0) + 1e-8)
        timeline_count = batch.ref_point_mask.sum(dim=(2, 3)).to(batch.ref_features.dtype)
        contact_count = batch.ref_contact_mask.sum(dim=(2, 3)).to(batch.ref_features.dtype)
        contact_fraction = contact_count / torch.clamp(timeline_count, min=1.0)
        # ``t`` is the maximum padded reference length of the *whole batch*.
        # Using it here made an otherwise identical reference set encode
        # differently merely because another sample in the batch was longer.
        # Normalize each reference by its own maximum pointer length instead.
        local_max_length = batch.ref_point_mask.sum(dim=-1).max(dim=2).values.to(
            batch.ref_features.dtype
        )
        occupancy = timeline_count / torch.clamp(
            local_max_length * float(MAX_POINTERS), min=1.0
        )
        pointer_fraction = batch.ref_pointer_mask.sum(dim=2).to(batch.ref_features.dtype) / float(MAX_POINTERS)
        ref_duration = torch.clamp(batch.ref_pointer_end_offset_ms.max(dim=-1).values, min=1.0).unsqueeze(-1)
        ref_start_fraction = batch.ref_pointer_start_offset_ms / ref_duration
        ref_end_fraction = batch.ref_pointer_end_offset_ms / ref_duration
        ref_key_mask = batch.ref_keycode_mask.unsqueeze(-1).to(batch.ref_features.dtype)
        ref_key_values = self.key_embedding(batch.ref_keycodes) * ref_key_mask
        ref_key_denominator = torch.clamp(ref_key_mask.sum(dim=2), min=1.0)
        ref_key_mean = ref_key_values.sum(dim=2) / ref_key_denominator
        ref_key_second = (ref_key_values ** 2).sum(dim=2) / ref_key_denominator
        ref_key_std = torch.sqrt(torch.clamp(ref_key_second - ref_key_mean ** 2, min=0.0) + 1e-8)
        token = self.reference_network(
            torch.cat(
                [
                    mean,
                    std,
                    ref_key_mean,
                    ref_key_std,
                    contact_fraction.unsqueeze(-1),
                    occupancy.unsqueeze(-1),
                    pointer_fraction.unsqueeze(-1),
                    ref_start_fraction,
                    ref_end_fraction,
                ],
                dim=-1,
            )
        )
        set_mask = batch.ref_mask.unsqueeze(-1).to(token.dtype)
        token = token * set_mask
        # Five-element reductions are accumulated in float64 so a mere ref
        # permutation does not create meaningful float32 condition drift.
        token64 = torch.sort(token.double(), dim=1).values
        set_mask64 = set_mask.double()
        set_count = torch.clamp(set_mask64.sum(dim=1), min=1.0)
        set_mean64 = token64.sum(dim=1) / set_count
        set_second64 = (token64 ** 2).sum(dim=1) / set_count
        set_std64 = torch.sqrt(torch.clamp(set_second64 - set_mean64 ** 2, min=0.0) + 1e-12)
        set_mean = set_mean64.to(token.dtype)
        set_std = set_std64.to(token.dtype)
        return self.set_network(torch.cat([set_mean, set_std], dim=-1))


class MaskedGroupNorm1d(nn.Module):
    """GroupNorm over channels and only the valid temporal positions.

    ``nn.GroupNorm`` includes right-padding in its moments.  Consequently a
    sample's valid output used to depend on the longest co-batched request.
    This implementation preserves the same learnable ``weight``/``bias``
    state-dict contract while excluding padding from both mean and variance.
    """

    def __init__(self, groups: int, channels: int, eps: float = 1.0e-5):
        super().__init__()
        if channels % groups:
            raise ValueError("channels must be divisible by groups")
        self.groups = int(groups)
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or valid_mask.shape != (x.shape[0], 1, x.shape[2]):
            raise ValueError("masked group norm expects x=[B,C,T], mask=[B,1,T]")
        batch, channels, points = x.shape
        if channels != self.channels:
            raise ValueError("masked group norm channel mismatch")
        channels_per_group = channels // self.groups
        # Float32 moments are stable under AMP.  Multiplication by the mask
        # also ensures invalid activations cannot feed a later convolution.
        values = x.float().reshape(batch, self.groups, channels_per_group, points)
        mask = valid_mask.to(dtype=torch.float32).reshape(batch, 1, 1, points)
        count = torch.clamp(
            mask.sum(dim=-1, keepdim=True) * float(channels_per_group), min=1.0
        )
        mean = (values * mask).sum(dim=(2, 3), keepdim=True) / count
        centered = (values - mean) * mask
        variance = (centered * centered).sum(dim=(2, 3), keepdim=True) / count
        normalized = centered * torch.rsqrt(variance + self.eps)
        normalized = normalized.reshape(batch, channels, points)
        output = (
            normalized * self.weight.float().reshape(1, channels, 1)
            + self.bias.float().reshape(1, channels, 1)
        )
        return output.to(dtype=x.dtype) * valid_mask.to(dtype=x.dtype)


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, dropout: float):
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = MaskedGroupNorm1d(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = MaskedGroupNorm1d(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.film = nn.Linear(cond_dim, channels * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, valid_mask)))
        h = h * valid_mask.to(dtype=h.dtype)
        scale, shift = self.film(condition).chunk(2, dim=-1)
        h = self.norm2(h, valid_mask) * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = h * valid_mask.to(dtype=h.dtype)
        h = self.conv2(self.dropout(F.silu(h))) * valid_mask.to(dtype=h.dtype)
        return (x + h) * valid_mask.to(dtype=x.dtype)


class TrajectoryDenoiser(nn.Module):
    def __init__(
        self,
        action: str,
        base_channels: int = 96,
        cond_dim: int = 192,
        time_dim: int = 96,
        n_blocks: int = 8,
        dropout: float = 0.05,
        keycode_vocab: int = KEYCODE_VOCAB_SIZE,
    ):
        super().__init__()
        if base_channels < 8 or n_blocks < 1:
            raise ValueError("invalid denoiser size")
        self.action = action
        self.time_dim = int(time_dim)
        self.condition_encoder = ConditionEncoder(action, cond_dim, keycode_vocab=keycode_vocab)
        self.reference_encoder = ReferenceSetEncoder(cond_dim, keycode_vocab=keycode_vocab)
        self.temporal_key_dim = 16
        self.temporal_key_projection = nn.Linear(self.condition_encoder.keycode_dim, self.temporal_key_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        input_channels = MAX_POINTERS * FEATURE_DIM + 3 * MAX_POINTERS + MAX_POINTERS * self.temporal_key_dim
        self.input_conv = nn.Conv1d(input_channels, base_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(base_channels, cond_dim, dropout) for _ in range(n_blocks)]
        )
        self.output_norm = MaskedGroupNorm1d(_group_count(base_channels), base_channels)
        self.output_conv = nn.Conv1d(base_channels, MAX_POINTERS * FEATURE_DIM, kernel_size=3, padding=1)

    def encode_condition(self, batch: TrajectoryBatch) -> torch.Tensor:
        batch.validate(require_references=True)
        return self.condition_encoder(batch) + self.reference_encoder(batch)

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        batch: TrajectoryBatch,
        encoded_metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tuple(x_t.shape) != tuple(batch.features.shape):
            raise ValueError("x_t shape must match batch.features")
        b, p, t, f = x_t.shape
        mask = batch.feature_mask.to(x_t.dtype)
        x_t = x_t * mask
        flattened = x_t.permute(0, 1, 3, 2).reshape(b, p * f, t)
        point_channels = batch.point_mask.to(x_t.dtype)
        contact_channels = batch.contact_mask.to(x_t.dtype)
        event_denominator = torch.clamp(batch.n_keys.to(x_t.dtype), min=1.0).view(b, 1, 1)
        event_channels = torch.clamp(batch.event_ids.to(x_t.dtype) + 1.0, min=0.0) / event_denominator
        key_tokens = self.temporal_key_projection(self.condition_encoder.keycode_embedding(batch.keycodes))
        key_count = key_tokens.shape[1]
        gather_index = torch.clamp(batch.event_ids, min=0, max=key_count - 1)
        expanded_keys = key_tokens[:, None, :, :].expand(b, p, key_count, self.temporal_key_dim)
        temporal_keys = torch.gather(
            expanded_keys,
            2,
            gather_index.unsqueeze(-1).expand(b, p, t, self.temporal_key_dim),
        )
        expanded_key_mask = batch.keycode_mask[:, None, :].expand(b, p, key_count)
        gathered_key_mask = torch.gather(expanded_key_mask, 2, gather_index)
        temporal_valid = batch.contact_mask & gathered_key_mask & (batch.event_ids >= 0)
        temporal_keys = temporal_keys * temporal_valid.unsqueeze(-1).to(x_t.dtype)
        temporal_key_channels = temporal_keys.permute(0, 1, 3, 2).reshape(b, p * self.temporal_key_dim, t)
        temporal_valid_mask = batch.point_mask.any(dim=1, keepdim=True).to(x_t.dtype)
        hidden = self.input_conv(
            torch.cat([flattened, point_channels, contact_channels, event_channels, temporal_key_channels], dim=1)
        ) * temporal_valid_mask
        if encoded_metadata is None:
            encoded_metadata = self.encode_condition(batch)
        condition = encoded_metadata + self.time_mlp(sinusoidal_embedding(timesteps, self.time_dim))
        for block in self.blocks:
            hidden = block(hidden, condition, temporal_valid_mask)
        output = self.output_conv(
            F.silu(self.output_norm(hidden, temporal_valid_mask))
        ) * temporal_valid_mask
        output = output.reshape(b, p, f, t).permute(0, 1, 3, 2)
        return output * mask


class TrajectoryDiffusion(nn.Module):
    """A separate conditional DDPM instance for exactly one action."""

    def __init__(
        self,
        action: str,
        diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        base_channels: int = 96,
        cond_dim: int = 192,
        time_dim: int = 96,
        n_blocks: int = 8,
        dropout: float = 0.05,
        keycode_vocab: int = KEYCODE_VOCAB_SIZE,
    ):
        super().__init__()
        if action not in ACTIONS:
            raise ValueError("unsupported action: %r" % action)
        if diffusion_steps < 2 or not (0.0 < beta_start < beta_end < 1.0):
            raise ValueError("invalid diffusion schedule")
        self.action = action
        self.diffusion_steps = int(diffusion_steps)
        self.denoiser = TrajectoryDenoiser(
            action=action,
            base_channels=base_channels,
            cond_dim=cond_dim,
            time_dim=time_dim,
            n_blocks=n_blocks,
            dropout=dropout,
            keycode_vocab=keycode_vocab,
        )

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_previous = torch.cat([torch.ones(1), alpha_bar[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("posterior_variance", posterior_variance)

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, ndim: int) -> torch.Tensor:
        result = values.gather(0, timesteps)
        return result.reshape(result.shape[0], *([1] * (ndim - 1)))

    def q_sample(self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = torch.sqrt(self._extract(self.alpha_bar, timesteps, clean.dim()))
        sqrt_one_minus = torch.sqrt(1.0 - self._extract(self.alpha_bar, timesteps, clean.dim()))
        return sqrt_alpha_bar * clean + sqrt_one_minus * noise

    def training_loss(
        self,
        batch: TrajectoryBatch,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch.validate(require_references=True)
        if batch.action != self.action:
            raise ValueError("model action %s cannot train on %s" % (self.action, batch.action))
        clean = batch.features
        b = clean.shape[0]
        if timesteps is None:
            timesteps = torch.randint(0, self.diffusion_steps, (b,), device=clean.device)
        if tuple(timesteps.shape) != (b,) or timesteps.dtype != torch.long:
            raise ValueError("timesteps must be int64 [B]")
        if torch.any(timesteps < 0) or torch.any(timesteps >= self.diffusion_steps):
            raise ValueError("timesteps outside diffusion schedule")
        if noise is None:
            noise = torch.randn_like(clean)
        if tuple(noise.shape) != tuple(clean.shape):
            raise ValueError("noise shape must match clean features")
        if not torch.all(torch.isfinite(noise)):
            raise ValueError("noise contains non-finite values")
        feature_mask = batch.feature_mask.to(clean.dtype)
        noise = noise * feature_mask
        noisy = self.q_sample(clean, timesteps, noise) * feature_mask
        predicted_noise = self.denoiser(noisy, timesteps, batch)
        squared_error = (predicted_noise - noise) ** 2 * feature_mask
        denominator = torch.clamp(feature_mask.sum(), min=1.0)
        loss = squared_error.sum() / denominator
        return {"loss": loss, "masked_epsilon_mse": loss, "valid_feature_count": denominator.detach()}

    @torch.no_grad()
    def sample(
        self,
        batch: TrajectoryBatch,
        generator: Optional[torch.Generator] = None,
    ) -> ConstrainedTrajectory:
        """Generate a new trajectory then enforce exact physical constraints."""
        batch.validate(require_references=True)
        if batch.action != self.action:
            raise ValueError("model action %s cannot sample %s" % (self.action, batch.action))
        was_training = self.training
        self.eval()
        try:
            mask = batch.feature_mask.to(batch.features.dtype)
            x = torch.randn(
                batch.features.shape,
                dtype=batch.features.dtype,
                device=batch.features.device,
                generator=generator,
            ) * mask
            # Metadata is invariant over reverse steps; encode it once rather
            # than repeating key/geometry encoding at every DDPM step.
            encoded_metadata = self.denoiser.encode_condition(batch)
            for step in reversed(range(self.diffusion_steps)):
                timesteps = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
                predicted_noise = self.denoiser(x, timesteps, batch, encoded_metadata=encoded_metadata)
                beta = self._extract(self.betas, timesteps, x.dim())
                alpha = self._extract(self.alphas, timesteps, x.dim())
                alpha_bar = self._extract(self.alpha_bar, timesteps, x.dim())
                mean = (x - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
                if step > 0:
                    noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
                    variance = self._extract(self.posterior_variance, timesteps, x.dim())
                    x = mean + torch.sqrt(torch.clamp(variance, min=1e-20)) * noise
                else:
                    x = mean
                x = x * mask
            return constrain_and_decode(x, batch)
        finally:
            self.train(was_training)

    def ddim_timesteps(self, inference_steps: int) -> torch.Tensor:
        """Return an auditable strict subsequence of the training schedule."""
        inference_steps = int(inference_steps)
        if inference_steps < 2 or inference_steps > self.diffusion_steps:
            raise ValueError("DDIM inference_steps must be in [2, diffusion_steps]")
        # Integer floor spacing includes exactly training steps 0 and N-1 and
        # is strictly increasing whenever inference_steps <= diffusion_steps.
        numerator = torch.arange(inference_steps, dtype=torch.long) * (self.diffusion_steps - 1)
        schedule = torch.div(numerator, inference_steps - 1, rounding_mode="floor")
        if schedule.numel() != inference_steps or torch.any(schedule[1:] <= schedule[:-1]):
            raise AssertionError("DDIM schedule construction produced duplicate/non-training steps")
        return schedule

    @torch.no_grad()
    def sample_ddim(
        self,
        batch: TrajectoryBatch,
        inference_steps: int = 50,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> ConstrainedTrajectory:
        """Generate with deterministic (eta=0) or stochastic DDIM.

        The denoiser is evaluated exactly ``inference_steps`` times on a strict
        subsequence of the trained DDPM schedule.  This is never a one-step x0
        reconstruction shortcut.
        """
        batch.validate(require_references=True)
        if batch.action != self.action:
            raise ValueError("model action %s cannot sample %s" % (self.action, batch.action))
        if not math.isfinite(float(eta)) or eta < 0.0:
            raise ValueError("DDIM eta must be finite and non-negative")
        schedule = self.ddim_timesteps(inference_steps).to(batch.features.device)
        was_training = self.training
        self.eval()
        try:
            mask = batch.feature_mask.to(batch.features.dtype)
            x = torch.randn(
                batch.features.shape,
                dtype=batch.features.dtype,
                device=batch.features.device,
                generator=generator,
            ) * mask
            encoded_condition = self.denoiser.encode_condition(batch)
            for schedule_index in reversed(range(schedule.numel())):
                step = int(schedule[schedule_index].item())
                timesteps = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
                predicted_noise = self.denoiser(x, timesteps, batch, encoded_metadata=encoded_condition)
                alpha_bar_t = self.alpha_bar[step].to(dtype=x.dtype, device=x.device)
                predicted_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
                if schedule_index == 0:
                    alpha_bar_previous = x.new_tensor(1.0)
                else:
                    previous_step = int(schedule[schedule_index - 1].item())
                    alpha_bar_previous = self.alpha_bar[previous_step].to(dtype=x.dtype, device=x.device)
                sigma = float(eta) * torch.sqrt(
                    torch.clamp(
                        (1.0 - alpha_bar_previous) / (1.0 - alpha_bar_t)
                        * (1.0 - alpha_bar_t / alpha_bar_previous),
                        min=0.0,
                    )
                )
                direction = torch.sqrt(torch.clamp(1.0 - alpha_bar_previous - sigma ** 2, min=0.0)) * predicted_noise
                x_previous = torch.sqrt(alpha_bar_previous) * predicted_x0 + direction
                if schedule_index > 0 and eta > 0.0:
                    noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
                    x_previous = x_previous + sigma * noise
                x = x_previous * mask
            return constrain_and_decode(x, batch)
        finally:
            self.train(was_training)
