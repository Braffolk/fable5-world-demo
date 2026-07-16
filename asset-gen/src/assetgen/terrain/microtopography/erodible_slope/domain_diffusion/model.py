"""Compact pixel-space diffusion used by the bounded R0 challenger."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DiffusionModel:
    state: dict[str, Any]
    condition_mean: np.ndarray
    condition_std: np.ndarray
    output_scale_m: float
    losses: list[float]
    parameter_count: int


def _cosine_schedule(torch: Any, steps: int = 1000) -> Any:
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float32)
    alpha = torch.cos(((t / steps + 0.008) / 1.008) * math.pi / 2.0) ** 2
    alpha = alpha / alpha[0]
    return torch.clamp(alpha[1:], 1.0e-5, 0.999999)


def _network(torch: Any, condition_channels: int, width: int):
    nn = torch.nn

    class TimeEmbedding(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.channels = channels
            self.project = nn.Sequential(nn.Linear(channels, channels * 2), nn.SiLU(), nn.Linear(channels * 2, channels * 2))

        def forward(self, timestep):
            half = self.channels // 2
            frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, device=timestep.device) / max(half - 1, 1))
            angles = timestep[:, None].float() * frequencies[None]
            return self.project(torch.cat((torch.sin(angles), torch.cos(angles)), dim=1))

    class Block(nn.Module):
        def __init__(self, input_channels: int, output_channels: int, time_channels: int):
            super().__init__()
            groups = min(8, output_channels)
            self.first = nn.Sequential(nn.GroupNorm(min(8, input_channels), input_channels), nn.SiLU(), nn.Conv2d(input_channels, output_channels, 3, padding=1))
            self.second_norm = nn.GroupNorm(groups, output_channels)
            self.second = nn.Conv2d(output_channels, output_channels, 3, padding=1)
            self.time = nn.Linear(time_channels * 2, output_channels * 2)
            self.skip = nn.Conv2d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity()

        def forward(self, value, time):
            hidden = self.first(value)
            scale, shift = self.time(time).chunk(2, dim=1)
            hidden = self.second_norm(hidden) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
            hidden = self.second(torch.nn.functional.silu(hidden))
            return hidden + self.skip(value)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.time = TimeEmbedding(width * 4)
            self.stem = nn.Conv2d(1 + condition_channels, width, 3, padding=1)
            self.b0 = Block(width, width, width * 4)
            self.down1 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
            self.b1 = Block(width * 2, width * 2, width * 4)
            self.down2 = nn.Conv2d(width * 2, width * 3, 4, stride=2, padding=1)
            self.mid1 = Block(width * 3, width * 3, width * 4)
            self.mid2 = Block(width * 3, width * 3, width * 4)
            self.up1 = nn.Conv2d(width * 3 + width * 2, width * 2, 3, padding=1)
            self.ub1 = Block(width * 2, width * 2, width * 4)
            self.up0 = nn.Conv2d(width * 2 + width, width, 3, padding=1)
            self.ub0 = Block(width, width, width * 4)
            self.out = nn.Sequential(nn.GroupNorm(min(8, width), width), nn.SiLU(), nn.Conv2d(width, 1, 3, padding=1))

        def forward(self, value, condition, timestep):
            time = self.time(timestep)
            level0 = self.b0(self.stem(torch.cat((value, condition), dim=1)), time)
            level1 = self.b1(self.down1(level0), time)
            hidden = self.mid2(self.mid1(self.down2(level1), time), time)
            hidden = torch.nn.functional.interpolate(hidden, size=level1.shape[-2:], mode="bilinear", align_corners=False)
            hidden = self.ub1(self.up1(torch.cat((hidden, level1), dim=1)), time)
            hidden = torch.nn.functional.interpolate(hidden, size=level0.shape[-2:], mode="bilinear", align_corners=False)
            hidden = self.ub0(self.up0(torch.cat((hidden, level0), dim=1)), time)
            return self.out(hidden)

    return UNet()


def _derivatives(torch: Any, value: Any) -> tuple[Any, Any, Any]:
    gx = value[..., :, 1:] - value[..., :, :-1]
    gy = value[..., 1:, :] - value[..., :-1, :]
    lap = -4.0 * value
    lap = lap + torch.roll(value, 1, -1) + torch.roll(value, -1, -1) + torch.roll(value, 1, -2) + torch.roll(value, -1, -2)
    return gx, gy, lap


def train(
    samples: np.ndarray,
    conditions: np.ndarray,
    masks: np.ndarray,
    *,
    iterations: int,
    batch_size: int,
    width: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> DiffusionModel:
    import torch

    generator = np.random.default_rng(seed)
    valid_values = samples[:, 0][masks]
    output_scale = max(float(np.quantile(np.abs(valid_values), 0.95)), 0.01)
    channel_values = np.moveaxis(conditions, 1, -1)[masks]
    condition_mean = np.mean(channel_values, axis=0).astype(np.float32)
    condition_std = np.maximum(np.std(channel_values, axis=0), 1.0e-4).astype(np.float32)
    normalized_conditions = np.clip((conditions - condition_mean[None, :, None, None]) / condition_std[None, :, None, None], -6.0, 6.0)
    x0 = (samples / output_scale).astype(np.float32)

    model = _network(torch, conditions.shape[1], width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1.0e-4)
    alpha = _cosine_schedule(torch).to(device)
    losses: list[float] = []
    sample_count = samples.shape[0]
    torch.manual_seed(seed)

    for iteration in range(iterations):
        indices = generator.integers(0, sample_count, size=batch_size)
        rotations = generator.integers(0, 8, size=batch_size)
        batch_x, batch_c, batch_m = [], [], []
        for index, transform in zip(indices, rotations):
            value, condition, mask = x0[index], normalized_conditions[index], masks[index]
            k = int(transform & 3)
            value, condition, mask = np.rot90(value, k, (-2, -1)), np.rot90(condition, k, (-2, -1)), np.rot90(mask, k, (-2, -1))
            if transform >= 4:
                value, condition, mask = value[..., ::-1].copy(), condition[..., ::-1].copy(), mask[..., ::-1].copy()
            batch_x.append(value.copy())
            batch_c.append(condition.copy())
            batch_m.append(mask.copy())
        clean = torch.from_numpy(np.stack(batch_x)).to(device)
        condition = torch.from_numpy(np.stack(batch_c)).to(device)
        mask = torch.from_numpy(np.stack(batch_m)[:, None].astype(np.float32)).to(device)
        timestep = torch.from_numpy(generator.integers(0, 1000, size=batch_size, dtype=np.int64)).to(device)
        a = alpha[timestep][:, None, None, None]
        noise = torch.randn_like(clean)
        noisy = torch.sqrt(a) * clean + torch.sqrt(1.0 - a) * noise
        target_v = torch.sqrt(a) * noise - torch.sqrt(1.0 - a) * clean
        predicted_v = model(noisy, condition, timestep)
        prediction = torch.sqrt(a) * noisy - torch.sqrt(1.0 - a) * predicted_v
        denominator = torch.clamp(mask.sum(), min=1.0)
        loss = (((predicted_v - target_v) ** 2) * mask).sum() / denominator
        pgx, pgy, plap = _derivatives(torch, prediction)
        tgx, tgy, tlap = _derivatives(torch, clean)
        derivative = torch.nn.functional.l1_loss(pgx, tgx) + torch.nn.functional.l1_loss(pgy, tgy)
        curvature = torch.nn.functional.l1_loss(plap, tlap)
        loss = loss + 0.10 * derivative + 0.05 * curvature
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if iteration % 100 == 0 or iteration == iterations - 1:
            losses.append(float(loss.detach().cpu()))

    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    return DiffusionModel(
        state=state,
        condition_mean=condition_mean,
        condition_std=condition_std,
        output_scale_m=output_scale,
        losses=losses,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def sample(
    trained: DiffusionModel,
    conditions: np.ndarray,
    *,
    width: int,
    seed: int,
    ddim_steps: int,
    device: str,
) -> np.ndarray:
    import torch

    condition = np.clip(
        (conditions - trained.condition_mean[:, None, None]) / trained.condition_std[:, None, None],
        -6.0,
        6.0,
    ).astype(np.float32)
    original_shape = condition.shape[-2:]
    pad_y = (-original_shape[0]) % 4
    pad_x = (-original_shape[1]) % 4
    condition = np.pad(condition, ((0, 0), (0, pad_y), (0, pad_x)), mode="reflect")
    model = _network(torch, condition.shape[0], width)
    model.load_state_dict(trained.state)
    model = model.to(device).eval()
    condition_tensor = torch.from_numpy(condition[None]).to(device)
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn((1, 1, *condition.shape[-2:]), generator=cpu_generator).to(device)
    alpha = _cosine_schedule(torch).to(device)
    times = np.unique(np.rint(np.linspace(999, 0, ddim_steps)).astype(np.int64))[::-1]
    with torch.no_grad():
        for index, timestep_value in enumerate(times):
            timestep = torch.full((1,), int(timestep_value), dtype=torch.long, device=device)
            a = alpha[timestep_value]
            predicted_v = model(value, condition_tensor, timestep)
            x0 = torch.sqrt(a) * value - torch.sqrt(1.0 - a) * predicted_v
            epsilon = torch.sqrt(1.0 - a) * value + torch.sqrt(a) * predicted_v
            if index == len(times) - 1:
                value = x0
            else:
                next_a = alpha[int(times[index + 1])]
                value = torch.sqrt(next_a) * x0 + torch.sqrt(1.0 - next_a) * epsilon
    return value[0, 0, : original_shape[0], : original_shape[1]].cpu().numpy().astype(np.float64) * trained.output_scale_m
