"""Terrain-native conditional SinGAN-style height-field pyramid."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class TrainedPyramid:
    states: list[dict[str, Any]]
    noise_amplitudes: list[float]
    normalization_m: float
    scales: list[int]
    losses: list[dict[str, float]]
    parameter_count: int


def normalize_robust(value: np.ndarray, valid: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(value[valid], [3.0, 97.0])
    return np.clip((value - lo) / max(float(hi - lo), 1.0e-8), 0.0, 1.0)


def terrain_conditions(height_m: np.ndarray, network: np.ndarray | None, pitch_m: float) -> np.ndarray:
    smooth = ndimage.gaussian_filter(height_m.astype(np.float64), 2.0 / pitch_m)
    gy, gx = np.gradient(smooth, pitch_m)
    slope = np.hypot(gx, gy)
    curvature = ndimage.gaussian_laplace(smooth, 1.5 / pitch_m) / max(pitch_m * pitch_m, 1.0e-8)
    valid = np.isfinite(height_m)
    slope_n = normalize_robust(slope, valid)
    convex = normalize_robust(np.maximum(-curvature, 0.0), valid)
    concave = normalize_robust(np.maximum(curvature, 0.0), valid)
    if network is None:
        convergence = ndimage.gaussian_filter(concave * (0.35 + 0.65 * slope_n), 1.0 / pitch_m)
        network_n = normalize_robust(convergence, valid)
    else:
        network_n = normalize_robust(np.asarray(network, dtype=np.float64), valid)
    return np.stack((slope_n, convex, concave, network_n)).astype(np.float32)


def _features(torch: Any, height: Any) -> Any:
    """Metric-shape channels; no RGB feature network or natural-image prior."""
    import torch.nn.functional as functional

    dx = functional.pad(height[:, :, :, 1:] - height[:, :, :, :-1], (0, 1, 0, 0), mode="replicate")
    dy = functional.pad(height[:, :, 1:, :] - height[:, :, :-1, :], (0, 0, 0, 1), mode="replicate")
    lap = -4.0 * height
    lap = lap + functional.pad(height[:, :, :, 1:], (0, 1, 0, 0), mode="replicate")
    lap = lap + functional.pad(height[:, :, :, :-1], (1, 0, 0, 0), mode="replicate")
    lap = lap + functional.pad(height[:, :, 1:, :], (0, 0, 0, 1), mode="replicate")
    lap = lap + functional.pad(height[:, :, :-1, :], (0, 0, 1, 0), mode="replicate")
    low3 = functional.avg_pool2d(functional.pad(height, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    low7 = functional.avg_pool2d(functional.pad(height, (3, 3, 3, 3), mode="reflect"), 7, stride=1)
    return torch.cat((height, dx, dy, lap, height - low3, low3 - low7), dim=1)


def _networks(torch: Any, channels: int, width: int, layers: int) -> tuple[Any, Any]:
    nn = torch.nn

    class ConvBlock(nn.Sequential):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, padding_mode="reflect"),
                nn.GroupNorm(min(8, output_channels), output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

    class Generator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks: list[Any] = [ConvBlock(2 + channels, width)]
            blocks.extend(ConvBlock(width, width) for _ in range(layers - 2))
            self.body = nn.Sequential(*blocks)
            self.tail = nn.Conv2d(width, 1, 3, padding=1, padding_mode="reflect")

        def forward(self, previous: Any, noise: Any, condition: Any) -> Any:
            residual = 0.45 * torch.tanh(self.tail(self.body(torch.cat((previous, noise, condition), dim=1))))
            return previous + residual

    class Discriminator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks = [ConvBlock(6 + channels, width)]
            blocks.extend(ConvBlock(width, width) for _ in range(layers - 2))
            self.body = nn.Sequential(*blocks)
            self.tail = nn.Conv2d(width, 1, 3, padding=1, padding_mode="reflect")

        def forward(self, height: Any, condition: Any) -> Any:
            return self.tail(self.body(torch.cat((_features(torch, height), condition), dim=1)))

    generator, discriminator = Generator(), Discriminator()
    for network in (generator, discriminator):
        for module in network.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    return generator, discriminator


def _resize_tensor(functional: Any, value: Any, shape: tuple[int, int]) -> Any:
    return functional.interpolate(value, size=shape, mode="bilinear", align_corners=False)


def train_pyramid(
    source_height_m: np.ndarray,
    source_conditions: np.ndarray,
    config: dict[str, Any],
    device: str,
) -> TrainedPyramid:
    import torch
    import torch.nn.functional as functional

    seed = int(config["training_seed"])
    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.use_deterministic_algorithms(True)
    source = source_height_m.astype(np.float32)
    source -= float(np.mean(source))
    normalization = float(np.percentile(np.abs(source), 98.0))
    source = np.clip(source / max(normalization, 1.0e-6), -1.5, 1.5)
    source_tensor = torch.from_numpy(source[None, None]).to(device)
    condition_tensor = torch.from_numpy(source_conditions[None]).to(device)
    scales = [int(value) for value in config["pyramid_sizes"]]
    states: list[dict[str, Any]] = []
    amplitudes: list[float] = []
    losses: list[dict[str, float]] = []
    previous_real = None
    parameter_count = 0

    for scale_index, size in enumerate(scales):
        shape = (size, size)
        real = _resize_tensor(functional, source_tensor, shape)
        condition = _resize_tensor(functional, condition_tensor, shape)
        previous = torch.zeros_like(real) if previous_real is None else _resize_tensor(functional, previous_real, shape)
        rmse = float(torch.sqrt(torch.mean((real - previous) ** 2)).detach().cpu())
        amplitude = 1.0 if scale_index == 0 else float(config["noise_amplitude"]) * rmse
        generator, discriminator = _networks(torch, source_conditions.shape[0], int(config["channels"]), int(config["layers"]))
        generator, discriminator = generator.to(device), discriminator.to(device)
        parameter_count += sum(parameter.numel() for parameter in generator.parameters())
        optimizer_g = torch.optim.AdamW(generator.parameters(), lr=float(config["learning_rate"]), betas=(0.5, 0.999), weight_decay=1.0e-4)
        optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=float(config["learning_rate"]), betas=(0.5, 0.999), weight_decay=1.0e-4)
        fixed_generator = torch.Generator(device="cpu").manual_seed(seed + 1009 * scale_index)
        fixed_noise = torch.randn(real.shape, generator=fixed_generator).to(device)
        final_g = final_d = final_rec = 0.0

        for iteration in range(int(config["iterations_per_scale"])):
            noise_generator = torch.Generator(device="cpu").manual_seed(seed + scale_index * 1000003 + iteration)
            noise = torch.randn(real.shape, generator=noise_generator).to(device)
            fake = generator(previous, amplitude * noise, condition)
            optimizer_d.zero_grad(set_to_none=True)
            real_score = discriminator(real, condition)
            fake_score = discriminator(fake.detach(), condition)
            loss_d = torch.relu(1.0 - real_score).mean() + torch.relu(1.0 + fake_score).mean()
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            fake = generator(previous, amplitude * noise, condition)
            adversarial = -discriminator(fake, condition).mean()
            reconstructed = generator(previous, amplitude * fixed_noise if scale_index == 0 else torch.zeros_like(fixed_noise), condition)
            height_loss = functional.smooth_l1_loss(reconstructed, real, beta=0.04)
            real_features = _features(torch, real)
            reconstruction_features = _features(torch, reconstructed)
            derivative_loss = functional.l1_loss(reconstruction_features[:, 1:], real_features[:, 1:])
            loss_g = adversarial + float(config["reconstruction_weight"]) * height_loss + float(config["derivative_weight"]) * derivative_loss
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer_g.step()
            final_g, final_d = float(loss_g.detach().cpu()), float(loss_d.detach().cpu())
            final_rec = float(height_loss.detach().cpu())

        state = {key: value.detach().cpu().contiguous() for key, value in generator.state_dict().items()}
        states.append(state)
        amplitudes.append(amplitude)
        losses.append({"generator": final_g, "discriminator": final_d, "height_reconstruction": final_rec, "scale_rmse": rmse})
        previous_real = real.detach()
    return TrainedPyramid(states, amplitudes, normalization, scales, losses, parameter_count)


def generate(
    trained: TrainedPyramid,
    target_conditions: np.ndarray,
    output_shape: tuple[int, int],
    config: dict[str, Any],
    device: str,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    seed = int(config["generation_seed"])
    condition_full = torch.from_numpy(target_conditions[None].astype(np.float32)).to(device)
    previous = None
    source_finest = trained.scales[-1]
    for index, (size, state, amplitude) in enumerate(zip(trained.scales, trained.states, trained.noise_amplitudes)):
        shape = (
            max(16, int(round(output_shape[0] * size / source_finest))),
            max(16, int(round(output_shape[1] * size / source_finest))),
        )
        condition = _resize_tensor(functional, condition_full, shape)
        previous = torch.zeros((1, 1, *shape), device=device) if previous is None else _resize_tensor(functional, previous, shape)
        generator, _ = _networks(torch, target_conditions.shape[0], int(config["channels"]), int(config["layers"]))
        generator.load_state_dict(state)
        generator = generator.to(device).eval()
        noise_generator = torch.Generator(device="cpu").manual_seed(seed + 1009 * index)
        noise = torch.randn((1, 1, *shape), generator=noise_generator).to(device)
        with torch.no_grad():
            previous = generator(previous, amplitude * noise, condition)
    result = previous[0, 0].detach().cpu().numpy().astype(np.float64)
    result -= float(np.mean(result))
    return result * trained.normalization_m

