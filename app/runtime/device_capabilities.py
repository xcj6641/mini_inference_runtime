from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceCapabilities:
    device_name: str
    compute_capability: tuple[int, int] | None
    supports_fp16: bool
    supports_bf16: bool
    hardware_fp8_candidate: bool


def detect_device_capabilities(
    device: torch.device | str,
) -> DeviceCapabilities:
    resolved_device = torch.device(device)

    if resolved_device.type != "cuda":
        return DeviceCapabilities(
            device_name=str(resolved_device),
            compute_capability=None,
            supports_fp16=False,
            supports_bf16=False,
            hardware_fp8_candidate=False,
        )

    device_index = resolved_device.index

    if device_index is None:
        device_index = torch.cuda.current_device()

    device_name = torch.cuda.get_device_name(
        device_index
    )

    compute_capability = (
        torch.cuda.get_device_capability(
            device_index
        )
    )

    major, minor = compute_capability

    return DeviceCapabilities(
        device_name=device_name,
        compute_capability=(major, minor),
        supports_fp16=True,
        supports_bf16=torch.cuda.is_bf16_supported(),
        hardware_fp8_candidate=major >= 9,
    )