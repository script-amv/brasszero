# SPDX-License-Identifier: Apache-2.0
"""Explicit numerical runtime probe; all optional imports happen inside the call."""
from io import BytesIO
import math
import platform

from .config import _device


def runtime_check(device: str) -> dict:
    """Prove finite forward/backward/Adam operations and a tensor checkpoint round-trip.

    The probe owns a tiny deterministic model and leaves the caller's CPU RNG and
    thread setting unchanged. Only tensor state dictionaries are deserialized.
    """
    _device(device)
    try:
        import numpy
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the optional RL dependencies in a separate training environment") from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda device is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("mps device is unavailable")

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        # Fixed tensors avoid changing caller RNG state or relying on random success.
        weights = torch.nn.Parameter(torch.tensor([[.25, -.5]], device=device))
        inputs = torch.tensor([[1., 2.], [2., 1.]], device=device)
        targets = torch.tensor([[1.], [0.]], device=device)
        optimizer = torch.optim.Adam([weights], lr=.01)
        initial = weights.detach().clone()
        loss = ((inputs @ weights.t() - targets) ** 2).mean()
        before = loss.item()
        loss.backward()
        grad_norm = weights.grad.norm().item()
        optimizer.step()
        with torch.no_grad():
            after = ((inputs @ weights.t() - targets) ** 2).mean().item()
        if not all(math.isfinite(value) for value in (before, after, grad_norm)):
            raise RuntimeError("runtime probe produced non-finite values")
        changed = not torch.equal(initial, weights.detach())
        checkpoint = BytesIO()
        torch.save({"weights": weights.detach()}, checkpoint)
        checkpoint.seek(0)
        restored = torch.load(checkpoint, map_location=device, weights_only=True)
        roundtrip = torch.equal(restored["weights"], weights.detach())
        if not changed or not roundtrip or after >= before or grad_norm <= 0:
            raise RuntimeError("runtime optimizer/checkpoint probe failed")
        return {"versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "numpy": numpy.__version__},
                "device": device, "platform": platform.platform(),
                "loss_before": before, "loss_after": after, "grad_norm": grad_norm,
                "parameters_changed": changed, "checkpoint_roundtrip": roundtrip}
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    import json
    print(json.dumps(runtime_check("cpu"), indent=2, allow_nan=False))
