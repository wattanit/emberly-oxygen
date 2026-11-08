"""Shared by the oracle scripts that run a model in float64 (Spec §10).

The float64 reference forces RMSNorm and the attention softmax to float64, which
the pinned library computes in float32 inside any model, and checks that the
replacement is active. `generate.py` (the base model) and `chat_reference.py` (the
instruct model) import it, so both references are the same kind of reference.
"""

import contextlib
import platform
import subprocess

import torch
import torch.nn.functional as F
from transformers.models.llama import modeling_llama


def machine() -> dict:
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor()
    return {
        "cpu": cpu,
        "arch": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "torch_threads": torch.get_num_threads(),
    }


def top2_gaps(logits: torch.Tensor) -> list[float]:
    """Gap between the two highest logits at each row of [positions, vocab]."""
    top = logits.topk(2, dim=-1).values
    return (top[:, 0] - top[:, 1]).tolist()


@contextlib.contextmanager
def float64_pure_ops():
    """Make RMSNorm and the attention softmax compute in the model's own dtype.

    The pinned library casts RMSNorm's input to float32 and the softmax to
    float32 even in a float64 model, which made its "float64" output depend on
    the thread count by up to 1.7e-5 in logits (plan 0.1.2, task 9.4). The
    originals are restored on exit, so the float32 run is the plain library.
    The rotary tables stay float32, as published (Spec section 10).
    """
    original_forward = modeling_llama.LlamaRMSNorm.forward
    original_softmax = F.softmax

    def rms_norm(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))

    def softmax(x, dim=None, _stacklevel=3, dtype=None):
        return original_softmax(x, dim=dim, dtype=None)

    modeling_llama.LlamaRMSNorm.forward = rms_norm
    F.softmax = softmax
    try:
        yield
    finally:
        modeling_llama.LlamaRMSNorm.forward = original_forward
        F.softmax = original_softmax


def check_float64_pure_ops() -> None:
    """Fail loudly if the replacement is not active (Spec section 10).

    The input differs from 1.0 by 1e-9, which float32 cannot hold, so a
    normalization that passed through float32 would miss the float64 result
    by about 1e-9, far over the 1e-14 allowed.
    """
    x = torch.tensor([[1.0 + 1e-9, 2.0, 3.0, 4.0]], dtype=torch.float64)
    norm = modeling_llama.LlamaRMSNorm(4, eps=1e-5).double()
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    assert (norm(x) - expected).abs().max().item() < 1e-14, "RMSNorm is not computing in float64"
    assert F.softmax(x, dim=-1, dtype=torch.float32).dtype == torch.float64, "softmax is still float32"
