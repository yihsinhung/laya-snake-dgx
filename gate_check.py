# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 laya-snake-dgx contributors
#
# Modified from laya-mlx (https://github.com/mizorewww/laya-mlx, Apache-2.0)
# and Laya (https://github.com/NandhaKishorM/laya, Apache-2.0).
# See LICENSE and NOTICE for details.

"""DGX Spark CUDA gate test for the laya runtime (self-contained, offline).

Run on the Spark inside the venv:
    ~/laya-snake-dgx-work/venv/bin/python ~/laya-snake-dgx/gate_check.py

This is the single checkpoint that decides whether the whole Snake port is
viable on a GB10. If this passes, everything downstream (pure Python plus the
verified-identical predict() shape) is on solid ground. If it fails, no amount
of Snake porting helps.
"""
import os
# Must be set BEFORE importing torch: avoids Triton first-inference C compile
# that needs python3-dev (absent; no sudo). Same answers/latency (see #365).
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
import platform
import torch

# Offline + point at the pre-downloaded weights so the gate never re-downloads.
os.environ.setdefault("HF_HOME", "/home/asus/laya-snake-dgx-work/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from laya import load

MODEL = os.environ.get("LAYA_GATE_MODEL", "convaiinnovations/laya-multilingual")
DEVICE = os.environ.get("LAYA_GATE_DEVICE", "cuda")

# Self-contained request (mirrors the repo's examples/docker/request.json shape).
STATE = (
    "Customer writes: I was billed twice for the same plan this month and want "
    "one of them refunded to my original payment method."
)
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs and outages",
            "sales": "new purchases",
        },
    },
    "refund": {
        "type": "noul",
        "instructions": "Does this customer ask for a refund?",
    },
}


def check():
    assert platform.machine() == "aarch64", f"expected aarch64, got {platform.machine()}"
    print(f"arch          : {platform.machine()}")
    print(f"torch         : {torch.__version__} (cuda {torch.version.cuda})")
    assert torch.cuda.is_available(), "torch.cuda.is_available() False -> NVIDIA driver / Container Toolkit not wired"
    print(f"gpu           : {torch.cuda.get_device_name(0)}")
    print(f"capability    : {torch.cuda.get_device_capability(0)}")
    print(f"arch list     : {torch.cuda.get_arch_list()}")

    print(f"loading       : {MODEL} on {DEVICE} ...")
    agent = load(MODEL, device=DEVICE)
    assert next(agent.model.parameters()).device.type == DEVICE, "model fell back to CPU"
    print("device check  : model loaded on", DEVICE)

    result = agent.predict(STATE, QUESTIONS)
    answers = result["answers"]
    assert set(answers) == set(QUESTIONS), "answers/question mismatch"
    # Shape compatibility with the Snake policy, verified against laya/agent.py:
    ans = answers[next(iter(answers))]
    assert "probabilities" in ans or "noul" in ans, "answer shape mismatch"
    for qid, a in answers.items():
        print(f"  {qid:12s} -> {sorted(a)} : {a.get('choice') or a.get('noul')}")
    print("usage keys    :", sorted(result.get("usage", {})))
    print(f"max mem      : {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    print("\nGATE PASSED")


if __name__ == "__main__":
    check()
