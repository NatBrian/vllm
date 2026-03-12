# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline SpecSteer example (greedy-only).

SpecSteer deploy checklist (rapid rollout):
- `speculative_config` required fields: `method="specsteer"`, `model`,
  `num_speculative_tokens`.
- Greedy-only path: use `SamplingParams(temperature=0, ...)`.
- `base_model`: omit/`None` to use the target `--model`; set a distinct model
  only when draft scoring should be anchored to another base model.
- `draft_prompt`: optional per request; if omitted, the regular `prompt` is
  used as the draft prompt.
- Current support scope: offline `LLM.generate` examples only (not server API
  path yet).

Usage:
  $env:PYTHONPATH = "."; python examples/offline_inference/specsteer.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --draft-model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import os

if os.name == "nt":
    # Windows PyTorch builds may not include libuv support for TCPStore.
    os.environ.setdefault("USE_LIBUV", "0")
    # Avoid opaque child-process startup failures on Windows for local runs.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # Force loopback rendezvous values to avoid hostname resolution issues.
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
    os.environ.setdefault("VLLM_LOOPBACK_IP", "127.0.0.1")

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    args = parser.parse_args()

    attention_config = None
    if os.name == "nt":
        # FlashAttention extension is often unavailable in local Windows setups.
        attention_config = {"backend": "FLEX_ATTENTION"}

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=(os.name == "nt"),
        compilation_config={"custom_ops": ["none"]},
        attention_config=attention_config,
        speculative_config={
            "method": "specsteer",
            "model": args.draft_model,
            "base_model": args.base_model,
            "num_speculative_tokens": 4,
            "disable_padded_drafter_batch": False,
            "gamma": 0.6,
            "eps": 1e-10,
            "fusion_method": "costeer",
            "specsteer_enable_bonus_token": False,
        },
    )

    outputs = llm.generate(
        [
            {
                "prompt": "Describe speculative decoding in one paragraph.",
                "draft_prompt": (
                    "Give a concise, high-level, approximate draft for "
                    "speculative decoding."
                ),
            }
        ],
        sampling_params=SamplingParams(temperature=0, max_tokens=args.max_tokens),
    )

    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
