# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for discrete diffusion (dLLM) models."""

from pydantic import Field

from vllm.config.utils import config


@config
class DiffusionConfig:
    """Configuration for discrete diffusion language models (dLLMs).

    dLLMs generate tokens via iterative denoising over a fixed-length canvas
    rather than left-to-right autoregressive decoding. They reuse the
    speculative-decoding data path (draft token ids, scheduled spec decode
    tokens) with overloaded semantics for block-based generation.
    """

    canvas_length: int = Field(default=None, gt=0)  # type: ignore[assignment]
    """Length of the denoising canvas (block).  Also determines the number of
    speculative tokens scheduled per step."""

    max_denoising_steps: int | None = None
    """Maximum number of denoising iterations per canvas block.
    If not set, read from the model's generation_config.json."""

    canvas_length_per_batch_size: list[tuple[int, int, int]] | None = None
    """Load-adaptive canvas width for generation. Each entry is an inclusive
    batch-size range ``(range_start, range_end, canvas_width)``; the async
    scheduler picks the width for each block of a request from the number of
    running and waiting requests when the block starts, so a lightly loaded
    server denoises wide blocks (more tokens per forward pass) and a busy one
    denoises narrow blocks (more requests per forward pass). Widths must not
    exceed ``canvas_length``. Requests that set ``diffusion_canvas_length``
    keep it. None uses ``canvas_length`` for every block."""
