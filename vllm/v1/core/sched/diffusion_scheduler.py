# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async scheduling for diffusion requests.

These rules live here so Scheduler.schedule() and AsyncScheduler stay
unchanged. VllmConfig selects this class for a diffusion model under async
scheduling. A sync scheduler creates no output placeholders, which both rules
read.
"""

from typing import Any

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request


def requested_canvas_width(request: Request) -> int | None:
    """The canvas width a request asked for with diffusion_canvas_length."""
    params = request.sampling_params
    extra = params.extra_args if params is not None else None
    width = extra.get("diffusion_canvas_length") if extra else None
    return int(width) if width else None


def diffusion_canvas_width(request: Request, canvas_length: int) -> int:
    """The canvas width a diffusion request asked for, else the served one."""
    return requested_canvas_width(request) or canvas_length


def build_canvas_width_lookup(
    schedule: Any, canvas_length: int, max_num_seqs: int
) -> list[int]:
    """Expand ``canvas_length_per_batch_size`` into a dense lookup indexed by
    batch size (index 0 unused). Batch sizes above the last range keep the
    last width; a gap between ranges keeps the width before it."""
    if not isinstance(schedule, list) or not schedule:
        raise ValueError(
            "canvas_length_per_batch_size must be a non-empty list of "
            "(range_start, range_end, canvas_width) entries."
        )
    parsed: list[tuple[int, int, int]] = []
    for entry in schedule:
        if not isinstance(entry, list | tuple) or len(entry) != 3:
            raise ValueError(
                "Each canvas_length_per_batch_size entry must be "
                "(range_start, range_end, canvas_width)."
            )
        start, end, width = int(entry[0]), int(entry[1]), int(entry[2])
        if start <= 0 or start > end:
            raise ValueError(f"Bad batch-size range ({start}, {end}).")
        if not 1 <= width <= canvas_length:
            raise ValueError(
                f"canvas width {width} must be in [1, {canvas_length}], the "
                "served canvas_length."
            )
        parsed.append((start, end, width))
    parsed.sort()
    if parsed[0][0] != 1:
        raise ValueError("The first batch-size range must start at 1.")
    previous_end = 0
    for start, end, _ in parsed:
        if start <= previous_end:
            raise ValueError("Batch-size ranges must not overlap.")
        previous_end = end
    lookup = [canvas_length] * (max_num_seqs + 1)
    width = parsed[0][2]
    for batch_size in range(1, max_num_seqs + 1):
        for start, end, w in parsed:
            if start <= batch_size <= end:
                width = w
        lookup[batch_size] = width
    return lookup


def _read_in_flight(request: Request, width: int) -> bool:
    """True when a read-only request has all its denoise steps in flight.

    The request emits its canvas on the last step and ends, so a further
    step is discarded.
    """
    params = request.sampling_params
    extra = params.extra_args if params is not None else None
    if not extra or not extra.get("diffusion_read_only"):
        return False
    steps = extra.get("diffusion_max_steps")
    if not steps:
        return False
    # Each in-flight denoise step holds one canvas of placeholders.
    return request.num_output_placeholders >= steps * width


class DiffusionAsyncScheduler(AsyncScheduler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Load-adaptive canvas: width per request, held for a whole block, and
        # the output count at which it was chosen (a change means a commit
        # landed and the next block may take a new width).
        self._canvas_lookup: list[int] | None = None
        self._block_width: dict[str, int] = {}
        self._block_start: dict[str, int] = {}
        self.init_canvas_schedule()

    def init_canvas_schedule(self) -> None:
        config = self.vllm_config.diffusion_config
        schedule = config.canvas_length_per_batch_size if config else None
        self._canvas_lookup = (
            build_canvas_width_lookup(
                schedule, self.num_spec_tokens, self.scheduler_config.max_num_seqs
            )
            if schedule
            else None
        )

    def _adaptive_canvas_width(self, request: Request, load: int) -> int | None:
        """The width for this request's current block under the schedule,
        or None when there is no schedule or the request set its own."""
        if self._canvas_lookup is None or requested_canvas_width(request):
            return None
        req_id = request.request_id
        produced = request.num_output_tokens
        width = self._block_width.get(req_id)
        if width is None or self._block_start.get(req_id) != produced:
            width = self._canvas_lookup[min(load, len(self._canvas_lookup) - 1)]
            self._block_width[req_id] = width
            self._block_start[req_id] = produced
        return width

    def _free_request(self, request: Request, *args: Any, **kwargs: Any):
        self._block_width.pop(request.request_id, None)
        self._block_start.pop(request.request_id, None)
        return super()._free_request(request, *args, **kwargs)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        # Waiting requests join the batch this step when they fit, so count
        # them: a burst arriving at an idle server starts narrow.
        load = max(
            1,
            min(
                len(self.running) + len(self.waiting),
                self.scheduler_config.max_num_seqs,
            ),
        )
        for request in self.running:
            width = self._adaptive_canvas_width(request, load) or (
                diffusion_canvas_width(request, self.num_spec_tokens)
            )
            # The placeholders are as wide as the served canvas.
            if len(request.spec_token_ids) > width:
                request.spec_token_ids = request.spec_token_ids[:width]
            if _read_in_flight(request, width):
                # Scheduler.schedule()'s max_tokens guard cannot be reached
                # from a subclass. schedule() advances current_step before it
                # checks decode eligibility, so current_step + 2 skips this
                # call alone. max keeps a longer pipeline-parallel wait.
                request.next_decode_eligible_step = max(
                    request.next_decode_eligible_step, self.current_step + 2
                )
        return super().schedule(throttle_prefills)
