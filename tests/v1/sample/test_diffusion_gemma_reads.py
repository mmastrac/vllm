# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request DiffusionGemma state behind structured reads: seed canvases,
pinned positions, read-only slots and the per-slot step cap."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.diffusion import DiffusionConfig
from vllm.model_executor.models.diffusion_gemma import (
    _MASKED_LOGIT,
    DiffusionGemmaRequestStates,
    DiffusionSampler,
    _compiled_sample_step,
    _concat_logprob_stashes,
    _denoise_temperature,
    _mask_rows_to_allowed,
    sample_row_stats_reference,
)
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the sampler state lives on the GPU"
)

CL = 8
VOCAB = 64
MAX_REQS = 4
MAX_STEPS = 48


def _states() -> DiffusionGemmaRequestStates:
    return DiffusionGemmaRequestStates(
        max_num_reqs=MAX_REQS,
        canvas_length=CL,
        vocab_size=VOCAB,
        max_denoising_steps=MAX_STEPS,
        device=torch.device("cuda"),
        hidden_size=4,
        stability_threshold=2,
    )


def _slots(*idx: int) -> tuple[np.ndarray, torch.Tensor]:
    slots = np.array(idx, dtype=np.int64)
    return slots, torch.tensor(slots, device="cuda")


def _sampler() -> DiffusionSampler:
    """Build the real sampler with flat logits and no early convergence."""
    states = _states()
    requests = RequestState(MAX_REQS, 64, MAX_REQS * CL, CL, VOCAB, states.device)
    return DiffusionSampler(
        Sampler(VllmConfig(), MAX_REQS, VOCAB, states.device, requests),
        DiffusionConfig(canvas_length=CL, max_denoising_steps=MAX_STEPS),
        VOCAB,
        states,
        confidence_threshold=-1.0,
        t_min=1.0,
        t_max=1.0,
        entropy_bound=0.1,
        embed_weight=torch.ones(VOCAB, 4, device=states.device),
        normalizer=torch.tensor(1.0, device=states.device),
    )


def _add_sampler_request(
    sampler: DiffusionSampler, req_id: str, cap: int, read_only: bool = True
) -> int:
    """Admit a request using the runner's real state lifecycle."""
    sampler.req_states.add_request(req_id, 2, [1, 2], 0, CL)
    slot = sampler.req_states.req_id_to_index[req_id]
    sampler.diffusion_states.add_request(slot)
    sampler.add_request(
        slot,
        SamplingParams(
            extra_args={"diffusion_max_steps": cap, "diffusion_read_only": read_only}
        ),
    )
    sampler.req_states.apply_staged_writes()
    sampler.apply_staged_writes()
    return slot


def _sampler_batch(
    sampler: DiffusionSampler, req_ids: list[str], prefill: bool = False
) -> InputBatch:
    """Populate a real InputBatch with complete prompts or full-width canvases."""
    n = len(req_ids)
    device = sampler.diffusion_states.device
    width = 2 if prefill else CL
    slots, slots_gpu = _slots(*(sampler.req_states.req_id_to_index[r] for r in req_ids))
    offsets = np.arange(n + 1, dtype=np.int32) * width
    return replace(
        InputBatch.make_dummy(
            n, n * width, InputBuffers(n, n * width, device), is_padding=False
        ),
        req_ids=req_ids,
        idx_mapping_np=slots,
        idx_mapping=slots_gpu,
        num_draft_tokens=0 if prefill else n * CL,
        num_draft_tokens_per_req=None if prefill else np.full(n, CL, dtype=np.int32),
        prefill_len_np=np.full(n, 2, dtype=np.int32),
        num_computed_tokens_np=np.full(n, 0 if prefill else 2, dtype=np.int32),
        num_computed_prefill_tokens_np=np.full(n, 0 if prefill else 2, dtype=np.int32),
        is_prefilling_np=np.full(n, prefill),
        has_prefill=prefill,
        cu_num_logits_np=offsets,
        cu_num_logits=torch.tensor(offsets, device=device),
    )


def _run_sampler(sampler: DiffusionSampler, req_ids: list[str], prefill: bool = False):
    """Count actual SC matmuls through __call__; this is not a timing benchmark."""
    batch = _sampler_batch(sampler, req_ids, prefill)
    logits = torch.zeros(
        batch.num_tokens, VOCAB, device=sampler.diffusion_states.device
    )
    # Eager operators remain visible to the profiler; no sampler method is replaced.
    with (
        torch.compiler.set_stance("force_eager"),
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p,
    ):
        output = sampler(logits, batch)
    matmuls = sum(
        event.count for event in p.key_averages() if event.key == "aten::matmul"
    )
    return output, matmuls


def test_sampler_last_step_read_cap_and_overshoot():
    """Only the first three of six read steps need self-conditioning at cap 4."""
    sampler = _sampler()
    slot = _add_sampler_request(sampler, "read", 4)
    output, matmuls = _run_sampler(sampler, ["read"], prefill=True)
    assert output.num_sampled.tolist() == [0]
    assert matmuls == 0
    assert sampler.diffusion_states.step[slot].item() == 0

    for step in range(1, 7):
        output, matmuls = _run_sampler(sampler, ["read"])
        assert matmuls == int(step < 4)
        assert output.num_sampled.tolist() == [CL if step >= 4 else 0]
        assert sampler.diffusion_states.step[slot].item() == step
        assert bool(sampler.diffusion_states.self_conditioning_embeds[slot].any()) == (
            step < 4
        )


@pytest.mark.parametrize("other_read_only", [False, True], ids=["generation", "read"])
def test_sampler_last_step_mixed_tile_preserves_live_request(other_read_only):
    """A same-width unfinished neighbour still needs the shared tile's matmul."""
    sampler = _sampler()
    done = _add_sampler_request(sampler, "read", 4)
    live = _add_sampler_request(sampler, "live", 6, read_only=other_read_only)
    _run_sampler(sampler, ["read", "live"], prefill=True)
    for _ in range(4):
        output, matmuls = _run_sampler(sampler, ["read", "live"])
        assert matmuls == 1
    assert output.num_sampled.tolist() == [CL, 0]
    assert not sampler.diffusion_states.self_conditioning_embeds[done].any()
    assert sampler.diffusion_states.self_conditioning_embeds[live].any()
    assert sampler.diffusion_states.step[[done, live]].tolist() == [4, 4]


def test_sampler_last_step_slot_reuse_resets_cap_and_counter():
    """A new read in a completed slot starts its own cap and SC schedule."""
    sampler = _sampler()
    slot = _add_sampler_request(sampler, "old", 4)
    _run_sampler(sampler, ["old"], prefill=True)
    for _ in range(4):
        _run_sampler(sampler, ["old"])
    sampler.diffusion_states.remove_request(slot)
    sampler.req_states.remove_request("old")
    assert _add_sampler_request(sampler, "new", 2) == slot
    _run_sampler(sampler, ["new"], prefill=True)
    for step in (1, 2):
        output, matmuls = _run_sampler(sampler, ["new"])
        assert matmuls == int(step < 2)
        assert output.num_sampled.tolist() == [CL if step == 2 else 0]
        assert sampler.diffusion_states.step[slot].item() == step
    assert sampler.diffusion_states.max_steps[slot].item() == 2


def test_seed_canvas_replaces_only_seeded_slots():
    states = _states()
    for slot in range(3):
        states.add_request(slot)
    seed = list(range(CL))
    states.set_seed_canvas(1, seed)
    slots, slots_gpu = _slots(0, 1, 2)
    states.init_canvas(slots_gpu)
    before = states.canvas[slots_gpu].clone()

    states.apply_seed_canvases(slots, slots_gpu)

    after = states.canvas[slots_gpu]
    assert after[1].tolist() == seed
    assert torch.equal(after[0], before[0])
    assert torch.equal(after[2], before[2])


def test_apply_seed_canvases_leaves_unseeded_batches_alone():
    states = _states()
    states.add_request(0)
    slots, slots_gpu = _slots(0)
    states.init_canvas(slots_gpu)
    before = states.canvas[0].clone()

    states.apply_seed_canvases(slots, slots_gpu)

    assert torch.equal(states.canvas[0], before)


def test_add_request_clears_seed_and_read_only():
    states = _states()
    states.add_request(0)
    states.set_seed_canvas(0, [1] * CL)
    states.set_read_only(0)
    assert states.seeded_slots == {0}
    assert states.read_only_slots == {0}

    states.add_request(0)

    assert not states.seeded_slots
    assert not states.read_only_slots
    assert not bool(states.has_seed[0])
    assert not bool(states.read_only[0])


def test_canvas_width_resets_with_the_slot():
    states = _states()
    states.add_request(0)
    states.canvas_width_np[0] = 4
    states.set_seed_canvas(0, [7, 7, 7, 7])
    assert states.seed_canvas[0, :4].tolist() == [7, 7, 7, 7]

    states.add_request(0)

    assert states.canvas_width_np[0] == CL


def test_remove_request_forgets_the_slot():
    states = _states()
    states.add_request(0)
    states.set_seed_canvas(0, [1] * CL)
    states.set_read_only(0)

    states.remove_request(0)

    assert not states.seeded_slots
    assert not states.read_only_slots


def _denoise_once(
    states: DiffusionGemmaRequestStates,
    slots: list[int],
    compute_sc: bool = True,
    width: int = CL,
    embed_weight: torch.Tensor | None = None,
    embed_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One compiled denoise step over ``slots`` with flat logits, so nothing
    converges by stability or confidence and only the step cap can end it.
    ``width`` below CL runs the step on [:, :width] views, as the sampler
    does for a narrow tile."""
    n = len(slots)
    device = states.device
    decode_slots = torch.tensor(slots, dtype=torch.int64, device=device)
    decode_idx = torch.arange(n, dtype=torch.int64, device=device)
    sampled = torch.zeros(n, CL, dtype=torch.int32, device=device)[:, :width]
    num_sampled = torch.zeros(n, dtype=torch.int32, device=device)
    temp = _denoise_temperature(states.step, decode_slots, float(MAX_STEPS), 0.5, 1.0)
    argmax, sample, entropy, probs = sample_row_stats_reference(
        torch.zeros(n * width, VOCAB, device=device),
        temp,
        width,
        embed_dtype if compute_sc else None,
    )
    _compiled_sample_step(
        sample.view(n, width),
        argmax.view(n, width),
        entropy.view(n, width),
        probs.view(n, width, -1) if probs is not None else None,
        decode_slots,
        decode_idx,
        decode_slots,
        torch.full((n,), width, dtype=torch.int64, device=device),
        states.canvas[:, :width],
        states.argmax_canvas[:, :width],
        states.step,
        states.is_encoder_phase,
        states.confident,
        states.self_conditioning_embeds[:, :width],
        (
            torch.zeros(VOCAB, 4, dtype=embed_dtype, device=device)
            if embed_weight is None
            else embed_weight
        ),
        torch.tensor(1.0, dtype=embed_dtype, device=device),
        states.accepted_canvas_history[:, :, :width],
        states.accepted_canvas_history_len,
        states.max_steps,
        states.pin_mask[:, :width],
        states.seed_canvas[:, :width],
        states.read_only,
        sampled,
        num_sampled,
        torch.zeros(MAX_REQS, CL, dtype=torch.int64, device=device),
        confidence_threshold=0.1,
        vocab_size=VOCAB,
        CL=width,
        ST=states.stability_threshold,
        entropy_bound=0.1,
        sc_vocab_start=0,
        sc_vocab_end=VOCAB,
        tp_size=1,
        tp_group_name="",
        compute_sc=compute_sc,
    )
    return sampled, num_sampled


def test_pinned_positions_hold_their_seed_through_a_step():
    states = _states()
    states.add_request(0)
    states.is_encoder_phase[0] = False
    seed = list(range(10, 10 + CL))
    states.set_seed_canvas(0, seed)
    states.canvas[0] = torch.tensor(seed, device="cuda")
    states.set_pins(0, [0, 1, 2, 3])

    # Flat logits accept nothing, so every free position is renoised.
    _denoise_once(states, [0], embed_weight=torch.ones(VOCAB, 4, device="cuda"))

    assert states.canvas[0, :4].tolist() == seed[:4]
    # The soft embed is zero at pinned positions and non-zero elsewhere.
    sc = states.self_conditioning_embeds[0]
    assert not sc[:4].any()
    assert sc[4:].any()


def test_add_request_clears_pins():
    states = _states()
    states.add_request(0)
    states.set_seed_canvas(0, [1] * CL)
    states.set_pins(0, [2, 3])
    assert states.pin_mask[0].tolist() == [False, False, True, True] + [False] * 4

    states.add_request(0)

    assert not states.pin_mask[0].any()


def test_single_step_tile_skips_self_conditioning():
    states = _states()
    states.add_request(0)
    states.is_encoder_phase[0] = False
    states.self_conditioning_embeds[0] = 1.0

    _denoise_once(states, [0], compute_sc=False)

    assert not states.self_conditioning_embeds[0].any()


def test_sample_step_is_compiled():
    assert hasattr(_compiled_sample_step, "_torchdynamo_orig_callable")


@pytest.mark.parametrize("stance", ["default", "force_eager"])
def test_self_conditioning_stores_a_bf16_model_in_the_fp32_buffer(stance):
    # The model's embeddings are bf16 while the buffer is fp32. Compiled code
    # casts on the store, but eager does not, and the step runs eager once
    # torch.compile hits its recompile limit.
    states = _states()
    states.add_request(0)
    states.is_encoder_phase[0] = False

    with torch.compiler.set_stance(stance):
        _denoise_once(
            states,
            [0],
            embed_weight=torch.ones(VOCAB, 4, dtype=torch.bfloat16, device="cuda"),
            embed_dtype=torch.bfloat16,
        )

    assert states.self_conditioning_embeds.dtype == torch.float32
    # uniform probs @ an all-ones embedding
    assert torch.allclose(
        states.self_conditioning_embeds[0], torch.ones(CL, 4, device="cuda")
    )


def test_narrow_tile_leaves_the_rest_of_the_canvas_alone():
    states = _states()
    states.add_request(0)
    states.is_encoder_phase[0] = False
    states.canvas[0] = 5
    states.argmax_canvas[0] = 5

    _denoise_once(states, [0], width=4)

    # Columns past the width are untouched; the step counted.
    assert states.canvas[0, 4:].tolist() == [5] * (CL - 4)
    assert states.argmax_canvas[0, 4:].tolist() == [5] * (CL - 4)
    assert states.step[0] == 1


def test_step_cap_is_per_slot():
    states = _states()
    for slot in (0, 1):
        states.add_request(slot)
        states.is_encoder_phase[slot] = False
    states.max_steps[0] = 1

    _denoise_once(states, [0, 1])

    # Slot 0 hit its cap and moves to commit. Slot 1 keeps denoising.
    assert states.is_encoder_phase[:2].tolist() == [True, False]
    assert states.step[:2].tolist() == [1, 1]


def _stash(rows: int, width: int, first_id: int) -> LogprobsTensors:
    ids = torch.arange(first_id, first_id + rows * width).reshape(rows, width)
    return LogprobsTensors(
        logprob_token_ids=ids,
        logprobs=-ids.float(),
        selected_token_ranks=torch.zeros(rows, dtype=torch.int64),
    )


def test_stashes_of_different_widths_join():
    # A read that asked for 10 label ids and a generation that asked for 2
    # logprobs converged in different steps, so their stashes differ in width.
    wide, narrow = _stash(2, 11, 100), _stash(3, 3, 0)

    out = _concat_logprob_stashes([narrow, wide], [0, 3])

    assert out.logprob_token_ids.shape == (5, 11)
    assert out.logprobs.shape == (5, 11)
    assert out.cu_num_generated_tokens == [0, 3]
    # The narrow rows keep their values and pad with id 0 at -inf.
    assert torch.equal(out.logprob_token_ids[:3, :3], narrow.logprob_token_ids)
    assert not out.logprob_token_ids[:3, 3:].any()
    assert torch.isneginf(out.logprobs[:3, 3:]).all()
    assert torch.equal(out.logprobs[3:], wide.logprobs)


@pytest.mark.parametrize("width", [4, CL])
@pytest.mark.parametrize("steps", [1, 2])
def test_read_emits_at_convergence_while_generation_waits_for_commit(width, steps):
    states = _states()
    for slot in (2, 0):
        states.add_request(slot)
        states.is_encoder_phase[slot] = False
        states.max_steps[slot] = steps
    states.set_read_only(2)

    for step in range(steps):
        sampled, counts = _denoise_once(states, [2, 0], width=width)
        assert counts.tolist() == ([width, 0] if step == steps - 1 else [0, 0])

    assert torch.equal(sampled[0], states.argmax_canvas[2, :width].int())
    assert not states.is_encoder_phase[2]
    assert states.is_encoder_phase[0]
    assert not states.self_conditioning_embeds[2].any()

    sampled, counts = _denoise_once(states, [0], width=width)
    assert counts.tolist() == [width]
    assert torch.equal(sampled[0], states.argmax_canvas[0, :width].int())


def test_batch_allowed_needs_one_shared_set():
    states = _states()
    states.constrained[0] = (1, 2, 3)
    states.constrained[1] = (4, 5)

    # Mixed sets, or a slot with no set, fall back to per-row masking.
    assert states.batch_allowed([0, 1]) is None
    assert states.batch_allowed([0, 2]) is None
    assert states.batch_allowed([]) is None

    shared = states.batch_allowed([0])
    assert shared is not None
    assert shared.tolist() == [1, 2, 3]
    assert shared.dtype == torch.int64
    assert states.batch_allowed([0, 0]) is shared
    assert states.allowed_tensor((1, 2, 3)) is shared


def test_mask_rows_to_allowed_masks_only_constrained_rows():
    logits = torch.randn(5, VOCAB, device="cuda")
    before = logits.clone()
    first = torch.tensor([3, 7], device="cuda")
    third = torch.tensor([0], device="cuda")

    out = _mask_rows_to_allowed(logits, [0, 2, 3], [2, 1, 2], [first, None, third])

    assert out is not logits
    assert torch.equal(logits, before)
    # Masked columns hold the finite sentinel, so entropy stays finite.
    kept = out > _MASKED_LOGIT
    assert torch.isfinite(out).all()
    assert kept[0:2].sum(dim=1).tolist() == [2, 2]
    assert kept[0:2][:, first].all()
    assert torch.equal(out[0:2][:, first], before[0:2][:, first])
    assert torch.equal(out[2], before[2])
    assert kept[3:5].sum(dim=1).tolist() == [1, 1]
    assert torch.equal(out[3:5][:, third], before[3:5][:, third])
    # The masked row's softmax is the distribution renormalized over the set.
    torch.testing.assert_close(
        out[0].softmax(dim=-1)[first], before[0, first].softmax(dim=-1)
    )


def test_mask_rows_to_allowed_is_a_no_op_without_constrained_rows():
    logits = torch.randn(3, VOCAB, device="cuda")
    assert _mask_rows_to_allowed(logits, [0, 1], [1, 2], [None, None]) is logits
