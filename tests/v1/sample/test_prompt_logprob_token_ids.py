# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt logprobs return a request's logprob_token_ids at every prompt
position, in place of the top-k, while other requests in the same batch keep
their top-k."""

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState
from vllm.v1.worker.gpu.sample.prompt_logprob import (
    compute_prompt_logprobs_with_chunking,
)

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")

VOCAB = 512
HIDDEN = 64


def test_num_prompt_logprobs_counts_token_ids():
    assert SamplingParams(prompt_logprobs=5).num_prompt_logprobs == 5
    assert SamplingParams(prompt_logprobs=-1).num_prompt_logprobs == -1
    assert SamplingParams(logprob_token_ids=[1, 2]).num_prompt_logprobs is None
    assert (
        SamplingParams(
            prompt_logprobs=0, logprob_token_ids=[1, 2, 3]
        ).num_prompt_logprobs
        == 3
    )


@pytest.mark.parametrize("num_prompt_logprobs", [0, 2])
def test_prompt_logprobs_honor_logprob_token_ids(num_prompt_logprobs: int):
    device = torch.device("cuda")
    torch.manual_seed(0)

    # Slot 2 asks for three ids, slot 0 for one, slot 1 for none.
    state = LogprobTokenIdsState(max_num_reqs=4, device=device)
    wanted = {2: [7, 300, 511], 0: [5]}
    for slot, ids in wanted.items():
        state.add_request(slot, SamplingParams(logprob_token_ids=ids))
    state.add_request(1, SamplingParams())
    state.apply_staged_writes()
    max_ids = max(len(ids) for ids in wanted.values())

    # Three requests in the batch, in this slot order, with these prompt
    # lengths; the rows cross the chunk boundary when CHUNK_SIZE is small,
    # which the worker's chunking handles the same way as one chunk.
    slots = [2, 1, 0]
    query_lens = [3, 2, 4]
    row_idx_mapping = torch.as_tensor(
        np.repeat(np.array(slots, dtype=np.int32), query_lens), device=device
    )
    n = sum(query_lens)
    weight = torch.randn(HIDDEN, VOCAB, device=device)
    hidden = torch.randn(n, HIDDEN, device=device)
    next_ids = torch.randint(0, VOCAB, (n,), device=device)

    token_ids, logprobs, ranks = compute_prompt_logprobs_with_chunking(
        next_ids,
        hidden,
        lambda h: h @ weight,
        num_prompt_logprobs,
        "raw_logprobs",
        logprob_token_ids_state=state,
        row_idx_mapping=row_idx_mapping,
        max_per_req_token_ids=max_ids,
    )
    ref = torch.log_softmax(hidden @ weight, dim=-1)
    assert token_ids.shape[1] == 1 + max(num_prompt_logprobs, max_ids)

    row = 0
    for slot, query_len in zip(slots, query_lens):
        ids = wanted.get(slot)
        for r in range(row, row + query_len):
            # Column 0 is the prompt's own next token, as before.
            assert token_ids[r, 0].item() == next_ids[r].item()
            torch.testing.assert_close(logprobs[r, 0], ref[r, next_ids[r]])
            if ids:
                assert token_ids[r, 1 : 1 + len(ids)].tolist() == ids
                torch.testing.assert_close(logprobs[r, 1 : 1 + len(ids)], ref[r, ids])
                assert torch.isinf(logprobs[r, 1 + len(ids) :]).all()
            elif num_prompt_logprobs:
                top = ref[r].topk(num_prompt_logprobs).indices.tolist()
                assert token_ids[r, 1 : 1 + num_prompt_logprobs].tolist() == top
        row += query_len
    assert ranks.shape == (n,)
