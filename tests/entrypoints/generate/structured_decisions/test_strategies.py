# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI

from vllm.entrypoints.generate.structured_decisions.api_router import (
    register_structured_decisions_api_router,
)
from vllm.entrypoints.generate.structured_decisions.question_types import LABELS
from vllm.entrypoints.generate.structured_decisions.serving import state_text
from vllm.entrypoints.generate.structured_decisions.strategies import (
    CanvasStrategy,
    NextTokenStrategy,
    reply_label_ids,
    select_read_strategy,
)


def model(architecture: str, logprobs_mode: str = "raw_logprobs") -> Any:
    return SimpleNamespace(architecture=architecture, logprobs_mode=logprobs_mode)


def test_strategy_selection():
    qwen = "Qwen3ForCausalLM"
    assert select_read_strategy(model(qwen)) is NextTokenStrategy
    assert select_read_strategy(model(qwen, "processed_logprobs")) is NextTokenStrategy
    assert (
        select_read_strategy(model("DiffusionGemmaForBlockDiffusion")) is CanvasStrategy
    )
    with pytest.raises(ValueError, match="does not support LlamaForCausalLM"):
        select_read_strategy(model("LlamaForCausalLM"))
    with pytest.raises(ValueError, match="not raw_logits"):
        select_read_strategy(model(qwen, "raw_logits"))


def test_route_needs_the_flag():
    for enabled in (False, True):
        app = FastAPI()
        app.state.args = SimpleNamespace(enable_structured_decisions=enabled)
        register_structured_decisions_api_router(app)
        paths = {getattr(route, "path", None) for route in app.routes}
        assert ("/v1/systemone" in paths) == enabled


def test_structured_state_keeps_unicode_in_prompt():
    state = {"message": "Français 日本語"}
    assert state_text(state) == '{"message": "Français 日本語"}'


@pytest.fixture(scope="module")
def qwen():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "state"}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=False,
    )
    return tokenizer, prompt_ids


def test_labels_start_the_reply(qwen):
    tokenizer, prompt_ids = qwen
    tail, ids = reply_label_ids(tokenizer, prompt_ids)
    assert tokenizer.decode(tail) == "\n\n"
    assert [tokenizer.decode([i]) for i in ids] == list(LABELS)
    # A noul's and a score's labels are one token here too.
    _, ids = reply_label_ids(tokenizer, prompt_ids, ("yes", "no", *"0123456789"))
    assert len(set(ids)) == 12
    # After a colon, Qwen writes ":A" as one token, so "A" is not one token.
    with pytest.raises(ValueError, match="not one distinct token"):
        reply_label_ids(tokenizer, tokenizer.encode("team:"))


def test_canvas_read():
    strategy = CanvasStrategy.__new__(CanvasStrategy)
    strategy.thought, strategy.end, strategy.pad = [10, 11, 12, 13], 106, 0
    strategy.width, strategy.vocab_size = 16, 1000
    params = strategy._sampling_params([65, 66], prompt_ids=[1, 2, 3])
    assert params.extra_args is not None
    canvas = params.extra_args["diffusion_seed_canvas"]
    # The label slot as noise, the end of the turn, padding.
    assert canvas[1:] == [106] + [0] * 14
    assert params.max_tokens == 2 and params.logprob_token_ids == [65, 66]
    again = strategy._sampling_params([65, 66], prompt_ids=[1, 2, 3])
    assert again.extra_args == params.extra_args
    read_input = strategy._read_input(
        {"type": "token", "prompt_token_ids": [1, 2, 3]}, [1, 2, 3]
    )
    assert read_input == {
        "type": "token",
        "prompt_token_ids": [1, 2, 3, 10, 11, 12, 13],
    }
