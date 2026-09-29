# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read strategies: how a model is asked for label probabilities.

A model names the strategy it needs (``decision_read_strategy`` on the model
class, ``"next_token"`` by default). When no strategy is registered under that
name, the decision route answers 501.
"""

import math
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
from vllm.utils.async_utils import merge_async_iterators

from .protocol import ReadPromptRequest
from .question_types import Question, StructuredDecisionError, label_softmax
from .templates import DecisionTemplate, RenderedDecision


@dataclass
class ReadContext:
    engine_client: EngineClient
    online_renderer: OnlineRenderer
    chat_template: str | None
    chat_template_content_format: ChatTemplateContentFormatOption
    default_chat_template_kwargs: dict[str, Any]


@dataclass(frozen=True)
class DecisionLimits:
    max_questions: int
    max_options: int


@dataclass
class QuestionRead:
    probs: list[float]
    label_mass: float
    argmax_is_label: bool
    input_tokens: int
    output_tokens: int


class ReadStrategy(ABC):
    def __init__(self, context: ReadContext):
        self.context = context

    @abstractmethod
    def limits(self) -> DecisionLimits: ...

    @abstractmethod
    async def read(
        self,
        questions: list[Question],
        template: DecisionTemplate,
        instructions: str | None,
        state: str,
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
    ) -> list[QuestionRead]:
        """One read per question, in order. Raises StructuredDecisionError for
        a request the model cannot answer."""


READ_STRATEGIES: dict[str, type[ReadStrategy]] = {}


def register_read_strategy(name: str):
    def register(cls: type[ReadStrategy]) -> type[ReadStrategy]:
        if name in READ_STRATEGIES:
            raise ValueError(f"read strategy {name!r} is already registered")
        READ_STRATEGIES[name] = cls
        return cls

    return register


def select_read_strategy(model_config: ModelConfig) -> type[ReadStrategy] | None:
    return READ_STRATEGIES.get(model_config._model_info.decision_read_strategy)


@register_read_strategy("next_token")
class NextTokenStrategy(ReadStrategy):
    """Autoregressive models. Each question is one request: the chat prompt with
    the reply prefilled up to the question's label, one generated token, and the
    logprobs of the label tokens. The requests share the system prompt
    and the state, so prefix caching prefills them once."""

    def limits(self) -> DecisionLimits:
        return DecisionLimits(max_questions=64, max_options=MAX_LOGPROB_TOKEN_IDS)

    async def read(
        self,
        questions: list[Question],
        template: DecisionTemplate,
        instructions: str | None,
        state: str,
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
    ) -> list[QuestionRead]:
        ctx = self.context
        tokenizer = ctx.online_renderer.renderer.get_tokenizer()
        rendered = template.render(instructions, questions)
        read_request = ReadPromptRequest(chat_template_kwargs=chat_template_kwargs)

        slots, engine_inputs = [], []
        for q in questions:
            slot = rendered.slot(tokenizer, q)
            messages = [
                {"role": "system", "content": rendered.system_text},
                {"role": "user", "content": state},
                {"role": "assistant", "content": tokenizer.decode(slot.prefix_ids)},
            ]
            _, (engine_input,) = await ctx.online_renderer.preprocess_chat(
                read_request,
                messages,
                default_template=ctx.chat_template,
                default_template_content_format=ctx.chat_template_content_format,
                default_template_kwargs=ctx.default_chat_template_kwargs,
            )
            prompt_ids = extract_prompt_components(
                ctx.engine_client.model_config, engine_input
            ).token_ids
            n = len(slot.prefix_ids)
            if not prompt_ids or list(prompt_ids[-n:]) != slot.prefix_ids:
                raise StructuredDecisionError(
                    f"question {q.id!r}: the chat template changed the answer "
                    "text before the label"
                )
            slots.append(slot)
            engine_inputs.append(engine_input)

        generators: list[AsyncGenerator[RequestOutput, None]] = [
            ctx.engine_client.generate(
                engine_input,
                SamplingParams(
                    max_tokens=1, temperature=0.0, logprob_token_ids=slot.label_ids
                ),
                f"{request_id}-{i}",
                lora_request=lora_request,
                priority=priority,
            )
            for i, (slot, engine_input) in enumerate(zip(slots, engine_inputs))
        ]
        results: list[RequestOutput | None] = [None] * len(generators)
        async for i, res in merge_async_iterators(*generators):
            results[i] = res

        reads = []
        for q, slot, result in zip(questions, slots, results):
            if result is None or not result.outputs or not result.outputs[0].logprobs:
                raise RuntimeError(f"question {q.id!r}: the read returned no logprobs")
            output = result.outputs[0]
            logprobs = output.logprobs[0]
            label_logprobs = [logprobs[t].logprob for t in slot.label_ids]
            reads.append(
                QuestionRead(
                    probs=label_softmax(label_logprobs),
                    label_mass=sum(math.exp(lp) for lp in label_logprobs),
                    argmax_is_label=bool(output.token_ids)
                    and output.token_ids[0] in slot.label_ids,
                    input_tokens=len(result.prompt_token_ids or ()),
                    output_tokens=len(output.token_ids),
                )
            )
        return reads


@register_read_strategy("canvas")
class CanvasStrategy(ReadStrategy):
    """Diffusion models. One request reads every question: the chat prompt, and
    a canvas seeded with the whole reply, each label slot left as noise. One
    denoise step gives the label probabilities at every slot."""

    CANVAS_STEP = 16

    def _canvas_length(self) -> int:
        vllm_config = self.context.engine_client.vllm_config
        if (config := vllm_config.diffusion_config) and config.canvas_length:
            return config.canvas_length
        return vllm_config.model_config.hf_config.canvas_length

    def limits(self) -> DecisionLimits:
        # Every answer takes at least three canvas positions: id, label and
        # separator.
        return DecisionLimits(
            max_questions=self._canvas_length() // 3,
            max_options=MAX_LOGPROB_TOKEN_IDS,
        )

    def _width(self, need: int) -> int:
        canvas = self._canvas_length()
        if need > canvas:
            raise StructuredDecisionError(
                f"the answers need {need} canvas positions and the canvas holds "
                f"{canvas}"
            )
        if not self.context.engine_client.vllm_config.scheduler_config.async_scheduling:
            return canvas
        return min(canvas, -(-need // self.CANVAS_STEP) * self.CANVAS_STEP)

    async def _render(self, messages: list[dict[str, str]], **flags: Any) -> list[int]:
        ctx = self.context
        _, (engine_input,) = await ctx.online_renderer.preprocess_chat(
            ReadPromptRequest(**flags),
            messages,
            default_template=ctx.chat_template,
            default_template_content_format=ctx.chat_template_content_format,
            default_template_kwargs=ctx.default_chat_template_kwargs,
        )
        ids = extract_prompt_components(
            ctx.engine_client.model_config, engine_input
        ).token_ids
        return list(ids or ())

    async def _end_of_reply(
        self, rendered: RenderedDecision, chat_template_kwargs: dict[str, Any] | None
    ) -> int:
        """The token the chat template puts right after a finished reply."""
        messages = [
            {"role": "system", "content": rendered.system_text},
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "x"},
        ]
        kwargs = {"chat_template_kwargs": chat_template_kwargs}
        open_ids = await self._render(messages, **kwargs)
        closed_ids = await self._render(
            messages, continue_final_message=False, **kwargs
        )
        if len(closed_ids) <= len(open_ids) or closed_ids[: len(open_ids)] != open_ids:
            raise StructuredDecisionError(
                "the chat template does not end a reply with a token"
            )
        return closed_ids[len(open_ids)]

    async def read(
        self,
        questions: list[Question],
        template: DecisionTemplate,
        instructions: str | None,
        state: str,
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
    ) -> list[QuestionRead]:
        ctx = self.context
        tokenizer = ctx.online_renderer.renderer.get_tokenizer()
        rendered = template.render(instructions, questions)
        reply, slots = rendered.reply_slots(tokenizer, questions)
        end = await self._end_of_reply(rendered, chat_template_kwargs)
        if tokenizer.pad_token_id is None:
            raise StructuredDecisionError("the tokenizer has no pad token")
        width = self._width(len(reply) + 1)

        canvas = reply + [end] + [tokenizer.pad_token_id] * (width - len(reply) - 1)
        rng = random.Random(request_id)
        vocab_size = ctx.engine_client.model_config.get_vocab_size()
        for slot in slots:
            canvas[slot.pos] = rng.randrange(vocab_size)

        messages = [
            {"role": "system", "content": rendered.system_text},
            {"role": "user", "content": state},
        ]
        _, (engine_input,) = await ctx.online_renderer.preprocess_chat(
            ReadPromptRequest(
                chat_template_kwargs=chat_template_kwargs,
                add_generation_prompt=True,
                continue_final_message=False,
            ),
            messages,
            default_template=ctx.chat_template,
            default_template_content_format=ctx.chat_template_content_format,
            default_template_kwargs=ctx.default_chat_template_kwargs,
        )
        params = SamplingParams(
            max_tokens=width,
            logprob_token_ids=sorted({t for s in slots for t in s.label_ids}),
            extra_args={
                "diffusion_seed_canvas": canvas,
                "diffusion_canvas_length": width,
                "diffusion_max_steps": 1,
                "diffusion_read_only": True,
            },
        )
        result = None
        async for result in ctx.engine_client.generate(
            engine_input,
            params,
            request_id,
            lora_request=lora_request,
            priority=priority,
        ):
            pass
        if result is None or not result.outputs or not result.outputs[0].logprobs:
            raise RuntimeError("the canvas read returned no logprobs")
        output = result.outputs[0]

        reads = []
        for i, slot in enumerate(slots):
            logprobs = output.logprobs[slot.pos]
            label_logprobs = [logprobs[t].logprob for t in slot.label_ids]
            reads.append(
                QuestionRead(
                    probs=label_softmax(label_logprobs),
                    label_mass=sum(math.exp(lp) for lp in label_logprobs),
                    argmax_is_label=output.token_ids[slot.pos] in slot.label_ids,
                    input_tokens=len(result.prompt_token_ids or ()) if i == 0 else 0,
                    output_tokens=len(output.token_ids) if i == 0 else 0,
                )
            )
        return reads
