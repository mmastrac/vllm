# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import dataclasses
import hashlib
import json
import random
from typing import Any

from fastapi import Request

from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.logger import init_logger

from .protocol import (
    DecisionUsage,
    Experiment,
    QuestionDiagnostics,
    StructuredDecisionRequest,
    StructuredDecisionResponse,
)
from .question_types import (
    LABELS,
    Question,
    StructuredDecisionError,
    build_question,
)
from .strategies import DecisionLimits, QuestionRead, ReadStrategy

logger = init_logger(__name__)


def state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def parse_questions(
    request: StructuredDecisionRequest,
    limits: DecisionLimits,
) -> list[Question]:
    if not request.questions:
        raise StructuredDecisionError("questions: needs at least one question")
    if len(request.questions) > limits.max_questions:
        raise StructuredDecisionError(
            f"questions: at most {limits.max_questions} for this model"
        )
    questions = []
    for qid, spec in request.questions.items():
        if spec.model_extra:
            raise StructuredDecisionError(
                f"question {qid!r}: unknown field(s) {sorted(spec.model_extra)}"
            )
        questions.append(
            build_question(
                qid,
                spec.type,
                spec.instructions,
                spec.criteria,
                limits.max_options,
            )
        )
    return questions


def experiment_variants(
    q: Question, experiment: Experiment
) -> list[tuple[Question, list[int]]]:
    """The reads of ``q``, each with ``perm``: read option ``j`` is
    ``q.options[perm[j]]``. Labels are shuffled for a choice only, and
    options for a choice or a noul."""
    variants = []
    for s in range(experiment.samples):
        seed = experiment.seed + s
        key = json.dumps([q.id, q.instructions, [o.name for o in q.options], seed])
        rng = random.Random(hashlib.sha256(key.encode()).digest())
        perm = list(range(len(q.options)))
        if experiment.options == "random" and q.type.name in ("choice", "noul"):
            rng.shuffle(perm)
        labels = q.labels
        if experiment.labels == "random" and q.type.name == "choice":
            labels = tuple(rng.sample(LABELS, len(q.options)))
        variant = dataclasses.replace(
            q,
            options=tuple(q.options[i] for i in perm),
            labels=labels,
            note=experiment.note,
            seed=seed,
        )
        variants.append((variant, perm))
    return variants


def fold_reads(
    variants: list[tuple[Question, list[int]]], reads: list[QuestionRead]
) -> QuestionRead:
    """The mean of a question's reads, in its own option order."""
    n = len(variants)
    probs = [0.0] * len(variants[0][1])
    for (_, perm), read in zip(variants, reads):
        for j, p in enumerate(read.probs):
            probs[perm[j]] += p / n
    return QuestionRead(
        probs=probs,
        label_mass=sum(r.label_mass for r in reads) / n,
        argmax_is_label=all(r.argmax_is_label for r in reads),
        input_tokens=sum(r.input_tokens for r in reads),
        output_tokens=sum(r.output_tokens for r in reads),
    )


class ServingStructuredDecisions(BaseServing):
    def __init__(
        self,
        models: OpenAIServingModels,
        strategy: ReadStrategy,
        *,
        request_logger: RequestLogger | None = None,
    ) -> None:
        model_config = strategy.context.engine_client.model_config
        super().__init__(
            models=models, model_config=model_config, request_logger=request_logger
        )
        self.strategy = strategy
        self.limits = strategy.limits()

    async def create_decision(
        self,
        request: StructuredDecisionRequest,
        raw_request: Request | None = None,
    ) -> StructuredDecisionResponse | ErrorResponse:
        if (error := await self._check_model(request)) is not None:  # type: ignore[arg-type]
            return error
        engine_client = self.strategy.context.engine_client
        if engine_client.errored:
            raise engine_client.dead_error

        base_id = self._base_request_id(raw_request, default=request.request_id)
        request_id = f"decision-{base_id}"
        try:
            questions = parse_questions(request, self.limits)
            lora_request = self._maybe_get_adapters(request)  # type: ignore[arg-type]
            variants = [
                experiment_variants(q, request.experiment or Experiment())
                for q in questions
            ]
            engine_client.check_admission(sum(len(v) for v in variants))
            flat = await self.strategy.read(
                [v for vs in variants for v, _ in vs],
                request.instructions,
                state_text(request.state),
                request_id=request_id,
                chat_template_kwargs=request.chat_template_kwargs,
                lora_request=lora_request,
                priority=request.priority,
            )
            reads, i = [], 0
            for vs in variants:
                reads.append(fold_reads(vs, flat[i : i + len(vs)]))
                i += len(vs)
        except StructuredDecisionError as e:
            return self.create_error_response(e)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        answers: dict[str, dict[str, Any]] = {}
        diagnostics: dict[str, QuestionDiagnostics] = {}
        for q, read in zip(questions, reads):
            answers[q.id] = q.type.answer(q, read.probs, read.label_mass)
            diagnostics[q.id] = QuestionDiagnostics(
                label_mass=read.label_mass, argmax_is_label=read.argmax_is_label
            )
        return StructuredDecisionResponse(
            id=request_id,
            model=self.models.model_name(lora_request),
            answers=answers,
            usage=DecisionUsage(
                input_tokens=sum(r.input_tokens for r in reads),
                output_tokens=sum(r.output_tokens for r in reads),
            ),
            diagnostics=diagnostics,
        )
