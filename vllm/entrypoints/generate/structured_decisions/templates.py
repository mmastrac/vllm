# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decision templates: Jinja that renders a decision's system prompt and says
how an answer is written.

The template receives ``instructions`` (a string or None) and ``questions``,
each with ``id``, ``type``, ``instructions`` and ``options`` (``label``,
``name``, ``description``).

It may define a macro ``answer(question, label)`` that returns one question's
answer as the model should write it, ``"<id>: <label>"`` by default. The
server renders the answer once per label and compares the tokens to find
where the label goes. The system prompt can call the same macro to show the
reply format, so the prompt shows the model the format the server reads.

A read that writes every answer at once joins them with ``answer_separator()``,
a newline by default, after ``reply_start()``, the text the model writes before
its answers, empty by default.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jinja2
import jinja2.ext
import jinja2.sandbox

from vllm.tokenizers import TokenizerLike

from .question_types import Question, StructuredDecisionError

DEFAULT_DECISION_TEMPLATE = """\
{%- macro answer(question, label) -%}{{ question.id }}: {{ label }}{%- endmacro -%}
Answer a fixed set of questions about the state the user provides. Each \
question lists its allowed answers; reply with exactly one label per question.
{% if instructions %}

{{ instructions | trim }}
{% endif %}
{% for question in questions %}

Question {{ question.id }}: {{ question.instructions | trim }}
{% for option in question.options %}
  {{ option.label }}: {{ option.name }}\
{% if option.description %} ({{ option.description | trim }}){% endif %}

{% endfor %}
{% endfor %}

Reply with one line per question, in this order, formatted as \
"{{ answer({"id": "id"}, "label") }}".\
"""


def question_vars(question: Question) -> dict[str, Any]:
    return {
        "id": question.id,
        "type": question.type.name,
        "instructions": question.instructions,
        "options": [
            {"label": label, "name": o.name, "description": o.description}
            for label, o in zip(question.labels, question.options)
        ],
    }


@contextmanager
def template_errors() -> Iterator[None]:
    """Report a template that fails to compile or render as a 400."""
    try:
        yield
    except jinja2.TemplateError as e:
        raise StructuredDecisionError(f"decision template: {e}") from e


@dataclass(frozen=True)
class CanvasSlot:
    pos: int
    label_ids: list[int]


@dataclass(frozen=True)
class AnswerSlot:
    """Where a question's label goes in its rendered answer."""

    prefix_ids: list[int]
    """The answer's tokens before the label."""
    label_ids: list[int]
    """The label token for each label, in label order."""


class DecisionTemplate:
    def __init__(self, source: str):
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[jinja2.ext.loopcontrols],
        )
        with template_errors():
            self._template = env.from_string(source)

    def render(
        self, instructions: str | None, questions: list[Question]
    ) -> "RenderedDecision":
        with template_errors():
            # The module's text is the rendered template, and its macros see
            # the same variables as the template body.
            module = self._template.make_module(
                vars={
                    "instructions": instructions,
                    "questions": [question_vars(q) for q in questions],
                }
            )
        return RenderedDecision(
            system_text=str(module),
            answer_macro=getattr(module, "answer", None),
            separator_macro=getattr(module, "answer_separator", None),
            reply_start_macro=getattr(module, "reply_start", None),
        )


@dataclass(frozen=True)
class RenderedDecision:
    system_text: str
    answer_macro: Any
    separator_macro: Any
    reply_start_macro: Any

    def answer(self, question: Question, label: str) -> str:
        if self.answer_macro is None:
            return f"{question.id}: {label}"
        with template_errors():
            return str(self.answer_macro(question_vars(question), label))

    def _macro_text(self, macro: Any, default: str) -> str:
        if macro is None:
            return default
        with template_errors():
            return str(macro())

    def slot(self, tokenizer: TokenizerLike, question: Question) -> AnswerSlot:
        """Where the label goes in the question's answer on its own."""
        variants = [
            tokenizer.encode(self.answer(question, label), add_special_tokens=False)
            for label in question.labels
        ]
        pos, label_ids = label_position(question, variants)
        if pos == 0:
            raise StructuredDecisionError(
                f"question {question.id!r}: the answer must have text before the "
                "label, such as the question id"
            )
        return AnswerSlot(prefix_ids=variants[0][:pos], label_ids=label_ids)

    def reply_slots(
        self, tokenizer: TokenizerLike, questions: list[Question]
    ) -> tuple[list[int], list[CanvasSlot]]:
        """The reply with every answer written at its first label, and where
        each question's label goes in it."""
        separator = self._macro_text(self.separator_macro, "\n")
        start = self._macro_text(self.reply_start_macro, "")

        def encode(labels: list[str]) -> list[int]:
            answers = separator.join(
                self.answer(q, label) for q, label in zip(questions, labels)
            )
            return tokenizer.encode(start + answers, add_special_tokens=False)

        first = [q.labels[0] for q in questions]
        reply = encode(first)
        slots = []
        for i, q in enumerate(questions):
            variants = [reply] + [
                encode(first[:i] + [label] + first[i + 1 :]) for label in q.labels[1:]
            ]
            pos, label_ids = label_position(q, variants)
            slots.append(CanvasSlot(pos=pos, label_ids=label_ids))
        return reply, slots


def label_position(
    question: Question, variants: list[list[int]]
) -> tuple[int, list[int]]:
    """``variants`` holds one tokenization per label of the same text. Returns
    the one position where they differ and each label's token there. Every
    label must be one token, the rest of the text must not change with the
    label, and no two labels may share a token."""
    if any(len(ids) != len(variants[0]) for ids in variants):
        raise StructuredDecisionError(
            f"question {question.id!r}: its labels are not all one token in the answer"
        )
    diffs = {
        i
        for ids in variants[1:]
        for i, (a, b) in enumerate(zip(variants[0], ids))
        if a != b
    }
    if len(diffs) != 1:
        raise StructuredDecisionError(
            f"question {question.id!r}: the labels change {len(diffs)} tokens of "
            "the answer, and must change exactly one"
        )
    (pos,) = diffs
    label_ids = [ids[pos] for ids in variants]
    if len(set(label_ids)) != len(label_ids):
        raise StructuredDecisionError(
            f"question {question.id!r}: two labels share a token"
        )
    return pos, label_ids


@lru_cache(maxsize=16)
def compile_template(source: str) -> DecisionTemplate:
    return DecisionTemplate(source)


def select_template(
    server_source: str | None,
    request_source: str | None,
    trust_request_template: bool,
) -> DecisionTemplate:
    """The template for one request: its own when the server trusts request
    templates, otherwise the server's, otherwise the default."""
    if request_source is not None:
        if not trust_request_template:
            raise StructuredDecisionError(
                "decision_template in a request needs the server to run with "
                "--trust-request-chat-template"
            )
        return compile_template(request_source)
    return compile_template(server_source or DEFAULT_DECISION_TEMPLATE)
