# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiffusionGemma borrows Gemma4's multimodal methods. Every attribute those
methods read from ``self`` must exist on DiffusionGemma too, or the first
image request raises AttributeError inside the engine."""

import ast
import inspect
import textwrap

from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaForConditionalGeneration as DiffusionGemma,
)
from vllm.model_executor.models.gemma4_mm import (
    Gemma4ForConditionalGeneration as Gemma4,
)

# Reached only for audio inputs, which DiffusionGemma's processor never emits.
AUDIO_ONLY = {"_parse_and_validate_audio_input", "_process_audio_input"}


def _function(obj):
    return getattr(obj, "__func__", obj)


def _borrowed() -> dict[str, object]:
    out = {}
    for name, value in vars(DiffusionGemma).items():
        fn = _function(value)
        if callable(fn) and _function(vars(Gemma4).get(name)) is fn:
            out[name] = fn
    return out


def _self_reads(fn) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, ast.Load)
    }


def _init_writes() -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(DiffusionGemma.__init__)))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, ast.Store)
    }


def test_borrowed_gemma4_methods_find_their_attributes():
    borrowed = _borrowed()
    assert "_process_image_input" in borrowed
    assert "embed_multimodal" in borrowed

    available = _init_writes()
    missing = {
        f"{method}: self.{attr}"
        for method, fn in borrowed.items()
        for attr in _self_reads(fn) - AUDIO_ONLY
        if attr not in available and not hasattr(DiffusionGemma, attr)
    }
    assert not missing, sorted(missing)
