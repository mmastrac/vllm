# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured decisions in front of a vLLM DiffusionGemma server.

POST /v1/chat/completions with a system message that is the question schema
and a user message that is the state. The reply's `content` is the JSON
answer set: one calibrated distribution per question, from a single denoise
step over a seeded canvas, averaged over a few noise draws. The canvas,
tokenizer, slot resolution, noise draws and averaging stay behind this server.

Schema (system message):
  {"questions": [
     {"id": "urgent", "type": "noul", "instructions": "..."},
     {"id": "bucket", "type": "choice", "instructions": "...",
      "options": [{"name": "billing", "description": "..."}, ...]},
     {"id": "tone", "type": "score", "instructions": "...",
      "levels": ["calm", "annoyed", "furious"]}],
   "instructions": "optional context",
   "samples": "auto" | N, "auto_threshold": 0.1, "auto_max": 4,
   "steps": 1}

A schema whose answer lines do not fit the canvas is split into chunks
that run together, each with its own question list ("chunk_rows" sets the
rows per chunk, "ask" picks a subset of question ids for one read).

Serve the model with a canvas that holds the answer template, for example:
  vllm serve google/diffusiongemma-26B-A4B-it \
      --diffusion-config '{"canvas_length": 64}' --max-logprobs 32 --enable-prefix-caching
then run this in front of it:
  python structured_server.py --upstream http://127.0.0.1:8000 \
      --tokenizer google/diffusiongemma-26B-A4B-it --canvas 64 --port 8011
"""
import argparse, json, math, random, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from transformers import AutoTokenizer

ARGS = None
TOK = None
CANVAS_LEN = 64      # the served canvas; a request may run narrower
CANVAS_STEP = 16     # request widths are multiples of this
VOCAB = 262144
TURN_CLOSE = 106
PAD = 0
TOPK = 20
SCAFFOLD_TEXT = "<|channel>thought\n<channel|>"  # the empty thought block the chat template leaves to the model
SCAFFOLD = None


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------

class SchemaError(ValueError):
    pass


def parse_schema(value):
    if not isinstance(value, dict) or not isinstance(value.get("questions"), list) or not value["questions"]:
        raise SchemaError("schema: needs a non-empty questions array")
    qs = []
    seen = set()
    for q in value["questions"]:
        qid = str(q.get("id", "")).strip()
        if not qid or ":" in qid or "\n" in qid:
            raise SchemaError(f"question id {qid!r} must be non-empty, no ':' or newline")
        if qid in seen:
            raise SchemaError(f"duplicate question id {qid!r}")
        seen.add(qid)
        kind = q.get("type")
        if kind in ("noul", "bool", "boolean"):
            kind = "noul"
            choices = [("yes", None), ("no", None)]
            labels = ["yes", "no"]
        elif kind == "choice":
            opts = q.get("options") or []
            choices = [(o["name"], o.get("description")) if isinstance(o, dict) else (str(o), None) for o in opts]
            labels = [chr(ord("A") + i) for i in range(len(choices))]
        elif kind == "score":
            choices = [(str(l), None) for l in (q.get("levels") or [])]
            labels = [str(i + 1) for i in range(len(choices))] if len(choices) <= 9 else [chr(ord("A") + i) for i in range(len(choices))]
        else:
            raise SchemaError(f"question {qid!r}: unknown type {kind!r}")
        if len(choices) < 2:
            raise SchemaError(f"question {qid!r}: needs at least two alternatives")
        if len(choices) > 26:
            raise SchemaError(f"question {qid!r}: at most 26 alternatives")
        qs.append({"id": qid, "type": kind, "instructions": str(q.get("instructions", "")), "choices": choices, "labels": labels})
    samples = value.get("samples", "auto")
    if samples == "auto":
        policy = {"mode": "auto", "max": int(value.get("auto_max", 4)), "threshold": float(value.get("auto_threshold", 0.1))}
    elif isinstance(samples, int) and samples >= 1:
        policy = {"mode": "fixed", "n": min(samples, 32)}
    else:
        raise SchemaError("schema: samples must be a positive count or \"auto\"")
    ask = value.get("ask")
    if ask is not None:
        if not isinstance(ask, list) or not ask or any(a not in seen for a in ask):
            raise SchemaError("schema: ask must list question ids from this schema")
    chunk_rows = value.get("chunk_rows")
    if chunk_rows is not None and (not isinstance(chunk_rows, int) or chunk_rows < 8):
        raise SchemaError("schema: chunk_rows must be an integer of at least 8")
    chunk_prompt = value.get("chunk_prompt", "own")
    if chunk_prompt not in ("shared", "own"):
        raise SchemaError("schema: chunk_prompt must be \"shared\" or \"own\"")
    sequential = bool(value.get("sequential", False))
    return {"questions": qs, "instructions": value.get("instructions"), "policy": policy,
            "steps": max(1, min(int(value.get("steps", 1)), 8)),
            "ask": ask, "chunk_rows": chunk_rows, "chunk_prompt": chunk_prompt, "sequential": sequential}


def system_text(schema, chunked=False):
    s = ("Answer a fixed set of questions about the state the user provides. "
         "Each question lists its allowed answers; reply with exactly one label per question.\n")
    if schema.get("instructions"):
        s += "\n" + str(schema["instructions"]).strip() + "\n"
    for q in schema["questions"]:
        s += f"\nQuestion {q['id']}: {q['instructions'].strip()}\n"
        for (name, desc), label in zip(q["choices"], q["labels"]):
            if q["type"] == "noul":
                s += f"  {label}\n"
            elif desc:
                s += f"  {label}: {name} ({str(desc).strip()})\n"
            else:
                s += f"  {label}: {name}\n"
    s += '\nReply with one line per question, in this order, formatted as "id: label".'
    if chunked:
        s += " A reply may cover only some of the questions; answer every line that is present."
    return s


def answer_text(qs, labels):
    return "\n".join(f"{q['id']}: {q['labels'][l]}" for q, l in zip(qs, labels))


def enc(text):
    return TOK.encode(text, add_special_tokens=False)


def resolve_template(qs, scaffold=True):
    """Tokenize the answer template and find each question's slot. Every label
    must change exactly one token, at the same position for all of a question's
    labels, or the schema is refused. ``scaffold`` is False for a continuation
    whose thought block and earlier answer lines are already in the prompt."""
    head = SCAFFOLD if scaffold else []
    base_labels = [0] * len(qs)
    base = head + enc(answer_text(qs, base_labels))
    if len(base) + 1 > CANVAS_LEN:
        raise SchemaError(f"answer template is {len(base)} tokens; the canvas holds {CANVAS_LEN - 1}")
    if len(qs) == 1 and len(base) + 1 > CANVAS_LEN:
        raise SchemaError(f"question {qs[0]['id']!r} alone needs {len(base) + 1} canvas rows")
    slots = []
    for qi, q in enumerate(qs):
        pos = None
        ids = [0] * len(q["labels"])
        for li in range(1, len(q["labels"])):
            labels = list(base_labels)
            labels[qi] = li
            e = head + enc(answer_text(qs, labels))
            if len(e) != len(base):
                raise SchemaError(f"question {q['id']!r}: label {q['labels'][li]!r} is not a single token")
            diffs = [i for i in range(len(e)) if e[i] != base[i]]
            if len(diffs) != 1 or (pos is not None and diffs[0] != pos):
                raise SchemaError(f"question {q['id']!r}: labels do not share one template slot")
            pos = diffs[0]
            ids[li] = e[pos]
        ids[0] = base[pos]
        if len(set(ids)) != len(ids):
            raise SchemaError(f"question {q['id']!r}: two labels tokenize to the same id")
        slots.append({"pos": pos, "label_ids": ids})
    return base, slots


_template_cache = {}


def template_for(schema, scaffold=True):
    key = json.dumps([scaffold] + [(q["id"], q["labels"]) for q in schema["questions"]])
    if key not in _template_cache:
        _template_cache[key] = resolve_template(schema["questions"], scaffold)
    return _template_cache[key]


# ----------------------------------------------------------------------------
# Reads
# ----------------------------------------------------------------------------

def canvas_width(template):
    """Smallest multiple of CANVAS_STEP that holds the template and the turn close."""
    need = len(template) + 1
    return min(CANVAS_LEN, -(-need // CANVAS_STEP) * CANVAS_STEP)


def build_canvas(template, slots, seed):
    rng = random.Random(seed)
    canvas = list(template) + [TURN_CLOSE]
    canvas += [PAD] * (canvas_width(template) - len(canvas))
    for s in slots:
        canvas[s["pos"]] = rng.randrange(VOCAB)
    return canvas


def label_id_union(slots):
    ids = sorted({i for s in slots for i in s["label_ids"]})
    return ids[:128]  # vLLM's cap per request; a schema needs far fewer


def upstream_chat(body, timeout=600):
    req = urllib.request.Request(ARGS.upstream.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def chat_prompt_ids(sys_text, state_text):
    """The prompt the chat endpoint would build, as token ids, ending after
    the model turn marker. Text states only."""
    messages = [{"role": "system", "content": sys_text}, {"role": "user", "content": state_text}]
    out = TOK.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    ids = out["input_ids"] if hasattr(out, "keys") else out  # newer transformers return a dict
    return [int(t) for t in ids]


def one_read(schema, template, slots, sys_text, state_content, seed, prefix=None):
    if prefix is not None:
        return one_read_continuation(schema, template, slots, prefix, seed)
    messages = [{"role": "system", "content": sys_text}, {"role": "user", "content": state_content}]
    body = {
        "model": ARGS.model,
        "messages": messages,
        "max_tokens": len(template) + 1,
        "logprobs": True,
        "top_logprobs": TOPK,
        # Exact logprobs for every label at every position. With a long
        # option list most labels never rank in the top-k, and the model's
        # mass sits on tokens that spell the option name instead.
        "logprob_token_ids": label_id_union(slots),
        "return_tokens_as_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "vllm_xargs": {"diffusion_seed_canvas": build_canvas(template, slots, seed), "diffusion_canvas_length": canvas_width(template),
                       "diffusion_max_steps": schema["steps"], "diffusion_read_only": True},
    }
    d = upstream_chat(body)
    content = d["choices"][0]["logprobs"]["content"]
    out = []
    for q, s in zip(schema["questions"], slots):
        top = {int(t["token"].split(":")[1]): t["logprob"] for t in content[s["pos"]]["top_logprobs"]}
        out.append(slot_distribution(top, s["label_ids"]))
    return out, d.get("usage", {})


def slot_distribution(top, label_ids):
    """Label probabilities at one slot from the returned logprobs: every
    label's own value plus the argmax token. Read-only logprobs are at
    temperature 1, so the label softmax uses them directly. The entropy is
    over that returned set."""
    floor = min(top.values()) - 5.0
    lp_t = [top.get(i, floor) for i in label_ids]
    mx = max(lp_t)
    ex = [math.exp(x - mx) for x in lp_t]
    probs = [e / sum(ex) for e in ex]
    top_p = [math.exp(v) for v in top.values()]
    return {
        "probs": probs,
        "label_mass": sum(math.exp(x) for x in lp_t),
        "entropy": -sum(p * math.log(p) for p in top_p if p > 0),
        "argmax_is_label": max(top, key=top.get) in label_ids,
    }


def one_read_continuation(schema, template, slots, prompt_ids, seed):
    """A read whose prompt already holds the thought scaffold and earlier
    answer lines, sent as token ids so the chat template cannot alter it."""
    body = {
        "model": ARGS.model,
        "prompt": prompt_ids,
        "max_tokens": len(template) + 1,
        "logprobs": TOPK,
        "logprob_token_ids": label_id_union(slots),
        "return_tokens_as_token_ids": True,
        "vllm_xargs": {"diffusion_seed_canvas": build_canvas(template, slots, seed), "diffusion_canvas_length": canvas_width(template),
                       "diffusion_max_steps": schema["steps"], "diffusion_read_only": True},
    }
    req = urllib.request.Request(ARGS.upstream.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    rows = d["choices"][0]["logprobs"]["top_logprobs"]
    out = []
    for q, sl in zip(schema["questions"], slots):
        top = {int(k.split(":")[1]): v for k, v in rows[sl["pos"]].items()}
        out.append(slot_distribution(top, sl["label_ids"]))
    return out, d.get("usage", {})


def read_many(schema, template, slots, sys_text, state_content, seed, n, prefix=None):
    results = [None] * n
    errors = [None] * n

    def run(k):
        try:
            results[k], _ = one_read(schema, template, slots, sys_text, state_content, seed + k * 7919, prefix)
        except Exception as e:  # surfaced as one failed request below
            errors[k] = e

    threads = [threading.Thread(target=run, args=(k,)) for k in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in errors:
        if e is not None:
            raise e
    return results


def question_groups(schema):
    """The questions of one read each. ``ask`` picks a subset; otherwise the
    questions are split, in order, into the fewest groups whose answer
    templates fit ``chunk_rows`` (the canvas by default)."""
    qs = schema["questions"]
    if schema.get("ask"):
        wanted = set(schema["ask"])
        return [[q for q in qs if q["id"] in wanted]]
    limit = schema.get("chunk_rows") or CANVAS_LEN
    groups, group = [], []
    for q in qs:
        trial = group + [q]
        rows = len(SCAFFOLD) + len(enc(answer_text(trial, [0] * len(trial)))) + 1
        if rows > limit and group:
            groups.append(group)
            group = [q]
        else:
            group = trial
    groups.append(group)
    return groups


def decide(schema, state_content, seed):
    """One decision, as one read set or several chunked ones run together."""
    groups = question_groups(schema)
    chunked = len(groups) > 1
    started = time.time()
    if not chunked:
        body, rows = decide_group(dict(schema, questions=groups[0]), system_text(schema), state_content, seed)
        return body, rows
    # Each chunk lists its own questions by default: a chunk answering a
    # subset of a longer list loses alignment and confidence (measured on
    # per-word PII: 3 of 42 decisions flipped, none with own lists).
    shared = schema["chunk_prompt"] == "shared"
    sys_shared = system_text(schema, chunked=True)

    def one(k_group):
        k, group = k_group
        sub = dict(schema, questions=group)
        sys_text = sys_shared if shared else system_text(sub)
        return decide_group(sub, sys_text, state_content, seed + 104729 * k)

    if schema["sequential"]:
        # Chunks continue one answer in order under the full question list.
        # Each chunk's argmax lines are prefilled before the next, so later
        # answers condition on earlier ones (conditionals, not marginals).
        sys_text = system_text(schema)
        if not isinstance(state_content, str):
            raise SchemaError("sequential chunks need a text state (images go through the chat endpoint)")
        base_ids = chat_prompt_ids(sys_text, state_content) + SCAFFOLD
        lines = []
        parts = []
        for k, group in enumerate(groups):
            sub = dict(schema, questions=group)
            prefix = base_ids + enc("".join(line + "\n" for line in lines)) if lines else None
            body, rows = decide_group(sub, sys_text, state_content, seed + 104729 * k, prefix=prefix)
            parts.append((body, rows))
            chosen = [q["labels"].index(body["answers"][q["id"]]["label"]) for q in group]
            lines.append(answer_text(group, chosen))
    else:
        with ThreadPoolExecutor(max_workers=len(groups)) as ex:
            parts = list(ex.map(one, enumerate(groups)))
    answers, diag_q = {}, {}
    for body, _ in parts:
        answers.update(body["answers"])
        diag_q.update(body["diagnostics"]["questions"])
    return {
        "answers": answers,
        "diagnostics": {
            "steps": schema["steps"],
            "chunks": [[q["id"] for q in g] for g in groups],
            "chunk_prompt": "full" if schema["sequential"] else schema["chunk_prompt"],
            "sequential": schema["sequential"],
            "samples": {"n": [b["diagnostics"]["samples"]["n"] for b, _ in parts],
                        "tops": [b["diagnostics"]["samples"]["tops"] for b, _ in parts],
                        "policy": [b["diagnostics"]["samples"]["policy"] for b, _ in parts]},
            "timing": {"total_ms": (time.time() - started) * 1e3,
                       "reads": sum(b["diagnostics"]["timing"]["reads"] for b, _ in parts)},
            "questions": diag_q,
            "engine": "vllm",
        },
    }, sum(rows for _, rows in parts)


def decide_group(schema, sys_text, state_content, seed, prefix=None):
    template, slots = template_for(schema, scaffold=prefix is None)
    started = time.time()
    policy = schema["policy"]
    if policy["mode"] == "fixed":
        reads = read_many(schema, template, slots, sys_text, state_content, seed, policy["n"], prefix)
        extended = None
        first_entropy = None
    else:
        reads = read_many(schema, template, slots, sys_text, state_content, seed, 1, prefix)
        first_entropy = {q["id"]: r["entropy"] for q, r in zip(schema["questions"], reads[0])}
        extended = max(first_entropy.values()) > policy["threshold"] and policy["max"] > 1
        if extended:
            reads += read_many(schema, template, slots, sys_text, state_content, seed + 1, policy["max"] - 1, prefix)
    elapsed_ms = (time.time() - started) * 1e3

    answers = {}
    diag_q = {}
    n = len(reads)
    for qi, q in enumerate(schema["questions"]):
        per = [r[qi]["probs"] for r in reads]
        mean = [sum(p[l] for p in per) / n for l in range(len(q["labels"]))]
        top = max(range(len(mean)), key=lambda l: mean[l])
        a = {"type": q["type"], "label": q["labels"][top], "confidence": mean[top],
             "probabilities": {c[0]: m for c, m in zip(q["choices"], mean)}}
        if q["type"] == "noul":
            a["noul"] = mean[0]
        elif q["type"] == "choice":
            a["choice"] = q["choices"][top][0]
        else:
            a["score"] = sum((i + 1) * m for i, m in enumerate(mean))
            a["level"] = q["choices"][top][0]
        if n > 1:
            var = sum((p[top] - mean[top]) ** 2 for p in per) / (n - 1)
            a["stderr"] = (var / n) ** 0.5
            a["agreement"] = sum(1 for p in per if max(range(len(p)), key=lambda l: p[l]) == top) / n
        answers[q["id"]] = a
        diag_q[q["id"]] = {"pos": slots[qi]["pos"], "entropy": [r[qi]["entropy"] for r in reads],
                           "label_mass": reads[0][qi]["label_mass"], "argmax_is_label": reads[0][qi]["argmax_is_label"]}
    tops = [{q["id"]: [q["labels"][max(range(len(r[qi]["probs"])), key=lambda l: r[qi]["probs"][l])],
                       max(r[qi]["probs"]), r[qi]["entropy"]] for qi, q in enumerate(schema["questions"])} for r in reads]
    return {
        "answers": answers,
        "diagnostics": {
            "steps": schema["steps"],
            "samples": {"n": n, "tops": tops, "policy": dict(policy, extended=extended, first_read_entropy=first_entropy)},
            "timing": {"total_ms": elapsed_ms, "reads": n},
            "questions": diag_q,
            "engine": "vllm",
        },
    }, len(template) + 1


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------

def message_text(m):
    c = m.get("content", "")
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c if isinstance(c, str) else ""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        return self._json(404, {"error": {"message": "unknown route"}})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": {"message": "unknown route"}})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))))
        except Exception as e:
            return self._json(400, {"error": {"message": f"invalid JSON body: {e}", "type": "invalid_request_error"}})
        msgs = req.get("messages") or []
        if len(msgs) != 2 or msgs[0].get("role") not in ("system", "developer") or msgs[1].get("role") != "user":
            return self._json(400, {"error": {"message": "a structured request is exactly two messages: the schema (system) and the state JSON (user)", "type": "invalid_request_error"}})
        try:
            schema_value = json.loads(message_text(msgs[0]))
            schema = parse_schema(schema_value)
            content = msgs[1].get("content", "")
            has_image = isinstance(content, list) and any(
                isinstance(p, dict) and p.get("type") in ("image_url", "image") for p in content)
            if has_image:
                # image parts pass through to vLLM as they are; any text part is context
                state = content
            else:
                state = message_text(msgs[1]).strip()
                json.loads(state)
        except SchemaError as e:
            return self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
        except Exception as e:
            return self._json(400, {"error": {"message": f"system must be a JSON question schema and user must be JSON state or image parts: {e}", "type": "invalid_request_error"}})
        seed = int(req.get("seed", 42))
        try:
            body, completion_tokens = decide(schema, state, seed)
        except SchemaError as e:
            return self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
        except urllib.error.HTTPError as e:
            return self._json(502, {"error": {"message": f"upstream {e.code}: {e.read()[:300].decode(errors='replace')}", "type": "server_error"}})
        except Exception as e:
            return self._json(500, {"error": {"message": repr(e), "type": "server_error"}})
        content = json.dumps(body, indent=2)
        labels = " ".join(f"{k}={v['label']}" for k, v in body["answers"].items())
        print(f"structured: {labels} reads={body['diagnostics']['samples']['n']} {body['diagnostics']['timing']['total_ms']:.0f}ms", flush=True)
        self._json(200, {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", "dgemma-structured"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": completion_tokens, "total_tokens": completion_tokens},
        })


def main():
    global ARGS, TOK, SCAFFOLD, CANVAS_LEN, CANVAS_STEP
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default="http://127.0.0.1:8010")
    p.add_argument("--model", default="dgemma")
    p.add_argument("--tokenizer", default="/models/dgemma", help="HF id or local path")
    p.add_argument("--canvas", type=int, default=64, help="the served canvas length")
    p.add_argument("--canvas-step", type=int, default=16, help="request widths round up to a multiple of this")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8011)
    ARGS = p.parse_args()
    CANVAS_LEN = ARGS.canvas
    CANVAS_STEP = ARGS.canvas_step
    TOK = AutoTokenizer.from_pretrained(ARGS.tokenizer)
    SCAFFOLD = enc(SCAFFOLD_TEXT)
    print(f"structured server on {ARGS.host}:{ARGS.port} -> {ARGS.upstream} (canvas {CANVAS_LEN})", flush=True)
    ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
