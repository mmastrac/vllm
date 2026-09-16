# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured decisions in front of a vLLM DiffusionGemma server.

POST /v1/chat/completions with a system message that is the question schema
and a user message that is the state. The reply's `content` is the JSON
answer set: one calibrated distribution per question, from a single denoise
step over a seeded canvas, averaged over a few noise draws. The canvas, the
tokenizer, the slot resolution, the noise draws and the averaging all stay
behind this server; clients never see a token id.

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

Serve the model with a canvas that holds the answer template, for example:
  vllm serve google/diffusiongemma-26B-A4B-it \
      --diffusion-config '{"canvas_length": 64}' --max-logprobs 32 --enable-prefix-caching
then run this in front of it:
  python structured_server.py --upstream http://127.0.0.1:8000 \
      --tokenizer google/diffusiongemma-26B-A4B-it --canvas 64 --port 8011
"""
import argparse, json, math, random, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from transformers import AutoTokenizer

ARGS = None
TOK = None
CANVAS_LEN = 64
VOCAB = 262144
TURN_CLOSE = 106
PAD = 0
TOPK = 20
SCAFFOLD = None      # <|channel>thought\n<channel|>: the chat template ends at <|turn>model\n


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
    return {"questions": qs, "instructions": value.get("instructions"), "policy": policy,
            "steps": max(1, min(int(value.get("steps", 1)), 8))}


def system_text(schema):
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
    return s


def answer_text(qs, labels):
    return "\n".join(f"{q['id']}: {q['labels'][l]}" for q, l in zip(qs, labels))


def enc(text):
    return TOK.encode(text, add_special_tokens=False)


def resolve_template(qs):
    """Tokenize the answer template and find each question's slot. Every label
    must change exactly one token, at the same position for all of a question's
    labels, or the schema is refused."""
    base_labels = [0] * len(qs)
    base = SCAFFOLD + enc(answer_text(qs, base_labels))
    if len(base) + 1 > CANVAS_LEN:
        raise SchemaError(f"answer template is {len(base)} tokens; the canvas holds {CANVAS_LEN - 1}")
    slots = []
    for qi, q in enumerate(qs):
        pos = None
        ids = [0] * len(q["labels"])
        for li in range(1, len(q["labels"])):
            labels = list(base_labels)
            labels[qi] = li
            e = SCAFFOLD + enc(answer_text(qs, labels))
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


def template_for(schema):
    key = json.dumps([(q["id"], q["labels"]) for q in schema["questions"]])
    if key not in _template_cache:
        _template_cache[key] = resolve_template(schema["questions"])
    return _template_cache[key]


# ----------------------------------------------------------------------------
# Reads
# ----------------------------------------------------------------------------

def build_canvas(template, slots, seed):
    rng = random.Random(seed)
    canvas = list(template) + [TURN_CLOSE]
    canvas += [PAD] * (CANVAS_LEN - len(canvas))
    for s in slots:
        canvas[s["pos"]] = rng.randrange(VOCAB)
    return canvas


def upstream_chat(body, timeout=600):
    req = urllib.request.Request(ARGS.upstream.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def one_read(schema, template, slots, sys_text, state_text, seed):
    body = {
        "model": ARGS.model,
        "messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": state_text}],
        "max_tokens": len(template) + 1,
        "logprobs": True,
        "top_logprobs": TOPK,
        "return_tokens_as_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "vllm_xargs": {"diffusion_seed_canvas": build_canvas(template, slots, seed), "diffusion_max_steps": schema["steps"], "diffusion_read_only": True},
    }
    d = upstream_chat(body)
    content = d["choices"][0]["logprobs"]["content"]
    out = []
    for q, s in zip(schema["questions"], slots):
        top = {int(t["token"].split(":")[1]): t["logprob"] for t in content[s["pos"]]["top_logprobs"]}
        floor = min(top.values()) - 5.0
        lp_t = [top.get(i, floor) for i in s["label_ids"]]
        # Read-only requests return temperature-1 logprobs, so the label
        # distribution is a softmax over the labels' logprobs as they are.
        mx = max(lp_t)
        ex = [math.exp(x - mx) for x in lp_t]
        probs = [e / sum(ex) for e in ex]
        top_p = [math.exp(v) for v in top.values()]
        out.append({
            "probs": probs,
            "label_mass": sum(math.exp(x) for x in lp_t),
            "entropy": -sum(p * math.log(p) for p in top_p if p > 0),
            "argmax_is_label": max(top, key=top.get) in s["label_ids"],
        })
    return out, d.get("usage", {})


def read_many(schema, template, slots, sys_text, state_text, seed, n):
    results = [None] * n
    errors = [None] * n

    def run(k):
        try:
            results[k], _ = one_read(schema, template, slots, sys_text, state_text, seed + k * 7919)
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


def decide(schema, state_text, seed):
    template, slots = template_for(schema)
    sys_text = system_text(schema)
    started = time.time()
    policy = schema["policy"]
    if policy["mode"] == "fixed":
        reads = read_many(schema, template, slots, sys_text, state_text, seed, policy["n"])
        extended = None
        first_entropy = None
    else:
        reads = read_many(schema, template, slots, sys_text, state_text, seed, 1)
        first_entropy = {q["id"]: r["entropy"] for q, r in zip(schema["questions"], reads[0])}
        extended = max(first_entropy.values()) > policy["threshold"] and policy["max"] > 1
        if extended:
            reads += read_many(schema, template, slots, sys_text, state_text, seed + 1, policy["max"] - 1)
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
            state = message_text(msgs[1]).strip()
            json.loads(state)
        except SchemaError as e:
            return self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
        except Exception as e:
            return self._json(400, {"error": {"message": f"system must be a JSON question schema and user must be JSON state: {e}", "type": "invalid_request_error"}})
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
    global ARGS, TOK, SCAFFOLD, CANVAS_LEN
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default="http://127.0.0.1:8010")
    p.add_argument("--model", default="dgemma")
    p.add_argument("--tokenizer", default="/models/dgemma", help="HF id or local path")
    p.add_argument("--canvas", type=int, default=64)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8011)
    ARGS = p.parse_args()
    CANVAS_LEN = ARGS.canvas
    TOK = AutoTokenizer.from_pretrained(ARGS.tokenizer)
    SCAFFOLD = enc("<|channel>thought\n<channel|>")
    print(f"structured server on {ARGS.host}:{ARGS.port} -> {ARGS.upstream} (canvas {CANVAS_LEN})", flush=True)
    ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
