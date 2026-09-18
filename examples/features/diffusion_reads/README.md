# Structured reads on DiffusionGemma

A discrete diffusion model denoises a whole canvas per forward pass. If the
canvas is seeded with the answer's fixed text and only the answer slots are
left as noise, one denoise step yields a distribution over each slot. Three `extra_args` fields (`vllm_xargs` on the OpenAI server) expose
that:

| field | type | meaning |
|---|---|---|
| `diffusion_seed_canvas` | `list[int]`, exactly `canvas_length` ids | replaces the random initial canvas after prefill |
| `diffusion_max_steps` | `int` | denoise steps before the canvas is emitted |
| `diffusion_read_only` | `bool` | emit the argmax canvas as soon as the cap is reached, end the request there, and return temperature-1 logprobs at every position |

`structured_server.py` is the layer that turns a question schema into those
fields. It speaks `/v1/chat/completions`: the system message is the schema,
the user message is the state JSON, and the reply content is one
distribution per question with a standard error over a few noise draws.

```bash
vllm serve google/diffusiongemma-26B-A4B-it \
    --diffusion-config '{"canvas_length": 64}' --max-logprobs 32 --enable-prefix-caching
python examples/features/diffusion_reads/structured_server.py \
    --upstream http://127.0.0.1:8000 --tokenizer google/diffusiongemma-26B-A4B-it --canvas 64
curl -s localhost:8011/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [
    {"role": "system", "content": "{\"questions\": [{\"id\": \"urgent\", \"type\": \"noul\", \"instructions\": \"Does the customer need a reply within the hour?\"}]}"},
    {"role": "user", "content": "{\"ticket\": \"Everything is down and we have a demo at noon.\"}"}
  ]}'
```

Question types: `noul` (yes/no), `choice` with `options`, `score` with
ordered `levels`. Each label must be a single token in the answer template,
which the server checks with the tokenizer before the first request.

The server also speaks the contract of the Jev decision API at
`POST /v1/systemone`: a body of `state`, `questions` (a map of id to
`type`, `instructions`, `criteria`) and `model`, with answers in that API's
shapes (`noul` probability; `choice` with `probabilities` and `confidence`;
`score` with a 0-indexed `legend`). The schema's options above go in the
same body as extensions. A question may declare `depends_on` (read in a later stage with those
answers in its prompt), `ask_if` (asked only when a named question's answer
is among the listed ones, else null) and `alone` (a read of its own). Images attach as `multipart/form-data` (the JSON
in a part named `request`, each image a file part) or as an `images` array
of data URLs. With `TEST_PAGE=1` in the environment, `GET /` serves
`playground.html`, a page for sending requests with an image file or
webcam frames.

```bash
curl -s localhost:8011/v1/systemone -H 'content-type: application/json' -d '{
  "model": "jev-latest",
  "state": {"ticket": "Everything is down and we have a demo at noon."},
  "questions": {"urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"}}}'
```

`"think": N` in the schema lets the model write up to N tokens in its
thought channel before the read. The thought is an ordinary generation with
the chat template's thinking marker on, and the read then runs with the
thought in its prompt, so the answer slots condition on it. One thought
serves every noise draw of a decision. `diagnostics.thought` returns the
text, its length in tokens, whether the model closed the channel itself and
the generation time.
