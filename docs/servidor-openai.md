# rdna4-infer — the OpenAI-compatible server (`serve`)

`rdna4-infer serve` puts the engine behind the subset of the OpenAI HTTP API that
real harnesses speak (Open WebUI, Continue, aider, LangChain, the `openai`
python/js SDKs). It is the same model, the same graph and the same sampler as
`run`; only the transport and the request/response translation are new.

Everything below was measured on the reference machine with the primary file
(`Qwen3.8-27B-UD-IQ3_S.gguf`, 11.2 GiB) unless a line says otherwise.

## Usage

```bash
# under the GPU lock, like every other command that maps the model
scripts/gpu-lock.sh ./build/rdna4-infer serve -m <model.gguf> --port 8080

# then, from any OpenAI client
curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen3.8-27B-UD-IQ3_S",
  "messages":[{"role":"user","content":"Say hi in five words."}],
  "temperature":0,"max_tokens":32}'
```

Flags: `-m PATH` (required), `--host` (default `127.0.0.1`; `0.0.0.0` exposes it
to the LAN), `--port` (default `8080`, `0` picks a free one), `--ctx-size`
(default `8192`; `run`'s default is 4096, a server wants room for longer
conversations), `--cache-type-k/-v` (`f32|f16|q8_0|q4_0`), `--n-predict-default`
(`max_tokens` when a request omits it, default `512`; `-1` = until the context is
full), `--model-name` (the id `/v1/models` reports; default = the file name
without `.gguf`), `-v/--verbose` (per-request log with the prompt ids, the
generated ids and the timings).

Exit codes follow the CLI: `0` ok, `1` usage/IO/model error (including a busy
port), `2` no `gfx1201` device, `3` insufficient VRAM.

## Endpoints

| method | path | notes |
|---|---|---|
| GET | `/health` | `{"status":"ok","model":...,"ctx_size":N}` — what harnesses probe |
| GET | `/v1/models` | `{"object":"list","data":[{"id","object":"model","created","owned_by"}]}` |
| GET | `/v1/models/{id}` | 404 `model_not_found` for anything but the loaded id |
| POST | `/v1/chat/completions` | messages → assistant message; SSE when `stream: true` |
| POST | `/v1/completions` | legacy prompt → text, same streaming shape with `text` deltas |
| OPTIONS | any known path | 204 + `Allow` + CORS (browser-based UIs). Every response also carries `Access-Control-Allow-Origin: *` |
| * | anything else | 404 `unknown_url`; known path with the wrong method → 405 + `Allow` |

Every error body is `{"error":{"message","type","param","code"}}`, the shape the
OpenAI SDKs surface to the user.

## Request fields

Honoured: `model` (a mismatch only logs a note — one model is loaded),
`messages[{role,content}]` with `role ∈ {system, developer, user, assistant,
tool}` (`content` may also be the vision-style array of parts: the text parts are
concatenated, images are ignored), `stream`, `stream_options.include_usage`,
`max_tokens` / `max_completion_tokens` / `n_predict`, `temperature`, `top_p`,
`top_k`, `min_p`, `seed`, `stop` (string or array, ≤ 16), `presence_penalty`,
`frequency_penalty`, `repeat_penalty` (llama.cpp's name; `repetition_penalty` is
accepted as an alias), `repeat_last_n`, `ignore_eos`, `reasoning_effort`
(`xhigh|high|medium|low`, `none` = thinking off) and `chat_template_kwargs`
(`enable_thinking`, `reasoning_effort`) for the qwen35 template.

Sampling defaults come from the GGUF's `general.sampling.*` metadata exactly as in
`run` (this model ships `temp 1.0`, `top_k 20`, `top_p 0.95`), so an omitted
`temperature` samples rather than silently becoming greedy. A field the request
does send always wins.

Accepted and ignored (logged with `-v`): `tools`/`functions` (no tool-call
rendering), `logit_bias`, `response_format`, `user`, `logprobs`, `echo`, `suffix`.

Rejected with 400 (rather than silently ignored): `n > 1`, `best_of > 1`
(the graph generates one sequence at a time), a bad role, `messages` missing or
empty, `max_tokens` out of range, sampling values out of range, a body that is not
a JSON object, and a prompt that does not fit the context.

## Streaming

`stream: true` returns a real SSE stream: `Content-Type: text/event-stream`,
`Transfer-Encoding: chunked` and `Connection: close`. Each event is
`data: {json}\n\n`, the stream ends with `data: [DONE]\n\n`, and
`stream_options.include_usage` adds a final chunk with `"choices": []` and the
`usage` object before `[DONE]`. Chunks are written straight from the generation
loop (a token's text leaves the process as soon as it is decoded), which is what
makes the first `delta.content` arrive long before the last one.

Deltas are emitted through `rdna4::utf8_keep_len`, so a multi-byte character split
across two tokens is never sent half-formed. With `stop` sequences the emitter
also holds back the last `max(len(stop)) - 1` bytes so a stop string spanning two
tokens is never emitted (nothing has to be retracted).

## Clients

Any OpenAI client: base URL `http://127.0.0.1:8080/v1`, any API key (the server
has **no authentication** — it is a local process, which is why it binds
`127.0.0.1` by default; `--host 0.0.0.0` exposes an unauthenticated endpoint to
the network, so only do that behind something that authenticates), and the model
id that `GET /v1/models` reports (or `--model-name` to choose one).

```python
from openai import OpenAI           # works with the same base URL from any SDK
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
print(client.chat.completions.create(
    model="Qwen3.8-27B-UD-IQ3_S",
    messages=[{"role": "user", "content": "Say hi in five words."}],
    temperature=0, max_tokens=32).choices[0].message.content)
```

Two settings worth raising in a harness because of this engine's known limits:

* **Prefill is not batched** (README "Known limitations"): a prompt costs
  `n_tokens × ~35 ms`, so a 4096-token conversation needs ~2.5 minutes before the
  first token. Set the client's read timeout generously (the `openai` python SDK:
  `timeout=900`) or keep the conversations short.
* **One request at a time**: a second request waits for the first to finish (it
  sits in the listen backlog). Parallel harnesses will see queueing, not errors.

## Deliberate scope limits

* **One request at a time, single-threaded.** The graph owns the only KV cache and
  the only recurrent (GDN) state; a second concurrent request would corrupt the
  first. Connections are accepted, served and closed one by one — a request that
  arrives while another is generating waits in the backlog.
* **`Connection: close` after every response** (no keep-alive, no pipelining). Every
  OpenAI client reconnects or tolerates the close, and SSE is delimited by the
  terminating `0\r\n\r\n` chunk.
* **No prefix/KV reuse between requests** (`--kv-reuse` from the task list is *not*
  implemented). Reusing the KV cache of a common prompt prefix requires rolling
  back the recurrent state too, which is only possible with llama.cpp's state
  checkpointing — out of scope for v1. Every request calls `Graph::reset_state()`.
* **No `reasoning_content` split.** The generated text (including the
  `<think>` block the template opens) is returned in `content`; clients that render
  reasoning separately will show it as part of the answer.
* **No tool calls, no logprobs, no logit_bias, no multi-choice, no embeddings.**
* **`run` and the server differ by ≤3 bytes in one rare case**: when generation
  ends in the middle of a multi-byte UTF-8 character, `run` writes those raw bytes
  to stdout while the server drops them (a JSON body must be valid UTF-8) and logs
  how many. The token ids are identical either way.

## Context overflow: 400, not truncation

A prompt that cannot fit is answered with

```
400 {"error":{"message":"the prompt is 4213 tokens but this model's context is 4096 tokens:
     shorten the conversation or restart the server with a larger --ctx-size",
     "code":"context_length_exceeded","type":"invalid_request_error"}}
```

Rationale: harnesses handle this code by trimming their history, and silently
truncating a conversation makes the model answer a question the user did not ask.
When only the *generated* side does not fit, `max_tokens` is clipped to the room
left and the clip is logged (`[serve] warning: max_tokens 512 clipped to 130 ...`)
— the same policy as `cmd_run`'s `-n`. Nothing ever overflows the cache: the
generation loop stops at the last position of the context.

## Files and how they were split

| file | what | compiled by |
|---|---|---|
| `include/rdna4/server.h`, `server_json.h` | HTTP layer + JSON declarations | — |
| `src/server/http.cpp` | blocking sockets, request parsing, router, error bodies, chunked writer | CXX (no HIP) |
| `src/server/json.cpp` | JSON parser/serializer | CXX (no HIP) |
| `src/server/serve.hip` | `cmd_serve`: model, graph, OpenAI handlers, generation loop | HIP (`--offload-arch=gfx1201`) |
| `include/rdna4/cli_commands.h` | `int cmd_serve(int, char**)` — the only thing `src/main.hip` needs | — |
| `tests/check_server_http.cpp` | CPU checks: JSON, router, sockets, SSE framing, timeouts | CXX |
| `tests/check_server.py` | the real acceptance test (model, SSE, error paths, greedy equivalence) | python3, stdlib only |
| `scripts/check_server.sh` | runs the above under `scripts/gpu-lock.sh` | bash |

`src/main.hip` gained exactly two lines (the `#include` of `cli_commands.h` and the
`serve` dispatch before the final `cmd_info`), and `CMakeLists.txt` only has
appended blocks.

### Why the server is a shared library

The kernel headers (`nn.cuh`, `matvec.cuh`, `attn.cuh`, `gdn.cuh`,
`dequant_row.cuh`, `graph.cuh`) define their `__global__` functions **without
`static`/`inline`** — they are header-only kernels meant to be included by exactly
one translation unit per binary. Adding `serve.hip` as a second object of
`rdna4-infer` therefore fails at link time:

```
ld.lld: error: duplicate symbol: rdna4::mul_kernel(float const*, float const*, float*, long)
>>> defined at main.hip
>>> defined at serve.hip
```

Two ways out: mark all 21 kernels `static`/`inline` (a change spread over six
headers owned by the ROCm workstream — a merge conflict for nothing), or keep the
server in its own module. It is built as `librdna4_serve.so` and linked by
`rdna4-infer`; the executable's copies of the kernels win the symbol lookup, the
device code is the same code compiled from the same headers, and the CLI subcommand
is a normal function call into the library. `serve --help` and the whole acceptance
test pass in this layout (see the report in the branch's commits) — and it is
exactly one appended `add_library` + one appended `target_link_libraries`.

## Verification

```bash
cmake --build build -j6 --target rdna4-infer check-server-http
./build/check-server-http          # 68 CPU checks, no GPU
./scripts/check_server.sh          # the real test, under the GPU lock
```

`scripts/check_server.sh` starts the server itself, waits for `/health`, runs the
checks listed in `tests/check_server.py`'s docstring, stops the server, and — with
`--greedy-cli-check` (the default) — then runs `rdna4-infer run --chat --greedy -v`
with the same prompt and compares **both** the generated bytes and the generated
token ids against the server's `temperature: 0` answer. The two model loads are
sequential, so the 12 GiB mapping is never duplicated.
