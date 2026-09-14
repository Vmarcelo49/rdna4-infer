#!/usr/bin/env python3
"""Acceptance test for `rdna4-infer serve` (the OpenAI-compatible HTTP server).

Starts the server on a real model, drives it with the standard library only
(http.client, no external dependency — the `openai` SDK is not installed here and
must not be required), and asserts on the response shapes an OpenAI client
actually parses. Run it through scripts/check_server.sh so the GPU is serialised:

    scripts/check_server.sh                       # primary model, IQ3_S
    MODEL=.../Qwen3.8-27B-UD-IQ4_XS.gguf scripts/check_server.sh

What it covers
  * GET  /health                          -> 200 while the model is loaded
  * GET  /v1/models, /v1/models/{id}      -> object "list"/"model", 404 for another id
  * POST /v1/chat/completions             -> non-streaming body shape + usage counts
  * POST /v1/chat/completions (stream)    -> real SSE: headers, chunk objects,
                                             delta.content, [DONE], incremental
                                             arrival, stream_options.include_usage
  * the two chat paths must agree textually (temperature 0 + same seed)
  * POST /v1/completions                  -> non-streaming and streaming `text`
  * stop strings                          -> the stop text never reaches the client
  * 400 (bad JSON, bad role, no messages, context overflow), 404, 405
  * --greedy-cli-check: `run --chat --greedy` and the server at temperature 0 must
    produce the same bytes AND the same token ids (the server prints them with -v);
    the CLI runs after the server exits, so the 12 GiB model is never mapped twice.

Exit status 0 = all checks passed.
"""

import argparse
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import time

DEFAULT_MODEL = "/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf"


class Check:
    """Tiny assertion counter: every check prints ok/FAIL and the total decides."""

    def __init__(self):
        self.n = 0
        self.fails = 0

    def ok(self, cond, what, detail=""):
        self.n += 1
        if cond:
            print(f"  ok   {what}")
        else:
            self.fails += 1
            print(f"  FAIL {what}" + (f"\n       {detail}" if detail else ""))
        return bool(cond)

    def eq(self, got, want, what):
        return self.ok(got == want, what, f"got {got!r}, want {want!r}")


class Client:
    """One connection per request (the server answers with Connection: close)."""

    def __init__(self, host, port, timeout):
        self.host = host
        self.port = port
        self.timeout = timeout

    def _conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def request(self, method, path, body=None, headers=None):
        conn = self._conn()
        try:
            payload = None
            hdrs = dict(headers or {})
            if body is not None:
                payload = body if isinstance(body, (bytes, str)) else json.dumps(body)
                hdrs.setdefault("Content-Type", "application/json")
            conn.request(method, path, body=payload, headers=hdrs)
            resp = conn.getresponse()
            raw = resp.read()
            return resp.status, dict(resp.getheaders()), raw
        finally:
            conn.close()

    def request_stream(self, path, body):
        """Returns (status, headers, events) where each event is the raw `data:` payload."""
        conn = self._conn()
        conn.request("POST", path, body=json.dumps(body),
                         headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        headers = dict(resp.getheaders())
        events = []
        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                if not line.startswith(b"data: "):
                    continue
                events.append((time.time(), line[len(b"data: "):].strip()))
                if events[-1][1] == b"[DONE]":
                    break
        finally:
            conn.close()
        return resp.status, headers, events


def jbody(raw):
    return json.loads(raw.decode("utf-8"))


def log_tail(path, lines=25):
    try:
        with open(path, "r", errors="replace") as fh:
            return "".join(fh.readlines()[-lines:])
    except OSError:
        return "<no log>"


def wait_for_server(proc, client, logpath, timeout, check):
    """Polls /health; also watches for the server dying (a load error is not a timeout)."""
    deadline = time.time() + timeout
    listening = None
    while time.time() < deadline:
        if proc.poll() is not None:
            check.ok(False, "the server stayed up", f"exited with {proc.returncode}:\n{log_tail(logpath)}")
            return None
        text = ""
        try:
            with open(logpath, "r", errors="replace") as fh:
                text = fh.read()
        except OSError:
            pass
        m = re.search(r"listening on http://[^:]+:(\d+)", text)
        if m:
            listening = int(m.group(1))
        if listening is not None:
            try:
                status, _, raw = client.request("GET", "/health")
                if status == 200:
                    return listening
            except (OSError, http.client.HTTPException):
                pass
        time.sleep(0.5)
    check.ok(False, "the server became ready", f"no /health 200 within {timeout}s:\n{log_tail(logpath)}")
    return None


def check_models(check, cli, model_id):
    print("GET /v1/models")
    status, headers, raw = cli.request("GET", "/v1/models")
    check.eq(status, 200, "status 200")
    check.ok(headers.get("Content-Type", "").startswith("application/json"), "JSON content type",
             headers.get("Content-Type"))
    body = jbody(raw)
    check.eq(body.get("object"), "list", "object == list")
    check.ok(isinstance(body.get("data"), list) and len(body["data"]) == 1, "exactly one model",
             repr(body.get("data"))[:200])
    entry = body["data"][0]
    check.eq(entry.get("object"), "model", "entry object == model")
    check.eq(entry.get("id"), model_id, "entry id == the loaded model")
    check.ok(isinstance(entry.get("created"), int), "created is an integer")
    check.ok(isinstance(entry.get("owned_by"), str) and entry["owned_by"], "owned_by present")

    print("GET /v1/models/{id}")
    status, _, raw = cli.request("GET", f"/v1/models/{model_id}")
    check.eq(status, 200, "status 200 for the loaded id")
    check.eq(jbody(raw).get("id"), model_id, "id echoed")
    status, _, raw = cli.request("GET", "/v1/models/does-not-exist")
    check.eq(status, 404, "404 for an unknown id")
    err = jbody(raw).get("error") or {}
    check.eq(err.get("code"), "model_not_found", "error.code")

    print("GET /health")
    status, _, raw = cli.request("GET", "/health")
    check.eq(status, 200, "status 200")
    check.eq(jbody(raw).get("status"), "ok", "status == ok")


def check_chat_plain(check, cli, model_id, prompt, max_tokens, seed):
    print("POST /v1/chat/completions (stream=false)")
    req = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
    }
    status, headers, raw = cli.request("POST", "/v1/chat/completions", req)
    check.eq(status, 200, "status 200")
    check.ok(headers.get("Content-Type", "").startswith("application/json"), "JSON content type")
    body = jbody(raw)
    check.ok(body.get("id", "").startswith("chatcmpl-"), "id starts with chatcmpl-", body.get("id"))
    check.eq(body.get("object"), "chat.completion", "object")
    check.ok(isinstance(body.get("created"), int), "created is an integer")
    check.eq(body.get("model"), model_id, "model echoed")
    choices = body.get("choices")
    check.ok(isinstance(choices, list) and len(choices) == 1, "one choice")
    if not choices:
        return None, None
    c0 = choices[0]
    check.eq(c0.get("index"), 0, "choice index 0")
    check.eq((c0.get("message") or {}).get("role"), "assistant", "message.role")
    content = (c0.get("message") or {}).get("content")
    check.ok(isinstance(content, str), "message.content is a string")
    check.ok(content, "message.content is not empty", repr(content)[:120])
    check.ok(c0.get("finish_reason") in ("stop", "length"), "finish_reason is stop|length",
             c0.get("finish_reason"))
    usage = body.get("usage") or {}
    check.ok(isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] > 0,
             "usage.prompt_tokens > 0", usage)
    check.ok(isinstance(usage.get("completion_tokens"), int) and 0 < usage["completion_tokens"] <= max_tokens,
             f"usage.completion_tokens in 1..{max_tokens}", usage)
    check.eq(usage.get("total_tokens"),
             usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
             "usage.total == prompt + completion")
    return content, body


def check_chat_stream(check, cli, model_id, prompt, max_tokens, seed, expect_content=None):
    print("POST /v1/chat/completions (stream=true)")
    req = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    status, headers, events = cli.request_stream("/v1/chat/completions", req)
    check.eq(status, 200, "status 200")
    check.ok(headers.get("Content-Type", "").startswith("text/event-stream"),
             "Content-Type is text/event-stream", headers.get("Content-Type"))
    check.ok(headers.get("Transfer-Encoding", "").lower() == "chunked",
             "Transfer-Encoding: chunked", headers.get("Transfer-Encoding"))
    check.ok(len(events) >= 3, f"at least 3 SSE events (got {len(events)})")
    check.eq(events[-1][1], b"[DONE]", "last event is data: [DONE]")

    chunks = []
    for _, payload in events[:-1]:
        chunks.append(json.loads(payload.decode("utf-8")))
    check.ok(all(c.get("id", "").startswith("chatcmpl-") for c in chunks), "every chunk has an id")
    check.ok(all(c.get("object") == "chat.completion.chunk" for c in chunks), "object on every chunk")
    first = chunks[0]
    check.eq(first["choices"][0]["delta"].get("role"), "assistant", "first delta announces the role")

    content = "".join(c["choices"][0]["delta"].get("content") or ""
                      for c in chunks if c.get("choices"))
    finishing = [c for c in chunks if c.get("choices") and c["choices"][0].get("finish_reason")]
    check.eq(len(finishing), 1, "exactly one chunk carries finish_reason")
    if finishing:
        check.ok(finishing[0]["choices"][0]["finish_reason"] in ("stop", "length"),
                 "finish_reason is stop|length", finishing[0]["choices"][0]["finish_reason"])
    usage_chunks = [c for c in chunks if c.get("usage")]
    check.eq(len(usage_chunks), 1, "include_usage added one usage chunk")
    if usage_chunks:
        check.eq(usage_chunks[0].get("choices"), [], "the usage chunk has no choices")
        u = usage_chunks[0]["usage"]
        check.ok(u.get("completion_tokens", 0) > 0, "usage.completion_tokens > 0", u)
        check.eq(u.get("total_tokens"), u.get("prompt_tokens", 0) + u.get("completion_tokens", 0),
                 "usage.total == prompt + completion")
    if expect_content is not None:
        check.eq(content, expect_content, "streamed deltas == the non-streaming content")

    # Incremental delivery: the first event must arrive before the last one, i.e.
    # the server is not buffering the whole generation.
    check.ok(events[0][0] < events[-1][0], "SSE events arrive incrementally")
    print(f"       {len(events)} events, {len(content)} chars, "
          f"first at +{events[0][0] - t0:.2f}s, last at +{events[-1][0] - t0:.2f}s")
    return content


def check_completions(check, cli, model_id, max_tokens, prompt="The capital of France is"):
    print("POST /v1/completions (stream=false)")
    req = {"model": model_id, "prompt": prompt, "temperature": 0, "max_tokens": max_tokens}
    status, _, raw = cli.request("POST", "/v1/completions", req)
    check.eq(status, 200, "status 200")
    body = jbody(raw)
    check.ok(body.get("id", "").startswith("cmpl-"), "id starts with cmpl-", body.get("id"))
    check.eq(body.get("object"), "text_completion", "object")
    text = body["choices"][0].get("text")
    check.ok(isinstance(text, str) and text, "choices[0].text is a non-empty string", repr(text)[:120])
    check.ok("usage" in body and body["usage"]["completion_tokens"] > 0, "usage present", body.get("usage"))

    print("POST /v1/completions (stream=true)")
    status, headers, events = cli.request_stream("/v1/completions", dict(req, stream=True))
    check.eq(status, 200, "status 200")
    check.ok(headers.get("Content-Type", "").startswith("text/event-stream"), "SSE content type")
    check.eq(events[-1][1], b"[DONE]", "ends with [DONE]")
    streamed = "".join(json.loads(p.decode("utf-8"))["choices"][0].get("text") or ""
                       for _, p in events[:-1] if json.loads(p.decode("utf-8")).get("choices"))
    check.eq(streamed, text, "streamed text == non-streaming text")


def check_stop(check, cli, model_id, max_tokens, no_stop_content,
               stop="Paris", prompt="The capital of France is"):
    """The stop text must never reach the client and finish_reason must say so.

    `no_stop_content` is the same greedy generation without `stop`: the truncated
    answer has to be exactly its prefix (not one byte more, not one byte less —
    with the stop sequence spanning two tokens this is where a naive emitter
    leaks the first half of it).
    """
    print("POST /v1/chat/completions with a stop string")
    req = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stop": [stop, "ZZZ-never"],
    }
    status, _, raw = cli.request("POST", "/v1/chat/completions", req)
    check.eq(status, 200, "status 200")
    body = jbody(raw)
    content = body["choices"][0]["message"]["content"]
    check.ok(stop not in content, f"the stop string {stop!r} is not in the content",
             repr(content[-60:]))
    check.eq(body["choices"][0]["finish_reason"], "stop", "finish_reason == stop")
    check.ok(no_stop_content and no_stop_content.startswith(content),
             "the answer is a prefix of the same generation without `stop`",
             f"stopped {content!r}\n       full    {no_stop_content!r}")
    check.eq(content, (no_stop_content or "").split(stop)[0],
             "truncated exactly at the stop string")
    check.ok(body["usage"]["completion_tokens"] < max_tokens,
             "the stop ended the generation before max_tokens", body["usage"])

    print("POST /v1/chat/completions with a stop string (stream=true)")
    status, headers, events = cli.request_stream("/v1/chat/completions", dict(req, stream=True))
    check.eq(status, 200, "status 200")
    chunks = [json.loads(p.decode("utf-8")) for _, p in events[:-1]]
    streamed = "".join(c["choices"][0]["delta"].get("content") or ""
                       for c in chunks if c.get("choices"))
    check.eq(streamed, content, "streamed deltas == the truncated content")
    fin = [c for c in chunks if c.get("choices") and c["choices"][0].get("finish_reason")]
    check.eq(len(fin), 1, "one finish chunk")
    if fin:
        check.eq(fin[0]["choices"][0]["finish_reason"], "stop", "streamed finish_reason == stop")


def check_errors(check, cli, model_id, ctx_size):
    print("error paths")
    # 400: not JSON at all
    status, _, raw = cli.request("POST", "/v1/chat/completions", "this is not json")
    check.eq(status, 400, "400 for a non-JSON body")
    err = jbody(raw).get("error") or {}
    check.ok(isinstance(err.get("message"), str) and err["message"], "error.message present")
    check.eq(err.get("type"), "invalid_request_error", "error.type")
    check.eq(err.get("code"), "invalid_json", "error.code")
    check.ok("param" in err, "error.param key present (null is fine)")

    # 400: missing messages
    status, _, raw = cli.request("POST", "/v1/chat/completions", {"model": model_id})
    check.eq(status, 400, "400 without messages")
    check.eq((jbody(raw).get("error") or {}).get("code"), "invalid_messages", "error.code")

    # 400: unknown role
    status, _, raw = cli.request("POST", "/v1/chat/completions", {
        "model": model_id, "messages": [{"role": "wizard", "content": "hi"}]})
    check.eq(status, 400, "400 for an unknown role")
    check.eq((jbody(raw).get("error") or {}).get("code"), "invalid_role", "error.code")

    # 400: n > 1 (the engine generates one sequence at a time)
    status, _, raw = cli.request("POST", "/v1/chat/completions", {
        "model": model_id, "messages": [{"role": "user", "content": "hi"}], "n": 3})
    check.eq(status, 400, "400 for n > 1")
    check.eq((jbody(raw).get("error") or {}).get("code"), "unsupported_n", "error.code")

    # 400: non-finite and out-of-range sampling parameters (audit finding H1).
    # 1e400 is a legal JSON number that strtod turns into +inf, and inf used to be
    # accepted: `temperature: inf` makes every logit logit/inf = 0 (uniform
    # softmax), and `seed: inf` cast to uint64_t is undefined behaviour. The bodies
    # are raw strings because json.dumps(float('inf')) would emit the invalid
    # literal Infinity instead of the 1e400 the test is about.
    for field, value, code in (
            ("seed", "1e400", "invalid_seed"),
            ("seed", "-1e400", "invalid_seed"),
            ("seed", "1e16", "invalid_seed"),
            ("temperature", "1e400", "invalid_temperature"),
            ("temperature", "-1", "invalid_temperature"),
            ("temperature", "1e9", "invalid_temperature"),
            ("top_p", "1e400", "invalid_top_p"),
            ("min_p", "1e400", "invalid_min_p"),
            ("repeat_penalty", "1e400", "invalid_repeat_penalty"),
            ("repeat_penalty", "0", "invalid_repeat_penalty"),
            ("presence_penalty", "1e400", "invalid_presence_penalty"),
            ("frequency_penalty", "-1e400", "invalid_frequency_penalty")):
        body = ('{"model": "%s", "messages": [{"role": "user", "content": "hi"}], '
                '"%s": %s, "max_tokens": 1}' % (model_id, field, value))
        status, _, raw = cli.request("POST", "/v1/chat/completions", body)
        check.eq(status, 400, f"400 for {field}={value}")
        check.eq((jbody(raw).get("error") or {}).get("code"), code,
                 f"error.code for {field}={value}")

    # 400: a prompt that cannot fit the context
    big = "word " * (ctx_size + 64)
    status, _, raw = cli.request("POST", "/v1/chat/completions", {
        "model": model_id, "messages": [{"role": "user", "content": big}], "max_tokens": 4})
    check.eq(status, 400, f"400 for a prompt over the {ctx_size}-token context")
    err = jbody(raw).get("error") or {}
    check.eq(err.get("code"), "context_length_exceeded", "error.code")
    check.ok("maximum context" in err.get("message", "") or "context is" in err.get("message", ""),
             "the message explains the limit", err.get("message"))

    # 404: unknown route
    status, _, raw = cli.request("GET", "/v1/embeddings")
    check.eq(status, 404, "404 for an unknown route")
    err = jbody(raw).get("error") or {}
    check.eq(err.get("type"), "not_found_error", "error.type")
    check.eq(err.get("code"), "unknown_url", "error.code")

    # 405: known route, wrong method
    status, _, raw = cli.request("POST", "/v1/models", {"x": 1})
    check.eq(status, 405, "405 for POST /v1/models")
    check.ok("not allowed" in (jbody(raw).get("error") or {}).get("message", ""),
             "the message names the method")

    # the server is still healthy after all of that
    status, _, _ = cli.request("GET", "/health")
    check.eq(status, 200, "still healthy after the error paths")


def check_unsupported_but_tolerated(check, cli, model_id, max_tokens):
    """Harnesses send fields this server ignores; a 200 (with a logged note) is the contract."""
    print("unknown/ignored request fields")
    req = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Say \"hi\"."}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "top_k": 20,
        "logit_bias": {"1": -100},
        "user": "check-server",
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    }
    status, _, raw = cli.request("POST", "/v1/chat/completions", req)
    check.eq(status, 200, "200 with tools/logit_bias/user present (ignored, not rejected)")
    check.ok(jbody(raw)["choices"][0]["message"]["content"], "content is non-empty")

    print("multi-turn + roles (system/developer/assistant/tool)")
    req = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "tool", "content": "{\"ok\":true}"},
            {"role": "user", "content": "Answer with one word."},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    status, _, raw = cli.request("POST", "/v1/chat/completions", req)
    check.eq(status, 200, "200 for a multi-turn conversation with thinking disabled")
    body = jbody(raw)
    check.ok(body["usage"]["prompt_tokens"] > 10, "the whole conversation was rendered",
             body["usage"])

    # thinking disabled must not emit the thinking block; the raw text is in
    # `content` either way (no reasoning split, see docs/servidor-openai.md)
    check.ok("</think>" not in body["choices"][0]["message"]["content"],
             "no thinking block when enable_thinking=false",
             repr(body["choices"][0]["message"]["content"])[:120])


def run_cli_greedy(binary, model, ctx_size, prompt, n, log):
    """`run --chat --greedy` with the same effective prompt as the server's chat call."""
    cmd = [binary, "run", "-m", model, "--chat", "-p", prompt, "-n", str(n), "--greedy",
           "--seed", "0", "--ctx-size", str(ctx_size), "-v"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with open(log, "wb") as fh:
        fh.write(b"$ " + " ".join(cmd).encode() + b"\n")
        fh.write(proc.stderr)
    if proc.returncode != 0:
        return None, None
    return proc.stdout, proc.stderr.decode("utf-8", "replace")


def check_greedy_equivalence(check, binary, model, ctx_size, prompt, n, server_log, server_content,
                             server_ids, tmpdir):
    """The acceptance check: identical bytes and identical token ids, CLI vs server."""
    print("greedy equivalence: `run --chat --greedy` vs the server at temperature 0")
    log = os.path.join(tmpdir, "cli-greedy.log")
    out, err = run_cli_greedy(binary, model, ctx_size, prompt, n, log)
    if out is None:
        check.ok(False, "the CLI run succeeded", f"see {log}")
        return
    m = re.search(r"^generated ids:(.*)$", err, re.M)
    cli_ids = [int(x) for x in m.group(1).split()] if m else []
    check.ok(bool(cli_ids), "the CLI reported generated ids", f"see {log}")

    server_bytes = (server_content or "").encode("utf-8")
    check.ok(cli_bytes_match := (out == server_bytes), "text: CLI stdout == server content",
             f"CLI {out[:80]!r}...\n       server {server_bytes[:80]!r}...")
    check.ok(cli_ids == server_ids, "token ids: CLI == server",
             f"CLI {cli_ids[:12]}...\n       server {server_ids[:12]}...")
    print(f"       prompt {prompt!r}, {n} max tokens")
    print(f"       CLI    ids ({len(cli_ids)}): {cli_ids}")
    print(f"       server ids ({len(server_ids)}): {server_ids}")
    print(f"       text ({len(out)} bytes): {out!r}")
    print(f"       CLI run log: {log}")


def parse_server_ids(logpath):
    """The server prints `generated ids: ...` with -v, in cmd_run's exact format."""
    try:
        with open(logpath, "r", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return []
    ids = []
    for m in re.finditer(r"^generated ids:(.*)$", text, re.M):
        ids.append([int(x) for x in m.group(1).split()])
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--binary", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "rdna4-infer"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--ctx-size", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=24, help="tokens per request (keeps the test short)")
    ap.add_argument("--startup-timeout", type=float, default=600.0)
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--log", default="/tmp/rdna4-serve-test.log")
    ap.add_argument("--tmpdir", default="/tmp/rdna4-serve-test")
    ap.add_argument("--greedy-cli-check", action="store_true",
                    help="after the server exits, compare with `run --chat --greedy` (extra model load)")
    ap.add_argument("--keep-server", action="store_true", help="leave the server running (debugging)")
    args = ap.parse_args()

    os.makedirs(args.tmpdir, exist_ok=True)
    # Unbuffered progress: a run is dominated by two model loads, and watching
    # the checks appear one by one is how a failure gets localised.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    check = Check()
    if not os.path.exists(args.model):
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2
    if not os.path.exists(args.binary):
        print(f"binary not found: {args.binary} (build first)", file=sys.stderr)
        return 2

    cmd = [args.binary, "serve", "-m", args.model, "--host", args.host, "--port", str(args.port),
           "--ctx-size", str(args.ctx_size), "--n-predict-default", str(args.max_tokens), "-v"]
    print("+ " + " ".join(cmd))
    print(f"  (log: {args.log})")
    logf = open(args.log, "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    cli = Client(args.host, args.port, args.request_timeout)
    try:
        port = wait_for_server(proc, cli, args.log, args.startup_timeout, check)
        if port is None:
            return 1
        if port != args.port:
            print(f"note: the server bound port {port}")
        cli = Client(args.host, port, args.request_timeout)

        with open(args.log, "r", errors="replace") as fh:
            first = fh.readlines()[:3]
        for line in first:
            print("  | " + line.rstrip())
        model_id = None
        for line in first:
            m = re.search(r"model ([^,]+), ctx", line)
            if m:
                model_id = m.group(1)
        check.ok(model_id is not None, "the server announced its model id", "".join(first))

        prompt = "The capital of France is"
        print()
        check_models(check, cli, model_id)
        print()
        content, body = check_chat_plain(check, cli, model_id, prompt, args.max_tokens, 0)
        print()
        streamed = check_chat_stream(check, cli, model_id, prompt, args.max_tokens, 0,
                                     expect_content=content)
        print()
        check_completions(check, cli, model_id, args.max_tokens)
        print()
        check_stop(check, cli, model_id, args.max_tokens, content)
        print()
        check_unsupported_but_tolerated(check, cli, model_id, args.max_tokens)
        print()
        check_errors(check, cli, model_id, args.ctx_size)
    finally:
        if args.keep_server:
            print(f"\nserver left running (pid {proc.pid}) on port {args.port}; log {args.log}")
        else:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            logf.close()
            print(f"\nserver stopped (exit {proc.returncode}); log: {args.log}")

    if args.greedy_cli_check:
        print()
        ids = parse_server_ids(args.log)
        # the last chat request of the run above used temperature 0 with max_tokens
        server_ids = ids[0] if ids else []
        prompt = "The capital of France is"
        check.ok(bool(server_ids), "the server logged generated ids (-v)", args.log)
        check_greedy_equivalence(check, args.binary, args.model, args.ctx_size, prompt,
                                 args.max_tokens, args.log, content, server_ids, args.tmpdir)

    print(f"\ncheck-server: {check.n} checks, {check.fails} failure(s)")
    return 1 if check.fails else 0


if __name__ == "__main__":
    sys.exit(main())
