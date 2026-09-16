// CPU-only checks for the server's non-GPU half: the JSON codec, the router
// (404/405/OPTIONS shapes) and the socket layer (request parsing, error status
// codes, chunked streaming framing, per-connection timeouts).
//
// Why it exists: tests/check_server.py exercises the real thing (the real model,
// real tokens, the real generation loop) but needs the GPU and ~20 s just to
// load the model. This binary is the fast loop for the parts of the server that
// have nothing to do with HIP — it links src/server/http.cpp and
// src/server/json.cpp only, never a kernel.
//
//   ./build/check-server-http
#include <arpa/inet.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "rdna4/server.h"
#include "rdna4/server_json.h"

using rdna4::server::Headers;
using rdna4::server::Request;
using rdna4::server::Responder;
using rdna4::server::Router;
using rdna4::server::ServerOptions;
using rdna4::server::json::Value;

namespace {

int g_fails = 0;
int g_checks = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
      ++g_fails;                                                             \
    }                                                                        \
  } while (0)

#define CHECK_EQ(a, b)                                                            \
  do {                                                                            \
    ++g_checks;                                                                   \
    const std::string va = (a), vb = (b);                                         \
    if (va != vb) {                                                               \
      std::printf("FAIL %s:%d: %s != %s\n  left:  %s\n  right: %s\n", __FILE__,   \
                  __LINE__, #a, #b, va.c_str(), vb.c_str());                      \
      ++g_fails;                                                                  \
    }                                                                             \
  } while (0)

bool contains(const std::string &hay, const std::string &needle) {
  return hay.find(needle) != std::string::npos;
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------
void check_json() {
  Value v;
  std::string err;

  // A realistic chat body, nested access and integral dumping.
  const std::string body =
      R"({"model":"m","messages":[{"role":"user","content":"hi"},)"
      R"({"role":"assistant","content":null}],"stream":true,"temperature":0,)"
      R"("max_tokens":32,"stop":["A","B"]})";
  CHECK(rdna4::server::json::parse(body, v, err));
  CHECK(v.is_object());
  CHECK(v.has("messages"));
  std::string model;
  CHECK(v.get_string("model", &model) && model == "m");
  const Value *msgs = v.member("messages");
  CHECK(msgs != nullptr && msgs->is_array() && msgs->size() == 2);
  std::string role;
  CHECK(msgs->at(0)->get_string("role", &role) && role == "user");
  CHECK(msgs->at(1)->member("content")->is_null());
  bool stream = false;
  CHECK(v.get_bool("stream", &stream) && stream);
  long long mt = 0;
  CHECK(v.get_int("max_tokens", &mt) && mt == 32);
  double temp = -1.0;
  CHECK(v.get_number("temperature", &temp) && temp == 0.0);
  const Value *stop = v.member("stop");
  CHECK(stop != nullptr && stop->is_array() && stop->size() == 2);
  // integral numbers must not be dumped as "32.0"
  CHECK(contains(v.dump(), "\"max_tokens\":32"));
  CHECK(!contains(v.dump(), "32.0"));

  // Escapes: \u (BMP + surrogate pair), \n, \", and UTF-8 passthrough.
  Value e;
  CHECK(rdna4::server::json::parse(R"({"s":"a\nb\"c\u00e9\u4e2d\ud83d\ude00"})", e, err));
  const std::string s = e.member("s")->as_string();
  CHECK_EQ(s, std::string("a\nb\"c\xc3\xa9\xe4\xb8\xad\xf0\x9f\x98\x80"));
  // and it survives a round trip
  Value back;
  CHECK(rdna4::server::json::parse(e.dump(), back, err));
  CHECK_EQ(back.member("s")->as_string(), s);

  // A raw control byte inside a string is accepted (lenient by design).
  CHECK(rdna4::server::json::parse("{\"s\":\"a\nb\"}", e, err));
  CHECK_EQ(e.member("s")->as_string(), std::string("a\nb"));

  // Errors: trailing garbage, unterminated, bad escape, deep nesting.
  CHECK(!rdna4::server::json::parse("{} x", e, err));
  CHECK(contains(err, "trailing characters"));
  CHECK(!rdna4::server::json::parse("{\"a\":1", e, err));
  CHECK(!rdna4::server::json::parse("\"\\q\"", e, err));
  CHECK(contains(err, "invalid escape"));
  std::string deep = "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[";
  CHECK(!rdna4::server::json::parse(deep, e, err));
  CHECK(contains(err, "nesting too deep"));
  CHECK(!rdna4::server::json::parse("", e, err));

  // Builders used by the responders.
  Value root = Value::object();
  root.set("id", Value::string("chatcmpl-1"));
  root.set("created", Value::integer(1700000000));
  root.set("logprobs", Value::null());
  Value arr = Value::array();
  arr.push(Value::number(0.5));
  root.set("data", std::move(arr));
  CHECK(contains(root.dump(), "\"logprobs\":null"));
  CHECK(contains(root.dump(), "\"created\":1700000000"));
  CHECK(contains(root.dump(), "\"data\":[0.5]"));
  Value nan = Value::number(std::nan(""));
  CHECK_EQ(nan.dump(), std::string("null"));

  std::printf("json: %d checks so far\n", g_checks);
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------
struct Recorder : public Responder {
  int status = 0;
  std::string ctype, body, chunks;
  Headers hdrs;
  bool streamed = false;

  bool send(int s, const std::string &ct, const std::string &b, const Headers &extra) override {
    status = s;
    ctype = ct;
    body = b;
    hdrs = extra;
    return true;
  }
  bool begin_stream(int s, const std::string &ct, const Headers &extra) override {
    status = s;
    ctype = ct;
    hdrs = extra;
    streamed = true;
    return true;
  }
  bool stream_chunk(const std::string &c) override {
    chunks += c;
    return true;
  }
  bool end_stream() override { return true; }
  bool client_gone() const override { return false; }
};

Request make_request(const char *method, const char *path) {
  Request r;
  r.method = method;
  r.path = path;
  return r;
}

void check_router() {
  Router router;
  router.add("GET", "/v1/models", [](const Request &, Responder &out) {
    out.send(200, "application/json", "{\"object\":\"list\"}", Headers{});
  });
  router.add("POST", "/v1/chat/completions", [](const Request &q, Responder &out) {
    out.send(200, "application/json", "{\"echo\":" + rdna4::server::json::quote(q.body) + "}",
             Headers{});
  });
  router.add_prefix("GET", "/v1/models/", [](const Request &q, Responder &out) {
    out.send(200, "application/json", "{\"id\":" + rdna4::server::json::quote(q.path) + "}",
             Headers{});
  });

  Recorder r;
  router.dispatch(make_request("GET", "/v1/models"), r);
  CHECK(r.status == 200);
  CHECK_EQ(r.body, std::string("{\"object\":\"list\"}"));

  // prefix route
  Recorder r2;
  router.dispatch(make_request("GET", "/v1/models/some-id"), r2);
  CHECK(r2.status == 200);
  CHECK_EQ(r2.body, std::string("{\"id\":\"/v1/models/some-id\"}"));

  // unknown route -> 404 with the OpenAI error shape
  Recorder r3;
  router.dispatch(make_request("GET", "/v1/nope"), r3);
  CHECK(r3.status == 404);
  CHECK_EQ(r3.ctype, std::string("application/json"));
  CHECK(contains(r3.body, "\"error\":{"));
  CHECK(contains(r3.body, "\"type\":\"not_found_error\""));
  CHECK(contains(r3.body, "\"code\":\"unknown_url\""));
  CHECK(contains(r3.body, "\"param\":null"));

  // known route, wrong method -> 405
  Recorder r4;
  router.dispatch(make_request("POST", "/v1/models"), r4);
  CHECK(r4.status == 405);
  CHECK(contains(r4.body, "method_not_allowed"));
  CHECK(contains(r4.body, "GET"));

  // the handler sees the body
  Recorder r5;
  Request post = make_request("POST", "/v1/chat/completions");
  post.body = "{\"a\":1}";
  router.dispatch(post, r5);
  CHECK(r5.status == 200);
  CHECK_EQ(r5.body, std::string("{\"echo\":\"{\\\"a\\\":1}\"}"));

  // OPTIONS preflight on a known path -> 204 + Allow (no 405)
  Recorder r6;
  router.dispatch(make_request("OPTIONS", "/v1/chat/completions"), r6);
  CHECK(r6.status == 204);
  bool has_allow = false;
  for (const auto &h : r6.hdrs) has_allow = has_allow || h.first == "Allow";
  CHECK(has_allow);

  std::printf("router: %d checks so far\n", g_checks);
}

// ---------------------------------------------------------------------------
// Sockets
// ---------------------------------------------------------------------------
std::string raw_request(int port, const std::string &req) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return "<socket() failed>";
  struct timeval tv;
  tv.tv_sec = 5;
  tv.tv_usec = 0;
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  struct sockaddr_in a;
  std::memset(&a, 0, sizeof(a));
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (::connect(fd, reinterpret_cast<struct sockaddr *>(&a), sizeof(a)) != 0) {
    ::close(fd);
    return "<connect() failed>";
  }
  std::size_t off = 0;
  while (off < req.size()) {
    const ssize_t w = ::send(fd, req.data() + off, req.size() - off, 0);
    if (w <= 0) break;
    off += static_cast<std::size_t>(w);
  }
  std::string out;
  char buf[4096];
  for (;;) {
    const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n <= 0) break;
    out.append(buf, static_cast<std::size_t>(n));
  }
  ::close(fd);
  return out;
}

void check_sockets() {
  Router router;
  router.add("GET", "/health", [](const Request &, Responder &out) {
    out.send(200, "application/json", "{\"status\":\"ok\"}", Headers{});
  });
  router.add("POST", "/v1/chat/completions", [](const Request &q, Responder &out) {
    if (!q.body.empty()) {
      out.send(200, "application/json", "{\"len\":" + std::to_string(q.body.size()) + "}",
               Headers{});
      return;
    }
    out.begin_stream(200, "text/event-stream", Headers{});
    out.stream_chunk("data: {\"a\":1}\n\n");
    out.stream_chunk("data: [DONE]\n\n");
    out.end_stream();
  });

  ServerOptions opts;
  opts.host = "127.0.0.1";
  opts.port = 0;  // let the kernel pick a free port
  opts.recv_timeout_s = 0.2;  // short on purpose: test 9 below only proves the
                              // 408 timeout path, so 0.2 s is plenty (was 1 s)
  opts.max_body = 1024;
  rdna4::server::HttpServer server(opts, &router);
  std::string err;
  CHECK(server.open(err));
  if (!err.empty()) std::printf("open err: %s\n", err.c_str());
  CHECK(server.port() > 0);
  std::thread th([&] {
    std::string serr;
    server.serve(serr);
  });
  const int port = server.port();

  // 1. a plain GET
  std::string resp = raw_request(port, "GET /health HTTP/1.1\r\nHost: x\r\n\r\n");
  CHECK(contains(resp, "HTTP/1.1 200 OK"));
  CHECK(contains(resp, "Content-Length: 15"));
  CHECK(contains(resp, "Connection: close"));
  CHECK(contains(resp, "{\"status\":\"ok\"}"));

  // 2. the query string is not part of the route
  resp = raw_request(port, "GET /health?probe=1 HTTP/1.1\r\n\r\n");
  CHECK(contains(resp, "HTTP/1.1 200 OK"));

  // 3. chunked SSE framing, byte for byte
  resp = raw_request(port, "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 0\r\n\r\n");
  CHECK(contains(resp, "Content-Type: text/event-stream"));
  CHECK(contains(resp, "Transfer-Encoding: chunked"));
  // "data: {\"a\":1}\n\n" is 15 bytes (0xf), "data: [DONE]\n\n" is 14 (0xe)
  CHECK(contains(resp, "\r\n\r\nf\r\ndata: {\"a\":1}\n\n\r\n"));
  CHECK(contains(resp, "e\r\ndata: [DONE]\n\n\r\n"));
  CHECK(contains(resp, "0\r\n\r\n"));
  // 4. a body reaches the handler
  resp = raw_request(port,
                     "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello");
  CHECK(contains(resp, "{\"len\":5}"));

  // 5. 404 and 405
  resp = raw_request(port, "GET /v1/whatever HTTP/1.1\r\n\r\n");
  CHECK(contains(resp, "HTTP/1.1 404 Not Found"));
  CHECK(contains(resp, "unknown_url"));
  resp = raw_request(port, "DELETE /health HTTP/1.1\r\n\r\n");
  CHECK(contains(resp, "HTTP/1.1 405 Method Not Allowed"));

  // 6. chunked request bodies are refused with 411
  resp = raw_request(port,
                     "POST /v1/chat/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n");
  CHECK(contains(resp, "HTTP/1.1 411 Length Required"));

  // 7. too-large body -> 413 (max_body is 1024 here)
  resp = raw_request(port, "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 99999\r\n\r\n");
  CHECK(contains(resp, "HTTP/1.1 413 Payload Too Large"));

  // 7b. the same 413, but with the body actually being written: the server
  // drains a bounded amount before answering so the response is not lost to a
  // TCP reset (a client that gets ECONNRESET instead of the error is a bug
  // report waiting to happen). 4096 bytes is plenty: drain logic is
  // size-independent (was 99999, same 413 path, just slower).
  {
    const std::string big(4096, 'A');
    resp = raw_request(port, "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 4096\r\n\r\n" + big);
    CHECK(contains(resp, "HTTP/1.1 413 Payload Too Large"));
    CHECK(contains(resp, "unknown route") == false);  // the 413 body, not something else
  }

  // 8. malformed request line -> 400
  resp = raw_request(port, "GARBAGE\r\n\r\n");
  CHECK(contains(resp, "HTTP/1.1 400 Bad Request"));

  // 9. a body that never arrives -> 408 after recv_timeout_s (0.2 s here)
  resp = raw_request(port, "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 10\r\n\r\n");
  CHECK(contains(resp, "HTTP/1.1 408 Request Timeout"));

  // 10. serve() stops on request_stop() (the SIGINT path)
  server.stop();
  th.join();
  std::printf("sockets: %d checks so far\n", g_checks);
}

}  // namespace

int main() {
  check_json();
  check_router();
  check_sockets();
  std::printf("check-server-http: %d checks, %d failure(s)\n", g_checks, g_fails);
  return g_fails == 0 ? 0 : 1;
}
