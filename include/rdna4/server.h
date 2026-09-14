// Blocking, single-threaded HTTP/1.1 server for the `serve` subcommand.
//
// No HTTP library is available (SPEC.md: ROCm and nothing else), so this is
// `<sys/socket.h>` + a poll/accept loop. It is deliberately minimal and honest
// about what it is:
//
//   - ONE request per connection, answered with `Connection: close`. Keep-alive
//     is not implemented: the GPU is not reentrant here, the server generates
//     one request at a time anyway, and closing removes a whole class of
//     pipelining bugs from the review surface. Every current OpenAI client
//     (openai-python/js, httpx, requests, LangChain, Open WebUI, Continue,
//     aider) copes with a closed connection after the response — SSE included,
//     because the terminating `0\r\n\r\n` chunk delimits the stream.
//   - Streaming responses use `Transfer-Encoding: chunked` so the SSE events
//     reach the client as they are generated instead of being buffered until
//     the body ends.
//   - Timeouts everywhere: a slow or vanished client must not hang the server.
//     SO_RCVTIMEO bounds the header+body read, SO_SNDTIMEO the writes. A write
//     failure marks the responder `client_gone()`, and the generation loop in
//     serve.hip aborts on it instead of burning GPU time for nobody.
//   - SIGINT/SIGTERM stop the accept loop (self-pipe + poll), and
//     `stop_requested()` is polled by the generation loop.
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "rdna4/server_json.h"

namespace rdna4 {
namespace server {

using Headers = std::vector<std::pair<std::string, std::string>>;

// One parsed request. `path` is percent-decoded with the query string removed;
// `headers` names are lowercased.
struct Request {
  std::string method;
  std::string path;
  std::string query;
  std::string body;
  Headers headers;

  std::string header(const char *name) const;
};

// Writes exactly one response: either `send()` (Content-Length) or
// begin_stream() / stream_chunk() / end_stream() (chunked). The return value of
// every call is "the bytes reached the socket".
class Responder {
 public:
  virtual ~Responder() = default;
  virtual bool send(int status, const std::string &content_type, const std::string &body,
                    const Headers &extra) = 0;
  virtual bool begin_stream(int status, const std::string &content_type, const Headers &extra) = 0;
  virtual bool stream_chunk(const std::string &chunk) = 0;
  virtual bool end_stream() = 0;
  virtual bool client_gone() const = 0;
};

using RouteFn = std::function<void(const Request &, Responder &)>;

// Exact-path routes win over prefix routes; both keep the order they were added
// in, so a handler can shadow a prefix with a more specific path.
//
// Routing lives in this CPU-only half on purpose: "unknown route -> 404, known
// route with the wrong method -> 405" is testable in tests/check_server_http.cpp
// without a GPU (no HIP in this translation unit).
class Router {
 public:
  void add(const std::string &method, const std::string &path, RouteFn fn);
  void add_prefix(const std::string &method, const std::string &prefix, RouteFn fn);
  void dispatch(const Request &req, Responder &out) const;

 private:
  struct Route {
    std::string method;
    std::string path;
    bool prefix = false;
    RouteFn fn;
  };
  std::vector<Route> routes_;
};

// {"error":{"message":..,"type":..,"param":null,"code":..}} — the shape every
// OpenAI client understands (they surface `error.message` verbatim).
json::Value openai_error(const std::string &message, const std::string &type,
                         const std::string &code);
void send_error(Responder &out, int status, const std::string &message, const std::string &type,
                const std::string &code);

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8080;
  int backlog = 16;
  int recv_timeout_s = 30;   // one request's header+body, per recv() call
  int send_timeout_s = 120;  // per send() call
  std::size_t max_body = 8u << 20;
  bool verbose = false;
};

// Two-phase on purpose: `open()` binds and listens, `serve()` runs the blocking
// accept loop. `serve` calls open() first so a busy port fails in milliseconds
// instead of after the 20+ s model load ("--port 8080 already in use" must not
// cost a full GGUF read).
//
// Sets the same flag SIGINT sets (one write to the self-pipe), which makes
// serve() return and the generation loop stop.
void request_stop();

// `serve()` returns when the socket fails, when stop_requested() becomes true
// (SIGINT/SIGTERM) or after a fatal error, in which case `err` carries the
// message. The bound port is available through port() (useful with --port 0,
// where the kernel picks one).
class HttpServer {
 public:
  HttpServer(const ServerOptions &opts, const Router *router) : opts_(opts), router_(router) {}
  ~HttpServer();
  bool open(std::string &err);
  bool serve(std::string &err);
  bool run(std::string &err) { return open(err) && serve(err); }
  // Wakes the accept loop from another thread (tests; it is also what SIGINT
  // does). Bound to the same flag the generation loop polls.
  void stop() { request_stop(); }
  int port() const { return bound_port_; }

 private:
  bool handle_connection(int fd);

  ServerOptions opts_;
  const Router *router_;
  int listen_fd_ = -1;
  int bound_port_ = 0;
};

// True once SIGINT/SIGTERM arrived (set by the handler installed in run()).
bool stop_requested();


// Human-readable status line text; unknown codes become "Unknown".
const char *status_text(int status);

// Installs the SIGINT/SIGTERM handlers (idempotent). Called by run(); exposed so
// a test can exercise the flag without a socket.
void install_signal_handlers();

}  // namespace server
}  // namespace rdna4
