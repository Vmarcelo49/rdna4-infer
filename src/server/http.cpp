#include "rdna4/server.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rdna4 {
namespace server {

namespace {

constexpr std::size_t kMaxHeaderBytes = 64u << 10;

volatile sig_atomic_t g_stop = 0;
int g_stop_pipe[2] = {-1, -1};

void on_signal(int) {
  g_stop = 1;
  if (g_stop_pipe[1] >= 0) {
    const char b = 'x';
    ssize_t n = ::write(g_stop_pipe[1], &b, 1);
    (void)n;
  }
}

std::string lower(const std::string &s) {
  std::string out = s;
  for (char &c : out) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return out;
}

std::string trim(const std::string &s) {
  std::size_t a = 0, b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) --b;
  return s.substr(a, b - a);
}

bool starts_with(const std::string &s, const std::string &p) { return s.compare(0, p.size(), p) == 0; }

int hex_nibble(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// %XX in a path. An invalid escape is left alone rather than rejected: the
// result is compared against ASCII route paths, so a garbled byte just 404s.
std::string url_decode(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size()) {
      const int h = hex_nibble(s[i + 1]), l = hex_nibble(s[i + 2]);
      if (h >= 0 && l >= 0) {
        out.push_back(static_cast<char>((h << 4) | l));
        i += 2;
        continue;
      }
    }
    out.push_back(s[i]);
  }
  return out;
}

}  // namespace

const char *status_text(int status) {
  switch (status) {
    case 200: return "OK";
    case 201: return "Created";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 411: return "Length Required";
    case 413: return "Payload Too Large";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    default: return "Unknown";
  }
}

std::string Request::header(const char *name) const {
  const std::string want = lower(name == nullptr ? "" : name);
  for (const auto &h : headers) {
    if (h.first == want) return h.second;
  }
  return std::string();
}

bool stop_requested() { return g_stop != 0; }

void request_stop() {
  g_stop = 1;
  if (g_stop_pipe[1] >= 0) {
    const char b = 'x';
    const ssize_t n = ::write(g_stop_pipe[1], &b, 1);
    (void)n;
  }
}

void install_signal_handlers() {
  static bool done = false;
  if (done) return;
  done = true;
  if (::pipe(g_stop_pipe) != 0) {
    g_stop_pipe[0] = g_stop_pipe[1] = -1;
  }
  struct sigaction sa;
  std::memset(&sa, 0, sizeof(sa));
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;  // no SA_RESTART: poll() and recv() must come back with EINTR
  ::sigaction(SIGINT, &sa, nullptr);
  ::sigaction(SIGTERM, &sa, nullptr);
  ::signal(SIGPIPE, SIG_IGN);
}

// ---------------------------------------------------------------------------
// Responder over one socket.
// ---------------------------------------------------------------------------
namespace {

class SocketResponder : public Responder {
 public:
  SocketResponder(int fd, bool verbose) : fd_(fd), verbose_(verbose) {}

  bool send(int status, const std::string &content_type, const std::string &body,
            const Headers &extra) override {
    if (started_) return false;
    started_ = true;
    std::string head = status_line(status);
    if (!content_type.empty()) head += "Content-Type: " + content_type + "\r\n";
    head += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    head += extra_headers(extra);
    head += "\r\n";
    if (!write_all(head)) return false;
    if (!body.empty() && !write_all(body)) return false;
    finished_ = true;
    return true;
  }

  bool begin_stream(int status, const std::string &content_type, const Headers &extra) override {
    if (started_) return false;
    started_ = true;
    std::string head = status_line(status);
    head += "Content-Type: " + content_type + "\r\n";
    head += "Transfer-Encoding: chunked\r\n";
    head += "Cache-Control: no-cache\r\n";
    head += "X-Accel-Buffering: no\r\n";
    head += extra_headers(extra);
    head += "\r\n";
    return write_all(head);
  }

  bool stream_chunk(const std::string &chunk) override {
    if (!started_ || finished_) return false;
    if (chunk.empty()) return true;
    char len[32];
    std::snprintf(len, sizeof(len), "%zx\r\n", chunk.size());
    if (!write_all(len)) return false;
    if (!write_all(chunk)) return false;
    return write_all("\r\n");
  }

  bool end_stream() override {
    if (!started_ || finished_) return false;
    finished_ = true;
    return write_all("0\r\n\r\n");
  }

  bool client_gone() const override { return gone_; }
  bool answered() const { return started_; }

 private:
  static std::string status_line(int status) {
    return "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) + "\r\n";
  }

  std::string extra_headers(const Headers &extra) const {
    std::string out = "Server: rdna4-infer\r\nConnection: close\r\n";
    // Browser-based UIs (a page talking straight to this server) need CORS; it
    // costs nothing for the server-to-server clients and nothing is exposed that
    // a local process could not reach anyway.
    out += "Access-Control-Allow-Origin: *\r\n";
    for (const auto &h : extra) out += h.first + ": " + h.second + "\r\n";
    return out;
  }

  bool write_all(const std::string &s) { return write_all(s.data(), s.size()); }

  bool write_all(const char *data, std::size_t n) {
    std::size_t off = 0;
    while (off < n) {
      const ssize_t w = ::send(fd_, data + off, n - off, MSG_NOSIGNAL);
      if (w > 0) {
        off += static_cast<std::size_t>(w);
        continue;
      }
      if (w < 0 && errno == EINTR) {
        if (g_stop) {
          gone_ = true;
          return false;
        }
        continue;
      }
      // EAGAIN/EWOULDBLOCK here is SO_SNDTIMEO expiring; anything else is a
      // dead connection. Either way the client is not reading.
      gone_ = true;
      if (verbose_) std::fprintf(stderr, "[serve] write failed: %s\n", std::strerror(errno));
      return false;
    }
    return true;
  }

  int fd_;
  bool verbose_;
  bool started_ = false;
  bool finished_ = false;
  bool gone_ = false;
};

// Reads one request. On success returns true. On failure: `*status` is 0 when
// there is nothing to answer (the peer closed or the connection broke) and an
// HTTP status when a response explaining the problem should be sent.
bool read_request(int fd, const ServerOptions &opts, Request &req, int *status, std::string *msg) {
  *status = 0;
  std::string buf;
  char tmp[8192];
  std::size_t head_end = std::string::npos;
  std::size_t sep = 0;
  for (;;) {
    const std::size_t crlf = buf.find("\r\n\r\n");
    const std::size_t lf = buf.find("\n\n");
    if (crlf != std::string::npos && (lf == std::string::npos || crlf <= lf)) {
      head_end = crlf;
      sep = 4;
      break;
    }
    if (lf != std::string::npos) {
      head_end = lf;
      sep = 2;
      break;
    }
    if (buf.size() > kMaxHeaderBytes) {
      *status = 431;
      *msg = "request headers too large";
      return false;
    }
    const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
    if (n > 0) {
      buf.append(tmp, static_cast<std::size_t>(n));
      continue;
    }
    if (n == 0) {
      if (buf.empty()) return false;  // connect + close (a port probe): stay quiet
      *status = 400;
      *msg = "connection closed before the request was complete";
      return false;
    }
    if (errno == EINTR) {
      if (g_stop) return false;
      continue;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      *status = 408;
      *msg = "timed out reading the request";
      return false;
    }
    return false;
  }

  std::size_t pos = 0;
  // request line
  std::size_t line_end = buf.find('\n', pos);
  if (line_end == std::string::npos) line_end = head_end;
  std::string line = trim(buf.substr(pos, line_end - pos));
  pos = line_end + 1;
  {
    const std::size_t s1 = line.find(' ');
    const std::size_t s2 = s1 == std::string::npos ? std::string::npos : line.find(' ', s1 + 1);
    if (s1 == std::string::npos || s2 == std::string::npos) {
      *status = 400;
      *msg = "malformed request line";
      return false;
    }
    req.method = line.substr(0, s1);
    std::string target = line.substr(s1 + 1, s2 - s1 - 1);
    // absolute-form (a proxy-style request line): drop scheme://host
    if (starts_with(target, "http://") || starts_with(target, "https://")) {
      const std::size_t slash = target.find('/', target.find("//") + 2);
      target = slash == std::string::npos ? "/" : target.substr(slash);
    }
    const std::size_t q = target.find('?');
    if (q == std::string::npos) {
      req.path = url_decode(target);
    } else {
      req.path = url_decode(target.substr(0, q));
      req.query = target.substr(q + 1);
    }
    // "/v1/models/" and "/v1/models" are the same route
    if (req.path.size() > 1 && req.path.back() == '/') req.path.pop_back();
    if (req.method.empty() || req.path.empty() || req.path[0] != '/') {
      *status = 400;
      *msg = "malformed request target";
      return false;
    }
  }
  // headers
  while (pos <= head_end) {
    std::size_t e = buf.find('\n', pos);
    if (e == std::string::npos || e > head_end) e = head_end;
    const std::string raw = buf.substr(pos, e - pos);
    pos = e + 1;
    const std::string h = trim(raw);
    if (h.empty()) break;
    if (raw[0] == ' ' || raw[0] == '\t') {
      *status = 400;
      *msg = "obsolete header line folding is not supported";
      return false;
    }
    const std::size_t colon = h.find(':');
    if (colon == std::string::npos) {
      *status = 400;
      *msg = "malformed header line";
      return false;
    }
    req.headers.emplace_back(lower(trim(h.substr(0, colon))), trim(h.substr(colon + 1)));
  }

  const std::string te = req.header("transfer-encoding");
  if (!te.empty() && lower(te).find("chunked") != std::string::npos) {
    *status = 411;
    *msg = "chunked request bodies are not supported: send Content-Length";
    return false;
  }
  std::size_t content_length = 0;
  const std::string cl = req.header("content-length");
  if (!cl.empty()) {
    if (cl.find_first_not_of("0123456789") != std::string::npos) {
      *status = 400;
      *msg = "malformed Content-Length";
      return false;
    }
    content_length = static_cast<std::size_t>(std::strtoull(cl.c_str(), nullptr, 10));
  }
  if (content_length > opts.max_body) {
    *status = 413;
    *msg = "request body larger than " + std::to_string(opts.max_body) + " bytes";
    return false;
  }

  const std::string expect = lower(req.header("expect"));
  if (expect.find("100-continue") != std::string::npos) {
    const char interim[] = "HTTP/1.1 100 Continue\r\n\r\n";
    std::size_t off = 0;
    while (off < sizeof(interim) - 1) {
      const ssize_t w = ::send(fd, interim + off, sizeof(interim) - 1 - off, MSG_NOSIGNAL);
      if (w > 0) {
        off += static_cast<std::size_t>(w);
        continue;
      }
      if (w < 0 && errno == EINTR) continue;
      *status = 0;
      return false;
    }
  }

  const std::size_t body_start = head_end + sep;
  req.body = buf.substr(body_start < buf.size() ? body_start : buf.size());
  while (req.body.size() < content_length) {
    const std::size_t want = std::min<std::size_t>(sizeof(tmp), content_length - req.body.size());
    const ssize_t n = ::recv(fd, tmp, want, 0);
    if (n > 0) {
      req.body.append(tmp, static_cast<std::size_t>(n));
      continue;
    }
    if (n < 0 && errno == EINTR) {
      if (g_stop) return false;
      continue;
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      *status = 408;
      *msg = "timed out reading the request body";
      return false;
    }
    *status = 400;
    *msg = "connection closed in the middle of the request body";
    return false;
  }
  if (req.body.size() > content_length) req.body.resize(content_length);
  return true;
}

// Reads (a bounded amount of) a body the client is still sending before an
// error response. Without this, closing the socket under a big POST makes the
// kernel send an RST and the already-written 413/411 body can be discarded, so
// the client reports a connection error instead of the explanation. The cap and
// the 1 s timeout keep a slow or hostile client from stalling the server here.
void drain_body(int fd, std::size_t announced) {
  if (announced == 0) return;
  struct timeval tv;
  tv.tv_sec = 1;
  tv.tv_usec = 0;
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  char tmp[8192];
  std::size_t left = std::min<std::size_t>(announced, 256u << 10);
  while (left > 0) {
    const ssize_t n = ::recv(fd, tmp, std::min(sizeof(tmp), left), 0);
    if (n <= 0) break;
    left -= static_cast<std::size_t>(n);
  }
}

void set_timeouts(int fd, const ServerOptions &opts) {
  struct timeval tv;
  tv.tv_sec = opts.recv_timeout_s;
  tv.tv_usec = 0;
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  tv.tv_sec = opts.send_timeout_s;
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  const int one = 1;
  // Nagle would sit on a small SSE chunk waiting for an ACK: the streaming
  // point of the server is that tokens leave immediately.
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

}  // namespace

// ---------------------------------------------------------------------------
// Router.
// ---------------------------------------------------------------------------
void Router::add(const std::string &method, const std::string &path, RouteFn fn) {
  routes_.push_back({method, path, false, std::move(fn)});
}

void Router::add_prefix(const std::string &method, const std::string &prefix, RouteFn fn) {
  routes_.push_back({method, prefix, true, std::move(fn)});
}

void Router::dispatch(const Request &req, Responder &out) const {
  auto methods_for = [&](auto pred) {
    std::vector<std::string> ms;
    for (const Route &r : routes_) {
      if (!pred(r)) continue;
      if (std::find(ms.begin(), ms.end(), r.method) == ms.end()) ms.push_back(r.method);
    }
    return ms;
  };
  auto reply_405 = [&](const std::vector<std::string> &ms) {
    std::string allow_list;
    for (const std::string &m : ms) allow_list += (allow_list.empty() ? "" : ", ") + m;
    if (req.method == "OPTIONS") {
      Headers h = {{"Allow", allow_list},
                   {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
                   {"Access-Control-Allow-Headers", "*"},
                   {"Content-Length", "0"}};
      out.send(204, "", "", h);
      return;
    }
    send_error(out, 405, "method " + req.method + " is not allowed for " + req.path +
                             " (allowed: " + allow_list + ")",
               "invalid_request_error", "method_not_allowed");
  };

  // exact matches first
  {
    const auto same = [&](const Route &r) { return !r.prefix && r.path == req.path; };
    const std::vector<std::string> ms = methods_for(same);
    if (!ms.empty()) {
      for (const Route &r : routes_) {
        if (same(r) && r.method == req.method) {
          r.fn(req, out);
          return;
        }
      }
      reply_405(ms);
      return;
    }
  }
  // then the longest matching prefix
  {
    const Route *best = nullptr;
    for (const Route &r : routes_) {
      if (!r.prefix || !starts_with(req.path, r.path)) continue;
      if (best == nullptr || r.path.size() > best->path.size()) best = &r;
    }
    if (best != nullptr) {
      const std::string prefix = best->path;
      const auto same = [&](const Route &r) { return r.prefix && r.path == prefix; };
      const std::vector<std::string> ms = methods_for(same);
      for (const Route &r : routes_) {
        if (same(r) && r.method == req.method) {
          r.fn(req, out);
          return;
        }
      }
      reply_405(ms);
      return;
    }
  }
  send_error(out, 404, "unknown route " + req.method + " " + req.path + " (see /v1/models, "
                       "/v1/chat/completions, /v1/completions, /health)",
             "not_found_error", "unknown_url");
}

// ---------------------------------------------------------------------------
// Error bodies.
// ---------------------------------------------------------------------------
json::Value openai_error(const std::string &message, const std::string &type,
                         const std::string &code) {
  json::Value err = json::Value::object();
  err.set("message", json::Value::string(message));
  err.set("type", json::Value::string(type));
  err.set("param", json::Value::null());
  err.set("code", code.empty() ? json::Value::null() : json::Value::string(code));
  json::Value root = json::Value::object();
  root.set("error", std::move(err));
  return root;
}

void send_error(Responder &out, int status, const std::string &message, const std::string &type,
                const std::string &code) {
  out.send(status, "application/json", openai_error(message, type, code).dump(), Headers{});
}

// ---------------------------------------------------------------------------
// HttpServer.
// ---------------------------------------------------------------------------
HttpServer::~HttpServer() {
  if (listen_fd_ >= 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
}

bool HttpServer::open(std::string &err) {
  install_signal_handlers();
  err.clear();

  std::string host = opts_.host.empty() ? "127.0.0.1" : opts_.host;
  if (host == "localhost") host = "127.0.0.1";

  struct sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<std::uint16_t>(opts_.port));
  if (host == "*" || host == "any") host = "0.0.0.0";
  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    err = "cannot parse --host '" + opts_.host + "' (IPv4 addresses only, e.g. 127.0.0.1 or 0.0.0.0)";
    return false;
  }

  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    err = std::string("socket() failed: ") + std::strerror(errno);
    return false;
  }
  const int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  if (::bind(fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) != 0) {
    err = "cannot bind " + host + ":" + std::to_string(opts_.port) + ": " + std::strerror(errno);
    if (errno == EADDRINUSE) err += " (another process is already listening there)";
    ::close(fd);
    return false;
  }
  if (::listen(fd, opts_.backlog) != 0) {
    err = std::string("listen() failed: ") + std::strerror(errno);
    ::close(fd);
    return false;
  }
  struct sockaddr_in bound;
  socklen_t blen = sizeof(bound);
  if (::getsockname(fd, reinterpret_cast<struct sockaddr *>(&bound), &blen) == 0) {
    bound_port_ = ntohs(bound.sin_port);
  } else {
    bound_port_ = opts_.port;
  }
  listen_fd_ = fd;
  return true;
}

bool HttpServer::serve(std::string &err) {
  err.clear();
  if (listen_fd_ < 0) {
    err = "serve() called before open()";
    return false;
  }
  const int stop_fd = g_stop_pipe[0];
  for (;;) {
    struct pollfd fds[2];
    fds[0].fd = listen_fd_;
    fds[0].events = POLLIN;
    fds[0].revents = 0;
    int nfds = 1;
    if (stop_fd >= 0) {
      fds[1].fd = stop_fd;
      fds[1].events = POLLIN;
      fds[1].revents = 0;
      nfds = 2;
    }
    const int pr = ::poll(fds, static_cast<nfds_t>(nfds), -1);
    if (pr < 0) {
      if (errno == EINTR) {
        if (g_stop) break;
        continue;
      }
      err = std::string("poll() failed: ") + std::strerror(errno);
      return false;
    }
    if (stop_fd >= 0 && (fds[1].revents & POLLIN)) break;
    if (!(fds[0].revents & POLLIN)) continue;

    struct sockaddr_in peer;
    socklen_t plen = sizeof(peer);
    const int cfd = ::accept(listen_fd_, reinterpret_cast<struct sockaddr *>(&peer), &plen);
    if (cfd < 0) {
      if (errno == EINTR || errno == ECONNABORTED) {
        if (g_stop) break;
        continue;
      }
      err = std::string("accept() failed: ") + std::strerror(errno);
      return false;
    }
    set_timeouts(cfd, opts_);
    handle_connection(cfd);
    ::close(cfd);
    if (g_stop) break;
  }
  if (listen_fd_ >= 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  return true;
}

bool HttpServer::handle_connection(int fd) {
  Request req;
  int status = 0;
  std::string msg;
  SocketResponder out(fd, opts_.verbose);
  if (!read_request(fd, opts_, req, &status, &msg)) {
    if (status == 413 || status == 411) {
      const std::string cl = req.header("content-length");
      if (!cl.empty() && cl.find_first_not_of("0123456789") == std::string::npos) {
        drain_body(fd, static_cast<std::size_t>(std::strtoull(cl.c_str(), nullptr, 10)));
      }
    }
    if (status != 0) {
      const char *type = status == 404 ? "not_found_error"
                                       : (status >= 500 ? "server_error" : "invalid_request_error");
      send_error(out, status, msg, type, "");
      if (opts_.verbose) {
        std::fprintf(stderr, "[serve] %d %s\n", status, msg.c_str());
      }
    }
    return false;
  }
  if (router_ != nullptr) {
    router_->dispatch(req, out);
  } else {
    send_error(out, 500, "no routes registered", "server_error", "internal_error");
  }
  if (!out.answered()) {
    send_error(out, 500, "the handler produced no response", "server_error", "internal_error");
  }
  return true;
}

}  // namespace server
}  // namespace rdna4
