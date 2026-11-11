import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# A local web server that shows a Monitor (monitor.py) in the browser: dashboard.html draws the models, and receives
# their state through Server-Sent Events (a response that stays open, where the server writes a new message whenever
# the state changes). It only listens on this machine (127.0.0.1) and uses only the standard library.

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


class Dashboard:
    def __init__(self, monitor, port=8000, max_updates_per_second=5):
        self.monitor = monitor
        self.port = port
        self.interval = 1 / max_updates_per_second
        self.server = None

    # Starts the server in a background thread, on the first free port from self.port. Returns its address.
    def start(self):
        handler = self.make_handler()

        for port in range(self.port, self.port + 20):
            try:
                self.server = ThreadingHTTPServer(("127.0.0.1", port), handler)
                break
            except OSError:  # the port is in use
                continue
        else:
            raise OSError(f"no free port between {self.port} and {self.port + 19}")

        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://localhost:{self.server.server_address[1]}"

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    def make_handler(self):
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    with open(PAGE, "rb") as file:
                        self.reply(200, "text/html; charset=utf-8", file.read())
                elif self.path == "/state":
                    self.reply(200, "application/json", dashboard.monitor.to_json().encode())
                elif self.path == "/events":
                    self.stream_events()
                else:
                    self.reply(404, "text/plain", b"not found")

            def reply(self, status, content_type, body):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            # Sends the state whenever it changes, until the browser closes the page
            def stream_events(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last_version = None

                try:
                    while True:
                        if dashboard.monitor.version != last_version:
                            last_version = dashboard.monitor.version
                            self.wfile.write(b"data: " + dashboard.monitor.to_json().encode() + b"\n\n")
                            self.wfile.flush()

                        time.sleep(dashboard.interval)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the page was closed

            def log_message(self, format, *args):
                pass  # no line in the terminal for every request

        return Handler
