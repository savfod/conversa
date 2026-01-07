import logging
import os
import queue
from typing import Optional

from flask import Flask, render_template, request
from flask_socketio import SocketIO

# Initialize Flask and SocketIO
app = Flask(__name__, template_folder=os.path.dirname(__file__))
# Note: In a real app, you might want to configure cors_allowed_origins more strictly
socketio = SocketIO(app, cors_allowed_origins="*")

# Queue for incoming audio (sid, audio_bytes)
input_queue: queue.Queue = queue.Queue()

logger = logging.getLogger(__name__)


@app.route("/")
def index():
    """Serve the index HTML page."""
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    logger.info(f"Client connected: {request.sid}")


@socketio.on("disconnect")
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on("audio_in")
def handle_audio_in(data):
    """
    Receives raw PCM audio bytes from browser.
    Stores (sid, bytes) in processing queue.
    """
    sid = request.sid
    # We push a tuple of (sid, data) so the consumer knows the source
    input_queue.put((sid, data))


def emit_audio_out(data: bytes, sid: Optional[str] = None):
    """
    Emit audio data to the client(s).

    Args:
        data: Audio bytes (WAV or raw PCM depending on client expectation,
              current client expects WAV).
        sid: Session ID to send to. If None, broadcasts to all.
    """
    if sid:
        socketio.emit("audio_out", data, to=sid)
    else:
        socketio.emit("audio_out", data)


def emit_audio_stop(sid: Optional[str] = None):
    """
    Emit stop audio signal to the client(s).

    Args:
        sid: Session ID to send to. If None, broadcasts to all.
    """
    if sid:
        socketio.emit("audio_stop", to=sid)
    else:
        socketio.emit("audio_stop")


def run_server(host="127.0.0.1", port=5555, debug=True):
    """Run the Flask-SocketIO server."""
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
