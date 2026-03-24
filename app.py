# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import signal
import sys
from types import FrameType

from flask import Flask, jsonify, render_template, request

from utils.inference import get_model_debug_status, predict_sentiment
from utils.logging import logger

app = Flask(__name__)


@app.route("/")
def hello() -> str:
    # Use basic logging with custom fields
    logger.info(logField="custom-entry", arbitraryField="custom-entry")

    # https://cloud.google.com/run/docs/logging#correlate-logs
    logger.info("Child logger with trace Id.")

    return render_template("index.html")


@app.route("/api/llamar", methods=["POST"])
def llamar_api():
    logger.info("api_llamada_exitosa")
    return jsonify({"message": "api llamada exitosamente"})


@app.route("/api/infer", methods=["POST"])
def infer():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    logger.info("infer_request_received", text_length=len(text))

    if not text:
        logger.warning("infer_request_missing_text")
        return jsonify({"error": "Debes enviar un campo 'text' con contenido"}), 400

    try:
        result = predict_sentiment(text)
    except ValueError as exc:
        logger.warning("infer_request_validation_error", error=str(exc))
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        logger.exception("model_runtime_error")
        return jsonify({"error": str(exc)}), 503
    except Exception:
        logger.exception("inference_failed")
        return jsonify({"error": "Ocurrió un error durante la inferencia"}), 500

    logger.info("inference_completed", label=result["label"])
    return jsonify(result)


@app.route("/health/model")
def health_model():
    status = get_model_debug_status()
    http_status = 200 if status["model_loaded"] else 503
    logger.info("model_health_requested", **status)
    return jsonify(status), http_status


def shutdown_handler(signal_int: int, frame: FrameType) -> None:
    logger.info(f"Caught Signal {signal.strsignal(signal_int)}")

    from utils.logging import flush

    flush()

    # Safely exit program
    sys.exit(0)


if __name__ == "__main__":
    # Running application locally, outside of a Google Cloud Environment

    # handles Ctrl-C termination
    signal.signal(signal.SIGINT, shutdown_handler)

    app.run(host="localhost", port=8080, debug=True)
else:
    # handles Cloud Run container termination
    signal.signal(signal.SIGTERM, shutdown_handler)
