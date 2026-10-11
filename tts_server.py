import io
import json
import os
import sys
import wave
import threading
from pathlib import Path

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# ============================================================
# CONFIGURATION
# ============================================================

HOST = os.environ.get("TTS_HOST", "127.0.0.1")
PORT = int(os.environ.get("TTS_PORT", "8880"))

API_KEY = os.environ.get("TTS_API_KEY", "tts-local")

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# PIPER
# ============================================================

PIPER_DIR = BASE_DIR / "piper"

PIPER_MODELS = {
    "ru_RU-irina-medium": PIPER_DIR / "ru_RU-irina-medium.onnx",
    "en_US-libritts-high": PIPER_DIR / "en_US-libritts-high.onnx",
}

DEFAULT_PIPER_VOICE = "ru_RU-irina-medium"


# ============================================================
# MODEL ALIASES
# ============================================================

MODEL_ALIASES = {
    "tts-1": "silero",
    "tts-1-hd": "silero",
    "ru-irina-medium": "baya",
}


# ============================================================
# VOICE ALIASES
# ============================================================

VOICE_ALIASES = {
    "alloy": "kseniya",
    "echo": "aidar",
    "fable": "baya",
    "onyx": "eugene",
    "nova": "xenia",
    "shimmer": "kseniya",

    # Your custom alias
    "ru-irina-medium": "baya",
}


# ============================================================
# SILERO
# ============================================================

SILERO_LANGUAGE = "ru"
SILERO_MODEL = "v5_ru"

SILERO_VOICES = {
    "aidar",
    "baya",
    "kseniya",
    "xenia",
    "eugene",
}

SILERO_SAMPLE_RATE = 48000


# ============================================================
# KOKORO
# ============================================================

KOKORO_VOICES = {
    "sveta",
    "masha",
    "dima",
}

KOKORO_DEFAULT_VOICE = "sveta"
KOKORO_SAMPLE_RATE = 24000

KOKORO_DIR = BASE_DIR / "kokoro"
KOKORO_ONNX_DIR = KOKORO_DIR / "onnx"
KOKORO_VOICE_DIR = KOKORO_DIR / "voices"

KOKORO_CONFIG_PATH = KOKORO_DIR / "kokoro-config.json"
KOKORO_G2P_PATH = KOKORO_DIR / "ru_g2p.py"

KOKORO_ONNX_BASE = KOKORO_ONNX_DIR / "model.onnx"
KOKORO_ONNX_DIMA = KOKORO_ONNX_DIR / "model_dima.onnx"

KOKORO_VOICE_FILES = {
    "sveta": KOKORO_VOICE_DIR / "sveta.bin",
    "masha": KOKORO_VOICE_DIR / "masha.bin",
    "dima": KOKORO_VOICE_DIR / "dima.bin",
}


# ============================================================
# GLOBAL STATE
# ============================================================

_piper_cache = {}
_piper_lock = threading.Lock()

_silero_model = None
_silero_lock = threading.Lock()

_kokoro_g2p = None
_kokoro_g2p_lock = threading.Lock()

_kokoro_sessions = {}
_kokoro_voice_cache = {}

# N5100: serialize Kokoro synthesis.
# This prevents several requests from trying to consume
# all CPU/RAM simultaneously.
_kokoro_synthesis_lock = threading.Lock()

_kokoro_lock = threading.Lock()


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(message, flush=True)


# ============================================================
# PIPER
# ============================================================

def load_piper_voice(voice_name):
    """
    Load Piper voice lazily and cache it.
    """

    if voice_name not in PIPER_MODELS:
        raise ValueError(
            f"Unknown Piper voice: {voice_name}. "
            f"Available: {', '.join(PIPER_MODELS.keys())}"
        )

    model_path = Path(PIPER_MODELS[voice_name])

    if not model_path.exists():
        raise FileNotFoundError(
            f"Piper model not found: {model_path}"
        )

    with _piper_lock:

        if voice_name in _piper_cache:
            return _piper_cache[voice_name]

        log(f"[PIPER] Loading: {voice_name}")
        log(f"[PIPER] Model: {model_path}")

        from piper import PiperVoice

        config_path = model_path.with_suffix(
            model_path.suffix + ".json"
        )

        if not config_path.exists():
            raise FileNotFoundError(
                f"Piper config not found: {config_path}"
            )

        voice = PiperVoice.load(
            str(model_path),
            config_path=str(config_path),
        )

        _piper_cache[voice_name] = voice

        log(f"[PIPER] Loaded: {voice_name}")

        return voice


def synthesize_piper(text, voice_name):
    """
    Generate WAV bytes using Piper.
    """

    voice = load_piper_voice(voice_name)

    output = io.BytesIO()

    with wave.open(output, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    return output.getvalue()


# ============================================================
# SILERO
# ============================================================

def load_silero():
    """
    Load Silero model lazily.
    """

    global _silero_model

    with _silero_lock:

        if _silero_model is not None:
            return _silero_model

        log("[SILERO] Loading model...")
        log(
            f"[SILERO] language={SILERO_LANGUAGE}, "
            f"model={SILERO_MODEL}"
        )

        from silero import silero_tts

        model, example_text = silero_tts(
            language=SILERO_LANGUAGE,
            speaker=SILERO_MODEL,
        )

        try:
            model.to("cpu")
        except Exception:
            pass

        _silero_model = model

        log("[SILERO] Model loaded")
        log(
            "[SILERO] Available voices: "
            + ", ".join(sorted(SILERO_VOICES))
        )

        return _silero_model


def tensor_to_wav(audio_tensor, sample_rate):
    """
    Convert Silero float tensor [-1, 1]
    to 16-bit PCM WAV.
    """

    import torch

    if hasattr(audio_tensor, "detach"):
        audio_tensor = audio_tensor.detach()

    audio_tensor = audio_tensor.cpu().float()
    audio_tensor = audio_tensor.flatten()

    audio_tensor = torch.clamp(
        audio_tensor,
        -1.0,
        1.0,
    )

    audio_int16 = (
        audio_tensor * 32767.0
    ).to(torch.int16)

    pcm_data = audio_int16.numpy().tobytes()

    output = io.BytesIO()

    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)

    return output.getvalue()


def synthesize_silero(text, voice_name):
    """
    Generate WAV bytes using Silero.
    """

    if voice_name not in SILERO_VOICES:
        raise ValueError(
            f"Unknown Silero voice: {voice_name}. "
            f"Available: {', '.join(sorted(SILERO_VOICES))}"
        )

    model = load_silero()

    log(f"[SILERO] Generating voice={voice_name}")
    log(f"[SILERO] Text: {text[:100]}")

    audio = model.apply_tts(
        text=text,
        speaker=voice_name,
        sample_rate=SILERO_SAMPLE_RATE,
    )

    return tensor_to_wav(
        audio,
        SILERO_SAMPLE_RATE,
    )


# ============================================================
# KOKORO
# ============================================================

def check_kokoro_files():
    """
    Verify files required by Kokoro ONNX.
    """

    required = [
        KOKORO_CONFIG_PATH,
        KOKORO_G2P_PATH,
        KOKORO_ONNX_BASE,
        KOKORO_ONNX_DIMA,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Kokoro file not found: {path}"
            )

    for voice_name, path in KOKORO_VOICE_FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Kokoro voice file not found "
                f"for {voice_name}: {path}"
            )


def load_kokoro_g2p():
    """
    Load RuG2P lazily.
    """

    global _kokoro_g2p

    with _kokoro_g2p_lock:

        if _kokoro_g2p is not None:
            return _kokoro_g2p

        check_kokoro_files()

        log("[KOKORO] Loading RuG2P...")

        if str(KOKORO_DIR) not in sys.path:
            sys.path.insert(0, str(KOKORO_DIR))

        from ru_g2p import RuG2P

        _kokoro_g2p = RuG2P()

        log("[KOKORO] RuG2P loaded")

        return _kokoro_g2p


def load_kokoro_config():
    """
    Load Kokoro vocabulary from kokoro-config.json.
    """

    if not KOKORO_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Kokoro config not found: "
            f"{KOKORO_CONFIG_PATH}"
        )

    with open(
        KOKORO_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    vocab = config.get("vocab")

    if not vocab:
        raise ValueError(
            "Kokoro config.json does not contain 'vocab'"
        )

    return vocab


def load_kokoro_session(voice_name):
    """
    Load ONNX Runtime session lazily.

    sveta/masha -> model.onnx
    dima        -> model_dima.onnx
    """

    if voice_name not in KOKORO_VOICES:
        raise ValueError(
            f"Unknown Kokoro voice: {voice_name}. "
            f"Available: {', '.join(sorted(KOKORO_VOICES))}"
        )

    with _kokoro_lock:

        if voice_name in _kokoro_sessions:
            return _kokoro_sessions[voice_name]

        check_kokoro_files()

        import onnxruntime as ort

        if voice_name == "dima":
            model_path = KOKORO_ONNX_DIMA
        else:
            model_path = KOKORO_ONNX_BASE

        log(f"[KOKORO] Loading ONNX model for {voice_name}")
        log(f"[KOKORO] Model: {model_path}")

        # CPU only.
        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        _kokoro_sessions[voice_name] = session

        log(f"[KOKORO] ONNX loaded: {voice_name}")

        return session


def load_kokoro_voice(voice_name):
    """
    Load raw float32 voice style [510, 256].
    """

    if voice_name not in KOKORO_VOICE_FILES:
        raise ValueError(
            f"Unknown Kokoro voice: {voice_name}"
        )

    with _kokoro_lock:

        if voice_name in _kokoro_voice_cache:
            return _kokoro_voice_cache[voice_name]

        import numpy as np

        path = KOKORO_VOICE_FILES[voice_name]

        if not path.exists():
            raise FileNotFoundError(
                f"Kokoro voice file not found: {path}"
            )

        log(f"[KOKORO] Loading voice: {voice_name}")

        style = np.fromfile(
            path,
            dtype=np.float32,
        )

        expected = 510 * 256

        if style.size != expected:
            raise ValueError(
                f"Invalid Kokoro voice file: {path}. "
                f"Expected {expected} float32 values, "
                f"got {style.size}"
            )

        style = style.reshape(510, 256)

        _kokoro_voice_cache[voice_name] = style

        log(f"[KOKORO] Voice loaded: {voice_name}")

        return style


def synthesize_kokoro(text, voice_name):
    """
    Generate WAV bytes using Kokoro ONNX.
    """

    if voice_name not in KOKORO_VOICES:
        raise ValueError(
            f"Unknown Kokoro voice: {voice_name}. "
            f"Available: {', '.join(sorted(KOKORO_VOICES))}"
        )

    log(f"[KOKORO] Generating voice={voice_name}")
    log(f"[KOKORO] Text: {text[:100]}")

    # Kokoro on N5100 is CPU-heavy.
    # Serialize generation.
    with _kokoro_synthesis_lock:

        import numpy as np

        g2p = load_kokoro_g2p()

        # ----------------------------------------------------
        # G2P
        # ----------------------------------------------------

        ipa, oov = g2p(text)

        log(
            f"[KOKORO] IPA length={len(ipa)} "
            f"OOV={oov}"
        )

        if not ipa:
            raise ValueError(
                "Kokoro G2P produced empty IPA"
            )

        # Voice style contains 510 style vectors.
        if len(ipa) > 510:
            raise ValueError(
                f"Kokoro input is too long after G2P: "
                f"{len(ipa)} IPA symbols. "
                f"Maximum supported by voice pack is 510."
            )

        # ----------------------------------------------------
        # VOCAB
        # ----------------------------------------------------

        vocab = load_kokoro_config()

        ids = [
            vocab[c]
            for c in ipa
            if c in vocab
        ]

        if not ids:
            raise ValueError(
                "Kokoro produced no recognized phoneme IDs"
            )

        # ----------------------------------------------------
        # SESSION
        # ----------------------------------------------------

        session = load_kokoro_session(
            voice_name
        )

        # ----------------------------------------------------
        # VOICE STYLE
        # ----------------------------------------------------

        voice_style = load_kokoro_voice(
            voice_name
        )

        style = voice_style[
            len(ipa) - 1
        ][None]

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        input_ids = np.array(
            [[0, *ids, 0]],
            dtype=np.int64,
        )

        speed = np.ones(
            1,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        audio = session.run(
            None,
            {
                "input_ids": input_ids,
                "style": style,
                "speed": speed,
            },
        )[0]

        audio = np.asarray(
            audio,
            dtype=np.float32,
        ).squeeze()

        if audio.size == 0:
            raise RuntimeError(
                "Kokoro returned empty audio"
            )

        # ----------------------------------------------------
        # FLOAT -> PCM16
        # ----------------------------------------------------

        audio = np.clip(
            audio,
            -1.0,
            1.0,
        )

        audio_int16 = (
            audio * 32767.0
        ).astype(np.int16)

        pcm_data = audio_int16.tobytes()

        output = io.BytesIO()

        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(
                KOKORO_SAMPLE_RATE
            )
            wav_file.writeframes(pcm_data)

        log(
            f"[KOKORO] Generated "
            f"{len(audio) / KOKORO_SAMPLE_RATE:.3f} sec"
        )

        return output.getvalue()


# ============================================================
# VOICES / MODELS
# ============================================================

def get_voices():

    voices = []

    # Piper
    for voice_name in PIPER_MODELS:

        voices.append({
            "id": voice_name,
            "model": "piper",
            "voice": voice_name,
            "language": (
                "ru-RU"
                if voice_name.startswith("ru_")
                else "en-US"
            ),
        })

    # Silero
    for voice_name in sorted(SILERO_VOICES):

        voices.append({
            "id": voice_name,
            "model": "silero",
            "voice": voice_name,
            "language": "ru-RU",
        })

    # Kokoro
    for voice_name in sorted(KOKORO_VOICES):

        voices.append({
            "id": voice_name,
            "model": "kokoro",
            "voice": voice_name,
            "language": "ru-RU",
        })

    return voices


def get_models():

    return [
        {
            "id": "piper",
            "voices": list(PIPER_MODELS.keys()),
        },
        {
            "id": "silero",
            "voices": sorted(SILERO_VOICES),
        },
        {
            "id": "kokoro",
            "voices": sorted(KOKORO_VOICES),
        },
    ]


# ============================================================
# HTTP SERVER
# ============================================================

class TTSHandler(BaseHTTPRequestHandler):

    server_version = "LocalTTS/1.1"

    def log_message(self, format_string, *args):

        log(
            "[HTTP] "
            + format_string % args
        )

    def send_json(self, status_code, data):

        body = json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        self.send_response(status_code)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.end_headers()

        self.wfile.write(body)

    def send_wav(self, wav_data):

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "audio/wav",
        )

        self.send_header(
            "Content-Length",
            str(len(wav_data)),
        )

        self.end_headers()

        self.wfile.write(wav_data)

    def check_auth(self):

        authorization = self.headers.get(
            "Authorization",
            "",
        )

        expected = f"Bearer {API_KEY}"

        return authorization == expected

    def do_GET(self):

        if self.path == "/":

            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "Local TTS",
                    "endpoint": "/v1/audio/speech",
                },
            )

            return

        if self.path == "/health":

            self.send_json(
                200,
                {
                    "status": "ok",
                },
            )

            return

        if self.path == "/voices":

            self.send_json(
                200,
                {
                    "voices": get_voices(),
                },
            )

            return

        if self.path == "/v1/models":

            self.send_json(
                200,
                {
                    "data": get_models(),
                },
            )

            return

        self.send_json(
            404,
            {
                "error": "Not found",
            },
        )

    def do_POST(self):

        if self.path != "/v1/audio/speech":

            self.send_json(
                404,
                {
                    "error": "Not found",
                },
            )

            return

        if not self.check_auth():

            self.send_json(
                401,
                {
                    "error": "Unauthorized",
                },
            )

            return

        try:

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            if content_length <= 0:
                raise ValueError(
                    "Empty request body"
                )

            body = self.rfile.read(
                content_length
            )

            request = json.loads(
                body.decode("utf-8")
            )

            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            model_name = request.get(
                "model",
                "piper",
            )

            model_name = MODEL_ALIASES.get(
                model_name,
                model_name,
            )

            # ------------------------------------------------
            # VOICE
            # ------------------------------------------------

            voice_name = request.get(
                "voice"
            )

            if voice_name:

                voice_name = VOICE_ALIASES.get(
                    voice_name,
                    voice_name,
                )

            # ------------------------------------------------
            # INPUT
            # ------------------------------------------------

            text = request.get(
                "input"
            )

            if not text:

                raise ValueError(
                    "Missing 'input'"
                )

            if not isinstance(text, str):

                raise ValueError(
                    "'input' must be a string"
                )

            if len(text) > 10000:

                raise ValueError(
                    "Text is too long "
                    "(maximum 10000 characters)"
                )

            # ------------------------------------------------
            # PIPER
            # ------------------------------------------------

            if model_name == "piper":

                if not voice_name:

                    voice_name = (
                        DEFAULT_PIPER_VOICE
                    )

                log(
                    f"[REQUEST] "
                    f"model=piper "
                    f"voice={voice_name}"
                )

                wav_data = synthesize_piper(
                    text,
                    voice_name,
                )

            # ------------------------------------------------
            # SILERO
            # ------------------------------------------------

            elif model_name == "silero":

                if not voice_name:

                    voice_name = "kseniya"

                log(
                    f"[REQUEST] "
                    f"model=silero "
                    f"voice={voice_name}"
                )

                wav_data = synthesize_silero(
                    text,
                    voice_name,
                )

            # ------------------------------------------------
            # KOKORO
            # ------------------------------------------------

            elif model_name == "kokoro":

                if not voice_name:

                    voice_name = (
                        KOKORO_DEFAULT_VOICE
                    )

                log(
                    f"[REQUEST] "
                    f"model=kokoro "
                    f"voice={voice_name}"
                )

                wav_data = synthesize_kokoro(
                    text,
                    voice_name,
                )

            # ------------------------------------------------
            # UNKNOWN
            # ------------------------------------------------

            else:

                raise ValueError(
                    f"Unknown model: {model_name}. "
                    f"Available: "
                    f"piper, silero, kokoro"
                )

            self.send_wav(
                wav_data
            )

            log(
                f"[OK] Generated "
                f"{len(wav_data)} bytes"
            )

        except Exception as exc:

            log(
                f"[ERROR] "
                f"{type(exc).__name__}: {exc}"
            )

            self.send_json(
                500,
                {
                    "error": str(exc),
                    "type": type(exc).__name__,
                },
            )


# ============================================================
# MAIN
# ============================================================

def main():

    log("=" * 60)
    log("LOCAL TTS SERVER")
    log("=" * 60)

    log(f"Python: {sys.version}")
    log(f"Base directory: {BASE_DIR}")
    log(f"Listening: http://{HOST}:{PORT}")
    log(f"API key: {API_KEY}")

    # --------------------------------------------------------
    # Piper
    # --------------------------------------------------------

    log("")
    log("Piper voices:")

    for voice_name, path in PIPER_MODELS.items():

        exists = (
            "OK"
            if path.exists()
            else "MISSING"
        )

        log(
            f"  {voice_name}: "
            f"{path} [{exists}]"
        )

    # --------------------------------------------------------
    # Silero
    # --------------------------------------------------------

    log("")
    log("Silero voices:")

    for voice_name in sorted(
        SILERO_VOICES
    ):

        log(
            f"  {voice_name}"
        )

    # --------------------------------------------------------
    # Kokoro
    # --------------------------------------------------------

    log("")
    log("Kokoro voices:")

    for voice_name in sorted(
        KOKORO_VOICES
    ):

        voice_path = (
            KOKORO_VOICE_FILES[
                voice_name
            ]
        )

        exists = (
            "OK"
            if voice_path.exists()
            else "MISSING"
        )

        log(
            f"  {voice_name}: "
            f"{voice_path} [{exists}]"
        )

    log("")
    log("Kokoro ONNX:")

    for path in [
        KOKORO_ONNX_BASE,
        KOKORO_ONNX_DIMA,
        KOKORO_CONFIG_PATH,
        KOKORO_G2P_PATH,
    ]:

        exists = (
            "OK"
            if path.exists()
            else "MISSING"
        )

        log(
            f"  {path} [{exists}]"
        )

    # --------------------------------------------------------
    # Endpoint
    # --------------------------------------------------------

    log("")
    log("Endpoint:")
    log(
        f"  POST "
        f"http://{HOST}:{PORT}"
        f"/v1/audio/speech"
    )

    log("")
    log("Models:")
    log("  piper")
    log("  silero")
    log("  kokoro")

    log("")
    log("Server starting...")
    log("=" * 60)

    server = ThreadingHTTPServer(
        (HOST, PORT),
        TTSHandler,
    )

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        log("")
        log("Stopping server...")

    finally:

        server.server_close()


if __name__ == "__main__":
    main()
