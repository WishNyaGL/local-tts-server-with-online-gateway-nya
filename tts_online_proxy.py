import base64
import io
import json
import os
import shlex
import threading
import urllib.error
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = os.environ.get("TTS_PROXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("TTS_PROXY_PORT", "8890"))
LOCAL_TTS_URL = os.environ.get(
    "TTS_LOCAL_URL",
    "http://127.0.0.1:8880/v1/audio/speech",
)
LOCAL_API_KEY = os.environ.get("TTS_API_KEY", "tts-local")
ONLINE_API_KEY = os.environ.get("TTS_ONLINE_API_KEY", "")
CONFIG_FILE = Path(__file__).resolve().parent / "tts_gateway_config.json"

# Debug logging. Keep disabled during normal use.
DEBUG_LOG_REQUEST = True
DEBUG_LOG_RESPONSE = True

# The proxy itself is intentionally separate from tts_server.py.
# tts_server.py remains untouched.
CONFIG = {
    "online_enabled": False,
    "online_api_key": ONLINE_API_KEY,
    "online_test_model": "3fc339d7c6474b11a221708e2a7b1c4b",
    # If an incoming local request has an unknown model/voice, use these.
    "local_model": "kokoro",
    "local_voice": "sveta",
    "test_text": "Здравствуйте! Это тест онлайн TTS.",
    "rewrite_rules": [],
    "presets": [],
    "selected_preset_id": "",
    "selected_voice_id": "",
    "curl": r"""curl --request POST \
--url 'https://api.fish.audio/v1/tts' \
--header 'Authorization: Bearer $API_KEY' \
--header 'Content-Type: application/json' \
--header 'model: s2.1-pro-free' \
--data '{
  "text": "${INPUT}",
  "reference_id": "${VOICE}",
  "temperature": 0.7,
  "top_p": 0.7,
  "prosody": {
    "speed": 1,
    "volume": 0,
    "normalize_loudness": true
  },
  "chunk_length": 300,
  "normalize": true,
  "format": "wav",
  "sample_rate": 44100,
  "latency": "normal",
  "max_new_tokens": 1024,
  "repetition_penalty": 1.2,
  "min_chunk_length": 50,
  "condition_on_previous_chunks": true,
  "early_stop_threshold": 1
}'""",
}

CONFIG_LOCK = threading.Lock()


def load_saved_config():
    if not CONFIG_FILE.exists():
        return

    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            saved = json.load(f)
        if not isinstance(saved, dict):
            raise ValueError("config root must be an object")

        with CONFIG_LOCK:
            for key in (
                "online_enabled",
                "online_api_key",
                "online_test_model",
                "local_model",
                "local_voice",
                "test_text",
                "rewrite_rules",
                "presets",
                "selected_preset_id",
                "selected_voice_id",
                "curl",
            ):
                if key in saved:
                    CONFIG[key] = saved[key]
            if not CONFIG.get("presets"):
                legacy_curl = CONFIG.get("curl", "")
                legacy_voice = str(CONFIG.get("online_test_model", ""))
                CONFIG["presets"] = [{
                    "id": "preset-default",
                    "name": "",
                    "api_key": str(CONFIG.get("online_api_key", "")),
                    "curl": legacy_curl,
                    "voices": ([{"id": "voice-default", "name": "", "value": legacy_voice}] if legacy_voice else [])
                }]
                CONFIG["selected_preset_id"] = "preset-default"
                CONFIG["selected_voice_id"] = "voice-default" if legacy_voice else ""
        log(f"[CONFIG] loaded: {CONFIG_FILE}")
    except Exception as exc:
        log(f"[CONFIG] cannot load {CONFIG_FILE}: {exc}; using defaults")


def save_config():
    with CONFIG_LOCK:
        data = dict(CONFIG)

    tmp = CONFIG_FILE.with_suffix(CONFIG_FILE.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, CONFIG_FILE)
    log(f"[CONFIG] saved: {CONFIG_FILE}")


def log(message):
    print(message, flush=True)


def parse_curl(curl_text):
    """
    Parse the practical curl format used by API documentation.
    Supports:
      curl URL
      -X/--request METHOD
      -H/--header HEADER
      -d/--data DATA
      --location
    """
    normalized = curl_text.replace("\\\r\n", " ").replace("\\\n", " ")
    try:
        tokens = shlex.split(normalized, posix=True)
    except ValueError as exc:
        raise ValueError(f"Cannot parse CURL: {exc}")

    if not tokens or tokens[0].lower() != "curl":
        raise ValueError("CURL must start with 'curl'")

    method = "POST"
    url = None
    headers = {}
    data = None

    i = 1
    while i < len(tokens):
        token = tokens[i]

        if token in ("--location", "--compressed", "--silent", "-s", "--globoff"):
            i += 1
            continue

        if token in ("-X", "--request"):
            if i + 1 >= len(tokens):
                raise ValueError(f"Missing value after {token}")
            method = tokens[i + 1].upper()
            i += 2
            continue

        if token in ("-H", "--header"):
            if i + 1 >= len(tokens):
                raise ValueError("Missing header value")
            header = tokens[i + 1]
            if ":" not in header:
                raise ValueError(f"Invalid header: {header}")
            name, value = header.split(":", 1)
            headers[name.strip()] = value.strip()
            i += 2
            continue

        if token in ("-d", "--data", "--data-raw", "--data-binary", "--data-ascii"):
            if i + 1 >= len(tokens):
                raise ValueError(f"Missing value after {token}")
            data = tokens[i + 1]
            if method == "GET":
                method = "POST"
            i += 2
            continue

        if token.startswith("-"):
            # Keep this deliberately conservative: unknown curl switches
            # are ignored rather than silently changing request semantics.
            i += 1
            continue

        if url is None:
            url = token

        i += 1

    if not url:
        raise ValueError("CURL URL not found")

    return method, url, headers, data


def substitute_string(value, text, model="", current_value="", field=""):
    with CONFIG_LOCK:
        api_key = CONFIG.get("online_api_key", "")

    if not api_key:
        api_key = os.environ.get("TTS_ONLINE_API_KEY", ONLINE_API_KEY)

    replacements = {
        "${INPUT}": text,
        "$INPUT": text,
        "${TEXT}": text,
        "$TEXT": text,
        "{{INPUT}}": text,
        "${VOICE}": model,
        "$VOICE": model,
        "${MODEL}": model,
        "$MODEL": model,
        "${VALUE}": current_value,
        "$VALUE": current_value,
        "${FIELD}": field,
        "$FIELD": field,
        "${API_KEY}": api_key,
        "$API_KEY": api_key,
        "API_KEY": api_key,
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    return value


def apply_rewrite_rules(request):
    """Rename JSON fields and optionally replace their values."""
    request = dict(request)

    with CONFIG_LOCK:
        rules = list(CONFIG.get("rewrite_rules", []))

    if not isinstance(rules, list):
        return request

    for rule in rules:
        if not isinstance(rule, dict):
            continue

        # New schema: field_from -> field_to; value_from -> value_to.
        # Also read the old schema so existing config files keep working.
        field_from = str(rule.get("field_from", rule.get("field", ""))).strip()
        if not field_from:
            # Without a source parameter, the entire rule is inactive.
            continue
        if field_from not in request:
            continue

        field_to = str(rule.get("field_to", "")).strip() or field_from
        current = request[field_from]
        current_text = "" if current is None else str(current)
        value_from = str(rule.get("value_from", rule.get("match", "")))
        value_to = str(rule.get("value_to", rule.get("replace", "")))

        # Empty value_from matches any incoming value.
        if value_from and current_text != value_from:
            continue

        # Empty value_to means keep the incoming value (parameter rename only).
        if value_to:
            new_value = substitute_string(
                value_to,
                str(request.get("input", "")),
                str(request.get("model", request.get("voice", ""))),
                current_text,
                field_from,
            )
        else:
            new_value = current

        if field_to != field_from:
            del request[field_from]
        request[field_to] = new_value
        log(
            f"[REWRITE] {field_from!r} -> {field_to!r}; "
            f"value {current_text!r} -> {new_value!r}"
        )

    return request


def substitute_json(value, text, model=""):
    if isinstance(value, str):
        return substitute_string(value, text, model)
    if isinstance(value, list):
        return [substitute_json(v, text, model) for v in value]
    if isinstance(value, dict):
        return {k: substitute_json(v, text, model) for k, v in value.items()}
    return value


def build_online_request(text, model=""):
    with CONFIG_LOCK:
        curl_text = CONFIG["curl"]

    method, url, headers, data = parse_curl(curl_text)

    headers = {
        k: substitute_string(v, text, model)
        for k, v in headers.items()
    }

    if data is not None:
        content_type = ""
        for k, v in headers.items():
            if k.lower() == "content-type":
                content_type = v.lower()

        if "application/json" in content_type:
            try:
                payload = json.loads(data)
                payload = substitute_json(payload, text, model)
                if isinstance(payload, dict) and "text" in payload:
                    payload["text"] = text
                data = json.dumps(
                    payload,
                    ensure_ascii=False,
                ).encode("utf-8")

                headers["Content-Type"] = "application/json"

            except json.JSONDecodeError:
                data = data.encode("utf-8")
        else:
            data = substitute_string(data, text, model).encode("utf-8")
    else:
        data = None

    if not any(k.lower() == "authorization" for k in headers):
        with CONFIG_LOCK:
            api_key = CONFIG.get("online_api_key", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    return method, url, headers, data


def call_url(method, url, headers, data, timeout=60):
    request = urllib.request.Request(
        url=url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(
            f"Online provider HTTP {exc.code}: "
            f"{body.decode('utf-8', errors='replace')[:1000]}"
        )
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Online provider connection error: {exc}")


def pcm_to_wav(pcm_data, sample_rate):
    output = io.BytesIO()

    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data)

    return output.getvalue()

def normalize_wav(data):
    import io
    import wave

    source = io.BytesIO(data)

    with wave.open(source, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    output = io.BytesIO()

    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.setcomptype("NONE", "not compressed")
        wav.writeframes(frames)

    return output.getvalue()

def call_online(text, model=""):
    method, url, headers, data = build_online_request(text, model)

    log(f"[ONLINE] {method} {url}")

    status, response_headers, response_body = call_url(
        method, url, headers, data
    )

    content_type = response_headers.get("Content-Type", "").lower()

    # Some TTS APIs return the audio bytes directly.
    # WAV output is already compatible with this gateway.
    if (
        "audio/wav" in content_type
        or "audio/x-wav" in content_type
        or response_body[:4] == b"RIFF"
    ):
        return response_body

    # Some providers may return
    # JSON containing base64 audioContent. Keep that path working too.
    try:
        result = json.loads(response_body.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Online provider returned unsupported audio format "
            f"(Content-Type={content_type}). Use format=wav for providers that return WAV directly."
        ) from exc

    audio_b64 = result.get("audioContent")

    if not audio_b64:
        audio_b64 = (
            result.get("result", {}).get("audioContent")
            if isinstance(result.get("result"), dict)
            else None
        )

    if not audio_b64:
        raise RuntimeError(
            "Online provider response does not contain audioContent"
        )

    audio_bytes = base64.b64decode(audio_b64)

    with CONFIG_LOCK:
        curl_text = CONFIG["curl"]

    _, _, _, raw_data = parse_curl(curl_text)
    sample_rate = 22050

    if raw_data:
        try:
            body = json.loads(substitute_string(raw_data, text, model))
            audio_config = body.get("audioConfig", {})
            sample_rate = int(
                audio_config.get("sampleRateHertz", sample_rate)
            )
            encoding = str(
                audio_config.get("audioEncoding", "LINEAR16")
            ).upper()
        except Exception:
            encoding = "LINEAR16"
    else:
        encoding = "LINEAR16"

    if encoding in ("LINEAR16", "PCM", "PCM16"):
        return pcm_to_wav(audio_bytes, sample_rate)

    if encoding == "WAV":
        return audio_bytes

    raise RuntimeError(
        f"Online audio encoding '{encoding}' is not directly compatible "
        f"with the local audio/wav endpoint. Use WAV or LINEAR16/PCM."
    )


LOCAL_MODELS = {
    "piper": {"ru_RU-irina-medium", "en_US-libritts-high"},
    "silero": {"aidar", "baya", "kseniya", "xenia", "eugene"},
    "kokoro": {"sveta", "masha", "dima"},
}

ONLINE_MODEL_ALIASES = {
    # Example:
    # "voice_name_from_game": "provider_voice_or_reference_id",
}

LOCAL_MODEL_ALIASES = {
    "tts-1": "silero",
    "tts-1-hd": "silero",
}

LOCAL_VOICE_ALIASES = {
    "alloy": "kseniya",
    "echo": "aidar",
    "fable": "baya",
    "onyx": "eugene",
    "nova": "xenia",
    "shimmer": "kseniya",
    "ru-irina-medium": "baya",
}

def normalize_local_request(request):
    request = dict(request)
    model = request.get("model", "")
    voice = request.get("voice")

    # send the selected voice as tts-model instead of voice.
    tts_model = request.get("tts-model", request.get("tts_model"))
    if not voice and tts_model:
        voice = tts_model

    canonical_model = LOCAL_MODEL_ALIASES.get(model, model)
    canonical_voice = LOCAL_VOICE_ALIASES.get(voice, voice) if voice else voice

    # put the selected voice into tts-model instead of voice.
    if (not model or canonical_model not in LOCAL_MODELS) and tts_model:
        tts_value = LOCAL_VOICE_ALIASES.get(tts_model, tts_model)
        for candidate_model, voices in LOCAL_MODELS.items():
            if tts_value in voices:
                canonical_model = candidate_model
                canonical_voice = tts_value
                break

    valid_model = canonical_model in LOCAL_MODELS
    valid_voice = (
        canonical_model in LOCAL_MODELS
        and canonical_voice in LOCAL_MODELS[canonical_model]
    )

    if not valid_model or not valid_voice:
        with CONFIG_LOCK:
            fallback_model = CONFIG.get("local_model", "kokoro")
            fallback_voice = CONFIG.get("local_voice", "sveta")

        if fallback_model not in LOCAL_MODELS:
            raise ValueError(
                f"Invalid local fallback model: {fallback_model}"
            )
        if fallback_voice not in LOCAL_MODELS[fallback_model]:
            raise ValueError(
                f"Invalid local fallback voice '{fallback_voice}' "
                f"for model '{fallback_model}'"
            )

        request["model"] = fallback_model
        request["voice"] = fallback_voice
        log(
            f"[LOCAL FALLBACK] invalid model/voice "
            f"({model!r}, {voice!r}) -> "
            f"{fallback_model}/{fallback_voice}"
        )
        return request

    return request


def get_online_model(request):
    value = request.get("tts-model", request.get("tts_model"))
    if value is None:
        value = request.get("voice", "")
    value = str(value or "")
    return ONLINE_MODEL_ALIASES.get(value, value)


def build_local_body(request):
    return json.dumps(
        normalize_local_request(request),
        ensure_ascii=False,
    ).encode("utf-8")

def call_local(raw_body, headers):
    local_headers = {
        "Content-Type": headers.get("Content-Type", "application/json"),
        "Authorization": f"Bearer {LOCAL_API_KEY}",
    }

    status, response_headers, response_body = call_url(
        "POST",
        LOCAL_TTS_URL,
        local_headers,
        raw_body,
        timeout=120,
    )

    if status != 200:
        raise RuntimeError(
            f"Local TTS returned HTTP {status}: "
            f"{response_body.decode('utf-8', errors='replace')[:1000]}"
        )

    return response_body


HTML = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TTS Gateway</title>
<style>
:root{color-scheme:dark;--bg:#15171a;--panel:#202328;--panel2:#181a1e;--line:#353941;--muted:#a1a7b2;--text:#eef0f4;--blue:#3978e8;--green:#73d99a;--yellow:#ffd166;--red:#ff6868}*{box-sizing:border-box}body{font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:20px}.wrap{max-width:1050px;margin:0 auto}header{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:14px}h1{font-size:24px;margin:0}h2{font-size:17px;margin:0 0 12px}h3{font-size:14px;margin:14px 0 8px}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px;margin-bottom:12px}label{display:block;margin:8px 0 5px;color:#c0c5ce;font-size:13px}input:not([type=checkbox]),textarea,select{display:block;width:100%;background:var(--panel2);color:var(--text);border:1px solid #454a54;border-radius:7px;padding:9px 10px;font:13px Consolas,'Courier New',monospace;outline:none}textarea{min-height:190px;resize:vertical;line-height:1.45}button{background:var(--blue);color:white;border:0;border-radius:7px;padding:9px 12px;font-weight:600;cursor:pointer}button:hover{filter:brightness(1.12)}button:disabled{opacity:.5;cursor:not-allowed}.secondary{background:#414650}.danger{background:#7e3038}.small{color:var(--muted);font-size:12px;line-height:1.5}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.grow{flex:1;min-width:200px}.top-actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.mode{display:flex;align-items:center;gap:8px;padding:8px 10px;background:var(--panel2);border:1px solid var(--line);border-radius:8px}.mode input{width:17px;height:17px}.mode label{margin:0;color:var(--text)}.warning{border:1px solid #ff4141;background:#401b20;color:#ffb8b8;border-radius:9px;padding:12px 14px;margin-bottom:12px;font-weight:700;line-height:1.45}.presetbar{display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:8px;align-items:end}.cascade{position:relative}.cascade-button{width:100%;text-align:left;background:var(--panel2);border:1px solid #454a54;padding:10px 12px}.menu{position:absolute;top:calc(100% + 5px);left:0;z-index:20;width:min(340px,85vw);max-height:360px;overflow:auto;background:#24272d;border:1px solid #505560;border-radius:9px;box-shadow:0 12px 30px #0009;padding:6px}.menu.hidden{display:none}.provider{padding:9px;border-radius:6px;font-weight:700;cursor:pointer}.provider:hover,.voice-option:hover{background:#343943}.voice-option{display:block;width:100%;text-align:left;background:transparent;font-weight:400;padding:8px 12px 8px 22px}.selected{color:#9fc0ff}.voice-row{display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:8px;align-items:center;margin-top:12px}.test-grid{display:grid;grid-template-columns:minmax(220px,.8fr) minmax(300px,1.5fr);gap:14px}.test-text textarea{min-height:104px}.test-actions{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:10px}.status{padding:9px 11px;border-radius:7px;background:var(--panel2);border:1px solid var(--line);font-size:13px;margin-top:10px}.ok{color:var(--green)}.warn{color:var(--yellow)}.error{color:var(--red)}.loading{display:inline-flex;align-items:center;gap:8px;color:var(--yellow);font-size:13px}.spinner{width:15px;height:15px;border:2px solid #ffffff35;border-top-color:var(--yellow);border-radius:50%;animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}.result{margin-top:12px;padding:12px;background:var(--panel2);border:1px solid var(--line);border-radius:8px}.result audio{width:100%;height:38px;margin-top:8px}.hidden{display:none!important}.rules-head,.rule{display:grid;grid-template-columns:minmax(105px,.8fr) minmax(120px,.9fr) minmax(125px,1fr) minmax(125px,1fr) auto;gap:7px;align-items:center}.rules-head{color:var(--muted);font-size:11px;margin:12px 0 6px}.rule{margin-bottom:7px}.subsection{padding-top:12px;margin-top:12px;border-top:1px solid var(--line)}.config-grid{display:grid;grid-template-columns:minmax(180px,.8fr) minmax(300px,1.5fr);gap:14px}.curl-area{min-height:260px}.modal{border:1px solid #505560;border-radius:12px;background:#22252b;color:var(--text);width:min(650px,calc(100% - 24px));max-height:90vh;overflow:auto;padding:18px;box-shadow:0 20px 60px #000b}.modal::backdrop{background:#000a}.modal h2{margin-bottom:12px}.modal textarea{min-height:240px}.token{display:inline-flex;align-items:center;gap:6px;background:#343943;border-radius:6px;padding:6px 8px;margin:4px 4px 0 0;font:12px Consolas,monospace;cursor:pointer}.modal-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:14px}.field-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media(max-width:700px){body{padding:11px}header{align-items:flex-start;flex-direction:column}.test-grid,.config-grid,.field-grid{grid-template-columns:1fr}.presetbar{grid-template-columns:1fr auto}.presetbar .full{grid-column:1/-1}.rules-head{display:none}.rule{grid-template-columns:1fr}.voice-row{grid-template-columns:1fr auto auto}}
</style></head><body><div class="wrap">
<header><div><h1>TTS Gateway</h1><div class="small">Единая точка входа для локальных и онлайн TTS-моделей</div></div><div class="top-actions"><div class="mode"><input id="enabled" type="checkbox"><label for="enabled">Использовать ONLINE</label></div><button onclick="save()">Сохранить настройки</button></div></header>
<div class="warning">⚠ НЕ ДЕЛИТЕСЬ ФАЙЛОМ КОНФИГУРАЦИИ! В tts_gateway_config.json хранятся API-ключи и другие секретные данные. Не публикуйте и не отправляйте этот файл другим людям.</div>
<div class="card"><h2>Пресет и голос</h2><div class="presetbar"><div class="full"><label>Провайдер / пресет → голос (нажмите, чтобы открыть меню)</label><div class="cascade"><button id="cascadeButton" class="cascade-button" onclick="toggleMenu()">Загрузка пресетов… ▾</button><div id="cascadeMenu" class="menu hidden"></div></div></div><button class="secondary" title="Редактировать выбранный пресет" onclick="editPreset()">✎ Пресет</button><button title="Добавить пресет" onclick="newPreset()">＋</button></div>
<div class="voice-row"><div><label for="voiceDisplay">Выбранный голос</label><input id="voiceDisplay" readonly placeholder="Выберите пресет и голос"><div id="voiceValueHelp" class="small">Реальное значение ID отправляется в API; отображаемое имя — только для удобства.</div></div><button class="secondary" onclick="editVoice()">✎ Голос</button><button onclick="newVoice()">＋ Голос</button></div>
<div class="small" style="margin-top:10px">Имя провайдера определяется по URL CURL (Fish Audio, OpenAI, Google и т. п.). При необходимости название можно задать вручную в настройках пресета. Меню раскрывается кликом.</div><div id="presetStatus" class="status">Загрузка конфигурации…</div></div>
<div class="card"><h2>Проверка голоса</h2><div class="test-grid"><div><label>Активный голос</label><div id="activeVoiceSummary" class="status">—</div><div class="small">Тест использует выбранный голос и CURL из активного пресета.</div></div><div class="test-text"><label for="text">Текст для озвучивания</label><textarea id="text" rows="3" style="min-height:104px">Здравствуйте! Это тест онлайн TTS.</textarea><div class="test-actions"><button id="testButton" onclick="test()">▶ Тестировать голос</button><span id="loading" class="loading hidden"><span class="spinner"></span><span>Генерация… <b id="elapsed">0.0 с</b></span></span><span class="small">Настройки сохраняются перед тестом.</span></div></div></div><div id="testStatus" class="status">Готово к тестированию.</div><div id="audioResult" class="result hidden"><b>Последний результат</b><div id="audioMeta" class="small"></div><audio id="audioPlayer" controls preload="metadata"></audio></div></div>
<div class="card"><h2>Настройки активного пресета</h2><div class="config-grid"><div><label for="api_key">API-ключ (общий для всех голосов этого пресета)</label><input id="api_key" type="password" autocomplete="off" placeholder="Ключ внешнего сервиса"><div class="small">В CURL используй <b>${API_KEY}</b>. Ключ хранится в локальном JSON.</div><label for="presetName">Название пресета (необязательно)</label><input id="presetName" placeholder="Автоопределение по URL CURL"><div class="row" style="margin-top:8px"><button class="secondary" onclick="editPreset()">✎ Изменить пресет</button><button class="danger" onclick="deletePreset()">Удалить пресет</button></div><div id="status" class="status">—</div></div><div><div class="row" style="justify-content:space-between"><label for="curl">CURL-запрос</label><button class="secondary" onclick="openCurlEditor()">✎ Редактировать CURL</button></div><textarea id="curl" class="curl-area" spellcheck="false" placeholder="curl --request POST ..."></textarea><div class="small">Переменные: <b>${API_KEY}</b> — API-ключ; <b>${VOICE}</b> — значение выбранного голоса; <b>${TEXT}</b> или <b>${INPUT}</b> — текст. Если в JSON есть поле text, шлюз автоматически подставит текст запроса.</div></div></div>
<div class="subsection"><h3>Правила подмен входящих параметров</h3><div class="small">Исходный параметр → новый параметр; исходное значение → новое значение. Пустой исходный параметр отключает правило.</div><div class="rules-head"><div>Исходный параметр</div><div>Новый параметр</div><div>Исходное значение</div><div>Новое значение</div><div></div></div><div id="rules"></div><div class="row" style="margin-top:9px"><button class="secondary" onclick="addRule()">＋ Добавить правило</button></div></div></div>
<div class="card"><h2>LOCAL fallback</h2><div class="row"><div class="grow"><label for="local_model">Локальная модель</label><input id="local_model" type="text" placeholder="kokoro"></div><div class="grow"><label for="local_voice">Локальный голос</label><input id="local_voice" type="text" placeholder="sveta"></div></div><div class="small">Используется в LOCAL-режиме, если входные model/voice не распознаны.</div></div>
</div>
<dialog id="presetDialog" class="modal"><h2 id="presetDialogTitle">Пресет</h2><div class="field-grid"><div><label for="dlgPresetName">Название пресета (необязательно)</label><input id="dlgPresetName" placeholder="Например, Fish Audio"></div><div><label for="dlgPresetKey">API-ключ</label><input id="dlgPresetKey" type="password" autocomplete="off" placeholder="Ключ для этого провайдера"></div></div><label for="dlgPresetCurl">CURL-запрос</label><textarea id="dlgPresetCurl" spellcheck="false"></textarea><div class="small">Быстро вставить переменную (нажмите на неё):<br><span class="token" onclick="insertToken('${API_KEY}','dlgPresetCurl')">${API_KEY}</span><span class="token" onclick="insertToken('${VOICE}','dlgPresetCurl')">${VOICE}</span><span class="token" onclick="insertToken('${TEXT}','dlgPresetCurl')">${TEXT}</span><span class="token" onclick="insertToken('${INPUT}','dlgPresetCurl')">${INPUT} (совместимость)</span></div><div class="modal-actions"><button class="secondary" onclick="closeDialog('presetDialog')">Отмена</button><button onclick="commitPresetDialog()">Сохранить</button></div></dialog>
<dialog id="voiceDialog" class="modal"><h2 id="voiceDialogTitle">Голос</h2><label for="dlgVoiceName">Название голоса (можно оставить пустым)</label><input id="dlgVoiceName" placeholder="Например, Masha / Спокойный голос"><label for="dlgVoiceValue">Значение / ID голоса</label><input id="dlgVoiceValue" placeholder="Фактический ID, reference_id или voice name"><div class="small">Если название пустое, в меню будет показано само значение ID.</div><div class="modal-actions"><button class="secondary" onclick="closeDialog('voiceDialog')">Отмена</button><button onclick="commitVoiceDialog()">Сохранить</button></div></dialog>
<dialog id="curlDialog" class="modal"><h2>Редактирование CURL</h2><textarea id="dlgCurl" spellcheck="false"></textarea><div class="small">Переменные для копирования/вставки:<br><span class="token" onclick="insertToken('${API_KEY}','dlgCurl')">${API_KEY}</span><span class="token" onclick="insertToken('${VOICE}','dlgCurl')">${VOICE}</span><span class="token" onclick="insertToken('${TEXT}','dlgCurl')">${TEXT}</span><span class="token" onclick="insertToken('${INPUT}','dlgCurl')">${INPUT} (совместимость)</span></div><div class="modal-actions"><button class="secondary" onclick="closeDialog('curlDialog')">Отмена</button><button onclick="commitCurlDialog()">Применить CURL</button></div></dialog>
<script>
let cfg={}, presets=[], activePresetId='', activeVoiceId='', presetDialogMode='new', voiceDialogMode='new', editingVoiceId='';let audioObjectUrl=null,timerHandle=null,testStartedAt=0;
const $=id=>document.getElementById(id);function esc(v){return String(v??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;')}
function activePreset(){return presets.find(p=>p.id===activePresetId)||presets[0]||null}function activeVoice(){const p=activePreset();return p?.voices?.find(v=>v.id===activeVoiceId)||p?.voices?.[0]||null}
function providerName(p){if(p?.name?.trim())return p.name.trim();const m=String(p?.curl||'').match(/https?:\/\/([^\s'"\\]+)/i);if(!m)return 'Провайдер без названия';const host=m[1].split('/')[0].split(':')[0].toLowerCase();if(host.includes('fish.audio'))return 'Fish Audio';if(host.includes('openai.com'))return 'OpenAI';if(host.includes('googleapis.com')||host.includes('generativelanguage'))return 'Google';if(host.includes('elevenlabs'))return 'ElevenLabs';if(host.includes('azure'))return 'Azure';if(host.includes('openrouter'))return 'OpenRouter';return host.replace(/^api\./,'')||'Провайдер'}
function voiceLabel(v){return v?.name?.trim()||v?.value||'(голос без ID)'}function newId(prefix){return prefix+'-'+Date.now().toString(36)+'-'+Math.random().toString(36).slice(2,7)}
function setPresetStatus(msg,kind=''){ $('presetStatus').className='status '+kind;$('presetStatus').textContent=msg }
function toggleMenu(){const menu=$('cascadeMenu');menu.classList.toggle('hidden');if(!menu.classList.contains('hidden'))renderMenu()}
function renderMenu(){const menu=$('cascadeMenu');menu.innerHTML='';if(!presets.length){menu.innerHTML='<div class="small" style="padding:10px">Пресетов нет. Нажмите +, чтобы создать.</div>';return}presets.forEach(p=>{const group=document.createElement('div');const title=document.createElement('div');title.className='provider';title.textContent=(p.id===activePresetId?'✓ ':'')+providerName(p)+' ▾';title.onclick=()=>{if(activePresetId===p.id){group.querySelector('.voice-list').classList.toggle('hidden')}else{selectPreset(p.id);renderMenu()}};group.appendChild(title);const list=document.createElement('div');list.className='voice-list'+(p.id===activePresetId?'':' hidden');(p.voices||[]).forEach(v=>{const b=document.createElement('button');b.className='voice-option'+(p.id===activePresetId&&v.id===activeVoiceId?' selected':'');b.textContent=(p.id===activePresetId&&v.id===activeVoiceId?'✓ ':'')+voiceLabel(v);b.onclick=()=>{selectPreset(p.id,v.id);menu.classList.add('hidden')};list.appendChild(b)});const add=document.createElement('button');add.className='voice-option';add.textContent='＋ Добавить голос…';add.onclick=()=>{selectPreset(p.id);menu.classList.add('hidden');newVoice()};list.appendChild(add);group.appendChild(list);menu.appendChild(group)})}
function selectPreset(pid,vid){activePresetId=pid;const p=activePreset();if(!p)return;activeVoiceId=vid||p.voices?.[0]?.id||'';fillActiveFields();renderMenu();$('cascadeButton').textContent=providerName(p)+' → '+(activeVoice()?voiceLabel(activeVoice()):'голос не добавлен')+' ▾';setPresetStatus('Активный пресет: '+providerName(p)+(activeVoice()?' · '+voiceLabel(activeVoice()):' · добавьте голос'),'ok')}
function fillActiveFields(){const p=activePreset(),v=activeVoice();$('api_key').value=p?.api_key||'';$('curl').value=p?.curl||'';$('presetName').value=p?.name||'';$('voiceDisplay').value=v?voiceLabel(v):'';$('voiceValueHelp').textContent=v?'Передаваемое значение: '+v.value:'Добавьте голос с его фактическим ID.';$('activeVoiceSummary').textContent=v?`${voiceLabel(v)} · ID: ${v.value}`:'Голос не выбран';}
function ruleRow(rule={}){const row=document.createElement('div');row.className='rule';row.innerHTML=`<input aria-label="Исходный параметр" placeholder="model" value="${esc(rule.field_from??rule.field??'')}"><input aria-label="Новый параметр" placeholder="voice (пусто = оставить)" value="${esc(rule.field_to||'')}"><input aria-label="Исходное значение" placeholder="пусто = любое" value="${esc(rule.value_from??rule.match??'')}"><input aria-label="Новое значение" placeholder="пусто = оставить" value="${esc(rule.value_to??rule.replace??'')}"><button class="danger" type="button" onclick="this.parentElement.remove()">Удалить</button>`;$('rules').appendChild(row)}function addRule(){ruleRow({})}function getRules(){return [...document.querySelectorAll('#rules .rule')].map(r=>{const a=r.querySelectorAll('input');return{field_from:a[0].value.trim(),field_to:a[1].value.trim(),value_from:a[2].value,value_to:a[3].value}}).filter(x=>x.field_from)}
function syncCurrentToPreset(){const p=activePreset();if(!p)return;p.api_key=$('api_key').value;p.curl=$('curl').value;p.name=$('presetName').value.trim()}
async function load(){try{const r=await fetch('/api/config');cfg=await r.json();if(!r.ok)throw Error(cfg.error||'Не удалось загрузить конфигурацию');presets=Array.isArray(cfg.presets)?cfg.presets:[];if(!presets.length){presets=[{id:'preset-default',name:'',api_key:cfg.online_api_key||'',curl:cfg.curl||'',voices:cfg.online_test_model?[{id:'voice-default',name:'',value:cfg.online_test_model}]:[]}]}presets=presets.map(p=>({...p,id:p.id||newId('preset'),name:p.name||'',api_key:p.api_key||'',curl:p.curl||'',voices:Array.isArray(p.voices)?p.voices.map(v=>({...v,id:v.id||newId('voice'),name:v.name||'',value:String(v.value??'')})):[]}));activePresetId=cfg.selected_preset_id&&presets.some(p=>p.id===cfg.selected_preset_id)?cfg.selected_preset_id:presets[0]?.id||'';const p=activePreset();activeVoiceId=cfg.selected_voice_id&&p?.voices.some(v=>v.id===cfg.selected_voice_id)?cfg.selected_voice_id:p?.voices[0]?.id||'';$('enabled').checked=!!cfg.online_enabled;$('local_model').value=cfg.local_model||'kokoro';$('local_voice').value=cfg.local_voice||'sveta';$('text').value=cfg.test_text||'Здравствуйте! Это тест онлайн TTS.';$('rules').innerHTML='';(cfg.rewrite_rules||[]).forEach(ruleRow);selectPreset(activePresetId,activeVoiceId);updateStatus();}catch(e){setPresetStatus('Не удалось загрузить настройки: '+e.message,'error')}}
function updateStatus(){const p=activePreset();$('status').innerHTML=$('enabled').checked?'<span class="warn">ONLINE включен</span> — запросы отправляются по CURL.':'<span class="ok">LOCAL режим</span> — запросы отправляются в локальный TTS-сервер.';if(p)$('presetName').placeholder='Авто: '+providerName({...p,name:''})}
async function save(){syncCurrentToPreset();const p=activePreset(),v=activeVoice();if(!p){setTestStatus('Сначала создайте пресет.','error');return false}if(!p.curl.trim()){setTestStatus('CURL активного пресета пустой.','error');return false}const body={online_enabled:$('enabled').checked,online_api_key:p.api_key,online_test_model:v?.value||'no-voice-selected',local_model:$('local_model').value.trim(),local_voice:$('local_voice').value.trim(),test_text:$('text').value,rewrite_rules:getRules(),curl:p.curl,presets,selected_preset_id:activePresetId,selected_voice_id:activeVoiceId};try{const r=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const x=await r.json();if(!r.ok)throw Error(x.error||'Ошибка сохранения');cfg=x;updateStatus();setPresetStatus('Настройки сохранены в tts_gateway_config.json','ok');return true}catch(e){setTestStatus('Не удалось сохранить настройки: '+e.message,'error');return false}}
function openDialog(id){$(id).showModal()}function closeDialog(id){$(id).close()}
function newPreset(){presetDialogMode='new';$('presetDialogTitle').textContent='Добавить пресет';$('dlgPresetName').value='';$('dlgPresetKey').value='';$('dlgPresetCurl').value='curl --request POST \\\n--url \'https://example.com/v1/audio/speech\' \\\n--header \'Authorization: Bearer ${API_KEY}\' \\\n--header \'Content-Type: application/json\' \\\n--data \'{"text":"${TEXT}","voice":"${VOICE}"}\'';openDialog('presetDialog')}
function editPreset(){const p=activePreset();if(!p){newPreset();return}syncCurrentToPreset();presetDialogMode='edit';$('presetDialogTitle').textContent='Редактировать пресет';$('dlgPresetName').value=p.name||'';$('dlgPresetKey').value=p.api_key||'';$('dlgPresetCurl').value=p.curl||'';openDialog('presetDialog')}
function commitPresetDialog(){const name=$('dlgPresetName').value.trim(),key=$('dlgPresetKey').value,curl=$('dlgPresetCurl').value;if(!curl.trim()){alert('CURL не может быть пустым');return}if(presetDialogMode==='new'){const p={id:newId('preset'),name,api_key:key,curl,voices:[]};presets.push(p);activePresetId=p.id;activeVoiceId='';}else{const p=activePreset();if(!p)return;p.name=name;p.api_key=key;p.curl=curl}closeDialog('presetDialog');selectPreset(activePresetId,activeVoiceId);setPresetStatus('Пресет изменён. Нажмите «Сохранить настройки», чтобы записать изменения в JSON.','warn')}
function deletePreset(){const p=activePreset();if(!p)return;if(!confirm('Удалить пресет «'+providerName(p)+'» и все его голоса?'))return;presets=presets.filter(x=>x.id!==p.id);activePresetId=presets[0]?.id||'';activeVoiceId=activePreset()?.voices?.[0]?.id||'';if(!presets.length){activePresetId='';activeVoiceId='';newPreset();return}selectPreset(activePresetId,activeVoiceId);save()}
function newVoice(){if(!activePreset()){newPreset();return}voiceDialogMode='new';editingVoiceId='';$('voiceDialogTitle').textContent='Добавить голос';$('dlgVoiceName').value='';$('dlgVoiceValue').value='';openDialog('voiceDialog')}
function editVoice(){const v=activeVoice();if(!v){newVoice();return}voiceDialogMode='edit';editingVoiceId=v.id;$('voiceDialogTitle').textContent='Редактировать голос';$('dlgVoiceName').value=v.name||'';$('dlgVoiceValue').value=v.value||'';openDialog('voiceDialog')}
function commitVoiceDialog(){const p=activePreset();if(!p)return;const value=$('dlgVoiceValue').value.trim(),name=$('dlgVoiceName').value.trim();if(!value){alert('Укажите значение/ID голоса');return}if(voiceDialogMode==='new'){const v={id:newId('voice'),name,value};p.voices=p.voices||[];p.voices.push(v);activeVoiceId=v.id}else{const v=p.voices.find(x=>x.id===editingVoiceId);if(v){v.name=name;v.value=value}}closeDialog('voiceDialog');selectPreset(activePresetId,activeVoiceId);setPresetStatus('Голос обновлён. Нажмите «Сохранить настройки», чтобы записать изменения в JSON.','warn')}
function openCurlEditor(){syncCurrentToPreset();$('dlgCurl').value=$('curl').value;openDialog('curlDialog')}function commitCurlDialog(){$('curl').value=$('dlgCurl').value;syncCurrentToPreset();closeDialog('curlDialog');setPresetStatus('CURL изменён. Нажмите «Сохранить настройки», чтобы записать изменения в JSON.','warn')}
function insertToken(token,targetId){const el=$(targetId),start=el.selectionStart??el.value.length,end=el.selectionEnd??start;el.setRangeText(token,start,end,'end');el.focus()}
function setTestStatus(message,kind=''){$('testStatus').className='status '+kind;$('testStatus').textContent=message}function startTimer(){testStartedAt=performance.now();$('loading').classList.remove('hidden');$('testButton').disabled=true;$('testButton').textContent='Генерация…';const tick=()=>$('elapsed').textContent=((performance.now()-testStartedAt)/1000).toFixed(1)+' с';tick();timerHandle=setInterval(tick,100)}function stopTimer(){if(timerHandle!==null){clearInterval(timerHandle);timerHandle=null}$('loading').classList.add('hidden');$('testButton').disabled=false;$('testButton').textContent='▶ Тестировать голос'}
async function test(){if($('testButton').disabled)return;if(!activeVoice()){setTestStatus('Добавьте и выберите голос перед тестированием.','error');return}$('audioResult').classList.add('hidden');const player=$('audioPlayer');player.pause();player.removeAttribute('src');player.load();if(audioObjectUrl){URL.revokeObjectURL(audioObjectUrl);audioObjectUrl=null}if(!await save())return;const text=$('text').value;startTimer();setTestStatus('Отправлен запрос. Ожидание ответа от TTS…','warn');try{const r=await fetch('/api/test',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input:text})});if(!r.ok){let m=await r.text();try{const j=JSON.parse(m);m=j.error||m}catch{}throw Error(m||('HTTP '+r.status))}const blob=await r.blob();if(!blob.size)throw Error('Сервер вернул пустой аудиофайл');audioObjectUrl=URL.createObjectURL(blob);player.src=audioObjectUrl;$('audioMeta').textContent=`Получено ${(blob.size/1024).toFixed(1)} КБ · время генерации ${((performance.now()-testStartedAt)/1000).toFixed(2)} с`;$('audioResult').classList.remove('hidden');setTestStatus('Аудио получено. Можно прослушать результат ниже.','ok');try{await player.play()}catch{}}catch(e){setTestStatus('Ошибка генерации: '+e.message,'error')}finally{stopTimer()}}
document.addEventListener('click',e=>{if(!e.target.closest('.cascade'))$('cascadeMenu').classList.add('hidden')});$('enabled').addEventListener('change',updateStatus);load();window.addEventListener('beforeunload',()=>{if(audioObjectUrl)URL.revokeObjectURL(audioObjectUrl)});
</script></body></html>
"""


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "TTS-Gateway/1.0"

    def log_message(self, fmt, *args):
        log("[GATEWAY] " + fmt % args)

    def send_json(self, status, data):
        body = json.dumps(
            data, ensure_ascii=False, indent=2
        ).encode("utf-8")

        if DEBUG_LOG_RESPONSE:
            log("[DEBUG RESPONSE] HTTP %s" % status)
            # Never write API secrets to console logs, even when response debugging is enabled.
            try:
                safe_data = json.loads(body.decode("utf-8"))
                if isinstance(safe_data, dict):
                    if safe_data.get("online_api_key"):
                        safe_data["online_api_key"] = "[REDACTED]"
                    if isinstance(safe_data.get("presets"), list):
                        for preset in safe_data["presets"]:
                            if isinstance(preset, dict) and preset.get("api_key"):
                                preset["api_key"] = "[REDACTED]"
                log(json.dumps(safe_data, ensure_ascii=False, indent=2))
            except Exception:
                log(body.decode("utf-8", errors="replace"))

        self.send_response(status)
        self.send_header(
            "Content-Type", "application/json; charset=utf-8"
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_wav(self, data):
        data = normalize_wav(data)

        try:
            import io
            import wave

            with wave.open(io.BytesIO(data), "rb") as wav:
                channels = wav.getnchannels()
                sample_rate = wav.getframerate()
                sample_width = wav.getsampwidth()
                frame_count = wav.getnframes()

                frames = wav.readframes(frame_count)
                expected = frame_count * channels * sample_width
                actual = len(frames)

                log(
                    f"[WAV DEBUG] file={len(data)} bytes, "
                    f"channels={channels}, "
                    f"rate={sample_rate}, "
                    f"width={sample_width}, "
                    f"frames={frame_count}, "
                    f"pcm={actual}/{expected}"
                )

        except Exception as e:
            log(f"[WAV DEBUG] INVALID WAV: {type(e).__name__}: {e}")

        if DEBUG_LOG_RESPONSE:
            log("[DEBUG RESPONSE] HTTP 200")
            log("[DEBUG RESPONSE] Content-Type: audio/wav")
            log(f"[DEBUG RESPONSE] Body: <binary WAV, {len(data)} bytes>")

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("Empty request body")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        if self.path in ("/", "/ui"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/config":
            with CONFIG_LOCK:
                data = dict(CONFIG)
            self.send_json(200, data)
            return

        if self.path == "/health":
            with CONFIG_LOCK:
                online = CONFIG["online_enabled"]
            self.send_json(
                200,
                {
                    "status": "ok",
                    "api_version": 1,
                    "online_enabled": online,
                    "local": LOCAL_TTS_URL,
                },
            )
            return

        self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        try:
            if self.path == "/api/config":
                data = self.read_json()

                curl = data.get("curl")
                enabled = bool(data.get("online_enabled", False))
                api_key = data.get("online_api_key", "")
                online_test_model = str(data.get("online_test_model", "")).strip()
                local_model = str(data.get("local_model", "")).strip()
                local_voice = str(data.get("local_voice", "")).strip()
                test_text = str(data.get("test_text", "")).strip()
                rewrite_rules = data.get("rewrite_rules", [])
                presets = data.get("presets", [])
                selected_preset_id = str(data.get("selected_preset_id", ""))
                selected_voice_id = str(data.get("selected_voice_id", ""))

                if not isinstance(presets, list):
                    raise ValueError("Presets must be a list")
                clean_presets = []
                for preset in presets:
                    if not isinstance(preset, dict):
                        raise ValueError("Each preset must be an object")
                    pid = str(preset.get("id", "")).strip()
                    pcurl = str(preset.get("curl", ""))
                    pkey = preset.get("api_key", "")
                    voices = preset.get("voices", [])
                    if not pid or not isinstance(pkey, str) or not isinstance(voices, list):
                        raise ValueError("Invalid preset structure")
                    if not pcurl.strip():
                        raise ValueError("CURL is empty in preset " + str(preset.get("name", pid)))
                    parse_curl(pcurl)
                    clean_voices = []
                    for voice in voices:
                        if not isinstance(voice, dict):
                            raise ValueError("Each voice must be an object")
                        vid = str(voice.get("id", "")).strip()
                        value = str(voice.get("value", "")).strip()
                        if not vid or not value:
                            raise ValueError("Each voice needs an ID and a value")
                        clean_voices.append({"id": vid, "name": str(voice.get("name", "")).strip(), "value": value})
                    clean_presets.append({"id": pid, "name": str(preset.get("name", "")).strip(), "api_key": pkey, "curl": pcurl, "voices": clean_voices})
                selected = next((p for p in clean_presets if p["id"] == selected_preset_id), None)
                if selected:
                    selected_voice = next((v for v in selected["voices"] if v["id"] == selected_voice_id), None)
                    curl = selected["curl"]
                    api_key = selected["api_key"]
                    online_test_model = selected_voice["value"] if selected_voice else "no-voice-selected"
                elif clean_presets:
                    raise ValueError("Selected preset does not exist")

                if not isinstance(curl, str) or not curl.strip():
                    raise ValueError("CURL is empty")
                if not isinstance(api_key, str):
                    raise ValueError("API key must be a string")
                if not test_text:
                    raise ValueError("Test text is empty")
                if not isinstance(rewrite_rules, list):
                    raise ValueError("Rewrite rules must be a list")
                clean_rules = []
                for rule in rewrite_rules:
                    if not isinstance(rule, dict):
                        raise ValueError("Each rewrite rule must be an object")
                    field_from = str(rule.get("field_from", rule.get("field", ""))).strip()
                    if not field_from:
                        # Empty source field disables the entire rule.
                        continue
                    clean_rules.append({
                        "field_from": field_from,
                        "field_to": str(rule.get("field_to", "")).strip(),
                        "value_from": str(rule.get("value_from", rule.get("match", ""))),
                        "value_to": str(rule.get("value_to", rule.get("replace", ""))),
                    })
                if not online_test_model:
                    raise ValueError("Online test model/voice is empty")
                if local_model not in LOCAL_MODELS:
                    raise ValueError(
                        "Unknown local model: " + local_model
                    )
                if local_voice not in LOCAL_MODELS[local_model]:
                    raise ValueError(
                        f"Unknown local voice '{local_voice}' for model '{local_model}'"
                    )

                # Validate immediately so a bad CURL does not get saved.
                parse_curl(curl)

                with CONFIG_LOCK:
                    CONFIG["curl"] = curl
                    CONFIG["online_enabled"] = enabled
                    CONFIG["online_api_key"] = api_key
                    CONFIG["online_test_model"] = online_test_model
                    CONFIG["local_model"] = local_model
                    CONFIG["local_voice"] = local_voice
                    CONFIG["test_text"] = test_text
                    CONFIG["rewrite_rules"] = clean_rules
                    CONFIG["presets"] = clean_presets
                    CONFIG["selected_preset_id"] = selected_preset_id
                    CONFIG["selected_voice_id"] = selected_voice_id
                    result = dict(CONFIG)

                save_config()
                self.send_json(200, result)
                log(
                    f"[CONFIG] online_enabled={enabled}"
                )
                return

            if self.path == "/api/test":
                data = self.read_json()
                text = data.get("input", "")
                if not isinstance(text, str) or not text:
                    with CONFIG_LOCK:
                        text = CONFIG.get("test_text", "")
                if not isinstance(text, str) or not text:
                    raise ValueError("Missing input")

                with CONFIG_LOCK:
                    online = CONFIG["online_enabled"]

                if online:
                    with CONFIG_LOCK:
                        test_model = CONFIG["online_test_model"]
                    test_request = apply_rewrite_rules({
                        "model": test_model,
                        "tts-model": test_model,
                        "input": text,
                    })
                    routed_model = get_online_model(test_request)
                    wav = call_online(text, routed_model)
                else:
                    with CONFIG_LOCK:
                        test_model = CONFIG["local_model"]
                        test_voice = CONFIG["local_voice"]
                    test_request = apply_rewrite_rules({
                        "model": test_model,
                        "voice": test_voice,
                        "input": text,
                    })
                    request_body = json.dumps(
                        test_request, ensure_ascii=False
                    ).encode("utf-8")
                    wav = call_local(
                        request_body,
                        {"Content-Type": "application/json"},
                    )

                self.send_wav(wav)
                return

            if self.path != "/v1/audio/speech":
                self.send_json(404, {"error": "Not found"})
                return

            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("Empty request body")

            raw_body = self.rfile.read(length)

            if DEBUG_LOG_REQUEST:
                log("=" * 70)
                log("[DEBUG REQUEST] INCOMING HTTP REQUEST")
                log(f"[DEBUG REQUEST] {self.command} {self.path} HTTP/1.1")
                for header_name, header_value in self.headers.items():
                    log(f"[DEBUG REQUEST] {header_name}: {header_value}")
                log("[DEBUG REQUEST] BODY:")
                log(raw_body.decode("utf-8", errors="replace"))
                log("=" * 70)

            request = json.loads(raw_body.decode("utf-8"))
            request = apply_rewrite_rules(request)

            text = request.get("input")
            if not isinstance(text, str) or not text:
                raise ValueError("Missing 'input'")

            with CONFIG_LOCK:
                online = CONFIG["online_enabled"]

            if online:
                online_model = get_online_model(request)
                wav = call_online(text, online_model)
                log(f"[ROUTE] ONLINE model={online_model!r}")
            else:
                local_request = build_local_body(request)
                wav = call_local(
                    local_request,
                    {
                        "Content-Type": "application/json",
                    },
                )
                log("[ROUTE] LOCAL")

            self.send_wav(wav)

        except Exception as exc:
            log(f"[ERROR] {type(exc).__name__}: {exc}")
            self.send_json(
                500,
                {
                    "error": str(exc),
                    "type": type(exc).__name__,
                },
            )


def main():
    load_saved_config()
    log("=" * 60)
    log("TTS GATEWAY")
    log("=" * 60)
    log(f"Listening: http://{HOST}:{PORT}")
    log(f"Local TTS: {LOCAL_TTS_URL}")
    with CONFIG_LOCK:
        online_key_set = bool(CONFIG.get("online_api_key"))
        local_fallback = f"{CONFIG.get('local_model')}/{CONFIG.get('local_voice')}"
    log(
        "Online API key: "
        + ("SET" if online_key_set else "NOT SET")
    )
    log(f"Local fallback: {local_fallback}")
    log("")
    log("Web UI:")
    log(f"  http://{HOST}:{PORT}/")
    log("")
    log("Game endpoint:")
    log(f"  POST http://{HOST}:{PORT}/v1/audio/speech")
    log("=" * 60)

    server = ThreadingHTTPServer((HOST, PORT), ProxyHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Stopping gateway...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
