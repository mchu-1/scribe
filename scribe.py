import os
import json
import uuid
import hmac
import yaml
import httpx
import hashlib
import tempfile
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import Response

import redis
from rq import Queue

from openai import OpenAI


from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# --- Configuration ---
workspace_dir = os.getcwd()
jobs_dir = os.path.join(workspace_dir, "jobs")

redis_url = os.getenv("REDIS_URL", "")
rq_queue_name = "scribe"
rq_default_timeout = 600

openai_responses_model = "gpt-5"
openai_transcription_model = "whisper-1"
max_total_sentences = 10

twilio_api_base_url = "https://api.twilio.com"
twilio_api_version = "2010-04-01"

# MMS character hard limit (Twilio MMS body limit)
mms_character_limit = 1600

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
PHONE_HASH_SECRET = os.getenv("PHONE_HASH_SECRET", "")


app = FastAPI(title="Scribe API")


# --- Utilities ---
def ensure_dirs() -> None:
    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(os.path.join(workspace_dir, "users"), exist_ok=True)


def resolve_workspace_path(relative_path: str) -> str:
    primary_path = os.path.join(workspace_dir, relative_path)
    secondary_path = os.path.join(os.getcwd(), relative_path)
    return primary_path if os.path.exists(primary_path) else secondary_path


def resolve_user_yaml_path() -> str:
    """Resolve the path to the `users.yaml` mapping file.

    Resolution rules (no silent fallbacks):
    1) If USER_YAML_PATH is set: use it if it exists, otherwise raise.
    2) Otherwise, use `/etc/secrets/users.yaml` if it exists, otherwise raise.
    """
    env_yaml_path = os.getenv("USER_YAML_PATH", "").strip()
    if env_yaml_path:
        if os.path.exists(env_yaml_path):
            return env_yaml_path
        raise RuntimeError(f"USER_YAML_PATH is set but file not found: {env_yaml_path}")

    render_secret_yaml = "/etc/secrets/users.yaml"
    if os.path.exists(render_secret_yaml):
        return render_secret_yaml

    raise RuntimeError("Render secret file not found at /etc/secrets/users.yaml")


def load_user_mapping_yaml(path: str) -> Dict[str, str]:
    """Load user mapping from YAML; return empty dict on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def load_system_prompt() -> str:
    path = resolve_workspace_path("scribe.md")
    if not os.path.exists(path):
        raise FileNotFoundError("System prompt for scribe not found in workspace.")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_redis_queue() -> Optional[Queue]:
    if not redis_url:
        return None
    try:
        connection = redis.from_url(redis_url)
        return Queue(name=rq_queue_name, connection=connection, default_timeout=rq_default_timeout)
    except Exception:
        return None


 


def compute_phone_hmac(phone_number: str) -> str:
    if not PHONE_HASH_SECRET:
        raise RuntimeError("PHONE_HASH_SECRET environment variable is required for secure phone mapping.")
    key = PHONE_HASH_SECRET.encode("utf-8")
    msg = phone_number.strip().encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def get_role_for_phone(phone_number: str) -> str:
    """Resolve role via HMAC mapping in users.yaml.
    Raises if the secret, mapping file, or mapping entry is missing.
    """
    phone_key = compute_phone_hmac(phone_number)

    map_path = resolve_user_yaml_path()
    if not map_path or not os.path.exists(map_path):
        raise RuntimeError(
            "USER_YAML_PATH is not set or mapping file not found; cannot resolve role."
        )

    mapping = load_user_mapping_yaml(map_path)
    role: Optional[str] = mapping.get(phone_key)
    if not role or not isinstance(role, str) or not role.strip():
        raise RuntimeError("No role mapping found for the provided phone number.")
    return role.strip()


 


def enforce_mms_limit(text: str, limit: int = mms_character_limit) -> str:
    if limit <= 0:
        return ""
    if not text:
        return ""
    return text[: limit]


def save_job(
    job_id: str,
    phone_number: str,
    role: str,
    sections: Dict[str, int],
    transcript: str,
    notes_plaintext: str,
) -> str:
    ensure_dirs()
    payload = {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phone_number": phone_number,
        "phone_hmac": compute_phone_hmac(phone_number) if PHONE_HASH_SECRET else None,
        "role": role,
        "sections": sections,
        "transcript": transcript,
        "notes_plaintext": notes_plaintext,
    }
    path = os.path.join(jobs_dir, f"{job_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def send_text_message(to_number: str, body: str) -> None:
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER):
        return
    url = f"{twilio_api_base_url}/{twilio_api_version}/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    data = {
        "To": to_number,
        "From": TWILIO_FROM_NUMBER,
        "Body": body,
    }
    with httpx.Client(timeout=30.0) as client:
        client.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))


def download_twilio_recording(recording_url: str) -> bytes:
    """Fetch the Twilio recording as MP3.

    Twilio stores recordings as WAV (default, 128 kbps) and returns MP3 (32 kbps)
    when ".mp3" is appended to the RecordingUrl.
    """
    url = recording_url
    if url.endswith(".wav"):
        url = url[:-4] + ".mp3"
    elif not url.endswith(".mp3"):
        url = url + ".mp3"
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        resp.raise_for_status()
        return resp.content


def transcribe_audio_bytes(file_bytes: bytes) -> str:
    if OpenAI is None:
        return ""
    client = OpenAI()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        result = client.audio.translations.create(
            model=openai_transcription_model,
            file=f,
        )
    # Fallbacks for various SDK return shapes
    text: str = ""
    try:
        text = getattr(result, "text", "") or ""
    except Exception:
        text = ""
    return text.strip()


def generate_plaintext_notes(transcript: str, role: str) -> str:
    if OpenAI is None:
        return ""
    try:
        print(
            f"[scribe] notes:starting responses.create model={openai_responses_model} "
            f"api_key_set={bool(os.getenv('OPENAI_API_KEY'))} role={role} "
            f"transcript_len={len(transcript)}",
            flush=True,
        )

        client = OpenAI()

        system_text = load_system_prompt()
        role_words = role.replace("_", " ").strip()
        user_text = (
            f"Role keywords: {role_words}\n\n"
            f"Transcript:\n{transcript.strip()}\n"
        )

        response = client.responses.create(
            model=openai_responses_model,
            reasoning={"effort": "low"},
            input=[
                {"role": "developer", "content": system_text},
                {"role": "user", "content": user_text},
            ],
        )
        print("[scribe] notes:responses.create OK", flush=True)

        text: str = getattr(response, "output_text", "") or ""
        print(f"[scribe] notes:text_len={len(text.strip())}", flush=True)
        return text.strip()
    except Exception as e:
        print(f"[scribe] notes:error {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return ""


def process_recording(from_number: str, recording_url: str, call_sid: str) -> None:
    try:
        job_id = str(uuid.uuid4())
        audio_bytes = download_twilio_recording(recording_url)
        transcript = transcribe_audio_bytes(audio_bytes) or ""
        print(
            f"[scribe] job_id={job_id} call_sid={call_sid} from={from_number} transcript:\n{transcript}",
            flush=True,
        )
        role = get_role_for_phone(from_number)

        notes_plaintext = generate_plaintext_notes(
            transcript=transcript,
            role=role,
        )
        notes_plaintext = enforce_mms_limit(notes_plaintext, mms_character_limit)
        print(
            f"[scribe] job_id={job_id} notes (len={len(notes_plaintext)}):\n{notes_plaintext}",
            flush=True,
        )

        save_job(
            job_id=job_id,
            phone_number=from_number,
            role=role,
            sections={},
            transcript=transcript,
            notes_plaintext=notes_plaintext,
        )

        if notes_plaintext.strip():
            send_text_message(to_number=from_number, body=notes_plaintext)
    except Exception as e:
        print(f"[scribe] job_id={job_id} error in process_recording: {e}", flush=True)
        print(traceback.format_exc(), flush=True)


# --- Routes ---
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/voice/incoming")
async def voice_incoming() -> Response:
    # TwiML: Say the prompt, then Record for up to 5 minutes, beep on start
    twiml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<Response>"
        "<Say>Record your notes after the beep</Say>"
        "<Record action=\"/voice/complete\" method=\"POST\" maxLength=\"300\" playBeep=\"true\" />"
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")


@app.post("/voice/complete")
async def voice_complete(request: Request, background_tasks: BackgroundTasks) -> Response:
    form = await request.form()
    from_number = str(form.get("From", "")).strip()
    recording_url = str(form.get("RecordingUrl", "")).strip()
    call_sid = str(form.get("CallSid", "")).strip()

    if from_number and recording_url:
        queue = get_redis_queue()
        if queue:
            try:
                queue.enqueue(process_recording, from_number, recording_url, call_sid)
            except Exception:
                background_tasks.add_task(process_recording, from_number, recording_url, call_sid)
        else:
            background_tasks.add_task(process_recording, from_number, recording_url, call_sid)

    # Respond quickly to Twilio to end the call politely
    twiml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<Response>"
        "<Say>Thanks. Your notes will be sent shortly. Goodbye.</Say>"
        "<Hangup/>"
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")


# Entrypoint for local dev: `uvicorn scribe:app --reload --host 0.0.0.0 --port 8000`
