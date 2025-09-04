import os
import json
import uuid
import hmac
import yaml
import httpx
import hashlib
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import Response, PlainTextResponse

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# --- Configuration ---
WORKSPACE_DIR = "/workspace"
ROLES_FILE = os.path.join(WORKSPACE_DIR, "roles.yaml")
USERS_MAP_FILE = os.path.join(WORKSPACE_DIR, "users", "map.yaml")
JOBS_DIR = os.path.join(WORKSPACE_DIR, "jobs")

OPENAI_RESPONSES_MODEL = os.getenv("OPENAI_RESPONSES_MODEL", "gpt-4o-mini")
OPENAI_TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")
DEFAULT_ROLE = os.getenv("DEFAULT_ROLE", "aged_care_worker")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
PHONE_HASH_SECRET = os.getenv("PHONE_HASH_SECRET", "")


app = FastAPI(title="Scribe API")


# --- Utilities ---
def ensure_dirs() -> None:
    os.makedirs(JOBS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(USERS_MAP_FILE), exist_ok=True)


def load_roles() -> Dict[str, Dict[str, int]]:
    if not os.path.exists(ROLES_FILE):
        return {}
    with open(ROLES_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Ensure all values are int
    roles: Dict[str, Dict[str, int]] = {}
    for role, sections in data.items():
        roles[role] = {str(k): int(v) for k, v in (sections or {}).items()}
    return roles


def compute_phone_hmac(phone_number: str) -> str:
    if not PHONE_HASH_SECRET:
        raise RuntimeError("PHONE_HASH_SECRET environment variable is required for secure phone mapping.")
    key = PHONE_HASH_SECRET.encode("utf-8")
    msg = phone_number.strip().encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def get_role_for_phone(phone_number: str) -> str:
    """Securely resolve a role for a phone number using HMAC mapping stored in /users/map.yaml.
    Falls back to DEFAULT_ROLE if no mapping exists.
    """
    try:
        phone_key = compute_phone_hmac(phone_number)
    except Exception:
        # If secret missing, default to configured default role.
        return DEFAULT_ROLE

    if not os.path.exists(USERS_MAP_FILE):
        return DEFAULT_ROLE

    try:
        with open(USERS_MAP_FILE, "r", encoding="utf-8") as f:
            mapping = yaml.safe_load(f) or {}
        role = mapping.get(phone_key)
        return role or DEFAULT_ROLE
    except Exception:
        return DEFAULT_ROLE


def build_json_schema_for_sections(sections: Dict[str, int]) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, max_sentences in sections.items():
        properties[name] = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": int(max_sentences),
        }
        required.append(name)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def format_plaintext_notes(sections: Dict[str, int], notes: Dict[str, List[str]]) -> str:
    lines: List[str] = []
    for section_name, max_sentences in sections.items():
        sentences = notes.get(section_name, []) or []
        sentences = [s.strip() for s in sentences if s and str(s).strip()]
        sentences = sentences[: int(max_sentences)]
        title = section_name.replace("_", " ").title()
        content = " ".join(sentences)
        lines.append(f"{title}: {content}".strip())
    # Simple newline separation between sections (no extra blank lines)
    return "\n".join(lines)


def save_job(
    job_id: str,
    phone_number: str,
    role: str,
    sections: Dict[str, int],
    transcript: str,
    notes_structured: Dict[str, List[str]],
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
        "notes": notes_structured,
        "notes_plaintext": notes_plaintext,
    }
    path = os.path.join(JOBS_DIR, f"{job_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def send_text_message(to_number: str, body: str) -> None:
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER):
        return
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    data = {
        "To": to_number,
        "From": TWILIO_FROM_NUMBER,
        "Body": body,
    }
    with httpx.Client(timeout=30.0) as client:
        client.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))


def download_twilio_recording(recording_url: str) -> bytes:
    # Twilio RecordingUrl may require an explicit extension for mp3
    url = recording_url
    if not url.endswith((".mp3", ".wav")):
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
        result = client.audio.transcriptions.create(
            model=OPENAI_TRANSCRIPTION_MODEL,
            file=f,
            temperature=0,
            response_format="json",
            translate=True,
        )
    # Fallbacks for various SDK return shapes
    text: str = ""
    try:
        text = getattr(result, "text", "") or ""
    except Exception:
        text = ""
    return text.strip()


def generate_structured_notes(transcript: str, role: str, sections: Dict[str, int]) -> Dict[str, List[str]]:
    if OpenAI is None:
        return {name: [] for name in sections.keys()}
    client = OpenAI()

    system_prompt = (
        "You are a scribe agent helping a human in a job to write work notes "
        "based on a recount of their day. Write work notes according to the provided "
        "structure with each section within the maximum number of sentences. Use "
        "concise professional language, past tense, and avoid PII."
    )

    # Soft prompt: present the schema and limits
    schema_lines: List[str] = []
    for name, max_sents in sections.items():
        schema_lines.append(f"- {name}: {max_sents}")
    soft_prompt = (
        f"Role: {role}\n"
        f"Work Note Schema (section: max_sentences):\n" + "\n".join(schema_lines) + "\n\n"
        f"Transcript:\n{transcript.strip()}\n"
    )

    json_schema = build_json_schema_for_sections(sections)

    response = client.responses.create(
        model=OPENAI_RESPONSES_MODEL,
        input=[
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": soft_prompt}]},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "work_notes",
                "schema": json_schema,
                "strict": True,
            },
        },
    )

    raw_text: str = ""
    # Try common accessors for Responses API aggregated text
    raw_text = getattr(response, "output_text", "") or raw_text
    if not raw_text:
        # Attempt to build text from output content blocks
        try:
            outputs = getattr(response, "output", [])
            if outputs:
                first = outputs[0]
                content = first.get("content") if isinstance(first, dict) else None
                if isinstance(content, list) and content and isinstance(content[0], dict):
                    raw_text = content[0].get("text", "")
        except Exception:
            raw_text = ""

    notes: Dict[str, List[str]] = {name: [] for name in sections.keys()}
    if raw_text:
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if k in sections and isinstance(v, list):
                        notes[k] = [str(s).strip() for s in v if str(s).strip()]
        except Exception:
            pass

    # Enforce max sentences per section
    for name, max_sents in sections.items():
        notes[name] = (notes.get(name) or [])[: int(max_sents)]
    return notes


def process_recording(from_number: str, recording_url: str, call_sid: str) -> None:
    try:
        audio_bytes = download_twilio_recording(recording_url)
        transcript = transcribe_audio_bytes(audio_bytes) or ""
        role = get_role_for_phone(from_number)

        roles = load_roles()
        sections = roles.get(role) or {}
        if not sections:
            sections = load_roles().get(DEFAULT_ROLE, {})

        notes_structured = generate_structured_notes(transcript=transcript, role=role, sections=sections)
        notes_plaintext = format_plaintext_notes(sections, notes_structured)

        job_id = str(uuid.uuid4())
        save_job(
            job_id=job_id,
            phone_number=from_number,
            role=role,
            sections=sections,
            transcript=transcript,
            notes_structured=notes_structured,
            notes_plaintext=notes_plaintext,
        )

        if notes_plaintext.strip():
            send_text_message(to_number=from_number, body=notes_plaintext)
    except Exception:
        # Swallow errors to avoid retry loops; Twilio will already have a call flow response
        pass


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
