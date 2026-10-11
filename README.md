# Online Class Transcriber

> A local-first capture and synthesis pipeline for online lectures. It records the target window's slides (screenshots) and the system's desktop audio via WASAPI loopback, transcribes + diarizes the audio with WhisperX, synthesizes structured study notes using domain-specific profiles, persists everything in SQLite, optionally syncs to Google Drive, and serves an interactive playback room via FastAPI.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [System Requirements & Prerequisites](#system-requirements--prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
  - [.env File](#env-file)
  - [Hugging Face Token (HF_TOKEN)](#hugging-face-token-hf_token)
  - [Google Drive OAuth (credentials.json)](#google-drive-oauth-credentialsjson)
- [Usage](#usage)
  - [1. Start the Web Server](#1-start-the-web-server)
  - [2. Access the Web UI](#2-access-the-web-ui)
  - [3. Running a Recording](#3-running-a-recording)
  - [4. Library & Playback](#4-library--playback)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)
- [License](#license)

## Overview

This application is designed for capturing and processing online lectures entirely on your local machine. It combines:

1. **Screen capture** – Periodically captures screenshots from a specific application window (e.g., Zoom, Teams, Google Meet, YouTube) to preserve slide context.
2. **System audio capture** – Records the internal desktop audio (what you hear) via Windows WASAPI Loopback.
3. **ASR + Diarization** – Uses [WhisperX](https://github.com/m-bain/whisperX) to transcribe audio and [Pyannote](https://github.com/pyannote) to identify speaker turns.
4. **AI Synthesis** – Uses a NVIDIA-hosted LLM to generate structured study notes, guided by configurable **Domain Profiles** (criminology, general, etc.).
5. **Persistence** – Stores sessions, transcript chunks, slide metadata, and synthesis artifacts in a local **SQLite** database.
6. **Cloud Sync** – Optionally uploads session artifacts to a Google Drive folder with OAuth 2.0.
7. **Playback** – Provides an interactive **Playback Room** to review synced audio, transcript, and slides together.

All processing happens locally by default (except LLM calls to the configured NVIDIA API endpoint and optional Google Drive sync).

## Architecture

The application is built on the following stack:

| Component | Purpose |
|---|---|
| **FastAPI + Uvicorn** | HTTP API and static web UI (`server.py`, `static/`) |
| **SQLite + SQLAlchemy** | Local session/ledger persistence (`sqlite_ledger.py`, `db.py`, `models.py`) |
| **soundcard + PyAudioWPatch** | Windows WASAPI Loopback audio capture (`recorder.py`). Captures default speaker output at 16 kHz mono WAV. |
| **mss / pygetwindow / Pillow / ImageHash** | Window enumeration, targeted window capture, image conversion, and slide deduplication (`window_capture.py`, `vision_utils.py`) |
| **WhisperX (torch/torchaudio)** | Offline ASR, word-level alignment, and Pyannote speaker diarization (`transcriber.py`) |
| **OpenAI SDK (NVIDIA endpoint)** | LLM synthesis of structured notes with per-stage model overrides (`summarizer.py`, `config.py`) |
| **Google Drive API (google-auth-oauthlib)** | OAuth 2.0 desktop flow + resumable uploads (`services/gdrive.py`) |
| **Domain Profiles** | Configurable synthesis prompts/structure (`domain_profiles.py`) |

**Core Pipeline:**

1. **Capture** – Start recording: enumerate windows, select target, record WASAPI loopback audio (16 kHz mono) and periodically capture slides from the target window. Frames are deduplicated via perceptual hashing.
2. **Transcribe & Diarize** – Convert audio to speaker-attributed transcript (`raw_transcript.txt`) via WhisperX (transcribe → align → diarize). Checkpoints allow resume on failure.
3. **Synthesize Notes** – Feed transcript + slide manifest into the LLM using the active domain profile. Supports separate MAP/REDUCE models and optional reasoning mode.
4. **Persist** – Write session metadata, transcript chunks, slides, and synthesis outputs to SQLite. Generate a complete study notes document.
5. **Sync (optional)** – Upload artifacts (audio, transcript, slides, notes) to Google Drive using stored OAuth tokens.
6. **Playback** – Browse sessions in the **Library Dashboard** and open the **Playback Room** (`static/playback.html`) to review audio synchronized with slides and transcript.

## System Requirements & Prerequisites

- **Operating System:** Windows 10/11 (required for WASAPI loopback capture). Other platforms are not supported.
- **Python:** 3.10 or 3.11 (tested with 3.11.9).
- **FFmpeg:** Must be installed and available in your `PATH`. Required by WhisperX/Pyannote and audio processing.
- **Hugging Face Account:** Required to obtain a token and accept Pyannote model user agreements.
- **NVIDIA API Access:** API key for the NVIDIA LLM endpoint (used by `summarizer.py`).
- **Google Cloud Project (optional):** Only needed if you want to enable Google Drive sync.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/online-class-transcriber.git
   cd online-class-transcriber
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Key dependencies include: `fastapi`, `uvicorn`, `soundcard`, `pyaudiowpatch`, `whisperx`, `torch`, `torchaudio`, `openai`, `pydantic-settings`, `sqlalchemy`, `mss`, `pygetwindow`, `pywin32`, `Pillow`, `ImageHash`, `soundfile`, `numpy`, `tenacity`, `rich`, `google-api-python-client`, `google-auth-oauthlib`.

4. **Verify FFmpeg**
   ```bash
   ffmpeg -version
   ```
   If not found, install [FFmpeg](https://ffmpeg.org/download.html) and add it to your system PATH.

## Configuration

### .env File

Create a `.env` file in the project root. The application uses [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) (`config.py`) for strict validation at boot — **the app will fail immediately on startup if mandatory keys are missing**.

| Variable | Required | Default | Description |
|---|---|---|---|
| `NVIDIA_API_KEY` | **Yes** | — | API key for NVIDIA LLM endpoint (used by summarizer). |
| `HF_TOKEN` | **Yes** | — | Hugging Face access token with read access (required for Pyannote diarization). |
| `LLM_MODEL_NAME` | No | `meta/llama-3.2-11b-vision-instruct` | Default LLM model used for synthesis. |
| `MAP_MODEL` | No | Falls back to `LLM_MODEL_NAME` | Model for stage 1 (map) — useful for vision/light tasks. |
| `REDUCE_MODEL` | No | Falls back to `LLM_MODEL_NAME` | Model for stage 3 (reduce) — typically heavier/reasoning model. |
| `REDUCE_REASONING_ENABLED` | No | `false` | Enable thinking/reasoning for reduce stage (`true`/`false`). |
| `WHISPER_COMPUTE_TYPE` | No | `int8` | CTranslate2 compute type for WhisperX (`float32`, `float16`, `int8`, `int8_float16`, etc.). Device-dependent. |
| `WHISPER_BATCH_SIZE` | No | `16` | Batch size for WhisperX processing. Tune based on CPU/RAM. |
| `DRIVE_FOLDER_ID` | No | `None` | Optional Google Drive folder ID to sync sessions into. If unset, defaults to root or app-configured behavior. |

**Example `.env`:**

```env
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional overrides
LLM_MODEL_NAME=meta/llama-3.2-11b-vision-instruct
MAP_MODEL=meta/llama-3.2-11b-vision-instruct
REDUCE_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
REDUCE_REASONING_ENABLED=false
WHISPER_COMPUTE_TYPE=int8
WHISPER_BATCH_SIZE=16
DRIVE_FOLDER_ID=your_drive_folder_id
```

### Hugging Face Token (HF_TOKEN)

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a **read** access token.
2. Accept the user agreements for the Pyannote models used by WhisperX:
   - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) (or the version pulled by `whisperx`)
   - Any other alignment/diarization models WhisperX downloads on first run
3. Add the token to `.env` as `HF_TOKEN=hf_...`

First-time download of Pyannote models requires network access and valid token.

### Google Drive OAuth (credentials.json)

Google Drive sync is optional. To enable it:

1. **Create a GCP Project**
   - Go to [console.cloud.google.com](https://console.cloud.google.com/)
   - Create a new project (or select existing)

2. **Enable the Google Drive API**
   - APIs & Services → Library → Search "Google Drive API" → Enable

3. **Create a Desktop OAuth Client ID**
   - APIs & Services → Credentials → Create Credentials → OAuth client ID
   - Application type: **Desktop app**
   - Name: e.g. `online-class-transcriber`
   - Download the JSON file (it will be named like `client_secret_*.json`)

4. **Save credentials.json in the project root**
   - Rename to exactly: `credentials.json`
   - Place it in `C:\Dev\online-class-transcriber\credentials.json` (root of this repo)

On first Drive sync attempt, the app will open a browser OAuth consent flow. After authorizing, it saves refresh tokens to a local file (typically `token.json` in the root) and reuses them thereafter.

## Usage

### 1. Start the Web Server

Activate your virtual environment and start FastAPI with Uvicorn:

```bash
# From project root
.venv\Scripts\activate
uvicorn server:app --reload
```

The server will start on `http://127.0.0.1:8000`. On import, `server.py` loads `config.settings` at top-level — so missing `.env`/keys will raise a validation error immediately.

### 2. Access the Web UI

Open your browser to [http://127.0.0.1:8000](http://127.0.0.1:8000). The dashboard provides:

- **Domain Profiles**: Choose synthesis style/profile (e.g., `criminology`, `general`). Affects prompts and note structure.
- **Window Targeting**: Enumerate open windows and select the lecture window to capture slides from. The app captures that window only (not full screen unless selected).
- **Library Dashboard**: List all recorded sessions with metadata, links to Playback Room, and actions (Delete/Retry Sync).
- **Playback Room**: Interactive review of audio, transcript, and slides in sync (`/static/playback.html`).

### 3. Running a Recording

1. From the main dashboard, select your **Domain Profile**.
2. Under **Window Targeting**, click **Refresh** to list windows, then select the target lecture window (Zoom/Teams/Meet/YouTube etc.).
3. Choose/confirm the **Course Code** (used to name output directories).
4. Click **Start Recording**. The UI will show pipeline states: `recording → transcribing → synthesizing → complete` (or `error`).
5. During recording: system audio (WASAPI loopback) + microphone (best-effort) are captured; target window slides are captured periodically with deduplication. Checkpoints allow resuming if interrupted.
6. On completion: transcript, slides manifest, study notes, and session metadata are persisted to SQLite. Artifacts appear under `output/` and in the Library.

**Notes:**
- WASAPI loopback requires Windows. If no loopback device is found, the recorder will raise a clear error.
- Microphone capture is best-effort: if unavailable, recording continues with system audio only (diarization may be less effective at separating room voices).
- The pipeline can be resumed on restart if checkpoints exist (`resume=True` path in CLI/programmatic flows).

### 4. Library & Playback

- **Library Dashboard** (`/` routes to library UI): View all sessions with course code, date/time, domain, capture mode/target, and status. Click **Open Playback** to review a session.
- **Playback Room** (`/static/playback.html?id={session_id}`): Syncs audio with transcript segments and slides. Use playback controls to scrub; transcript highlights follow playback; slides advance to the corresponding time.
- **Session API**: `GET /api/sessions/{id}` returns full session payload (transcript, slides, notes path, audio path) for playback.
- **Actions**: Delete session (removes DB row and associated artifacts as applicable) and Retry Sync (re-attempts Google Drive upload).

## Project Structure

```text
online-class-transcriber/
├── config.py                # Pydantic settings (env validation, singleton)
├── server.py                # FastAPI app: routes, state, background runs, static mount
├── main.py                  # CLI/programmatic pipeline (record→transcribe→synthesize)
├── recorder.py              # WASAPI loopback + mic capture, threading, checkpoints, atomic writes
├── transcriber.py           # WhisperX: transcribe/align/diarize with checkpoints
├── summarizer.py            # NVIDIA LLM synthesis with per-stage models/reasoning
├── domain_profiles.py       # Domain-specific synthesis profiles/prompts
├── window_capture.py        # Window enumeration + targeted screenshot capture
├── vision_utils.py          # Image hashing/deduplication utilities
├── sqlite_ledger.py         # SQLite persistence layer (sessions, transcript, slides)
├── db.py                    # DB engine/session setup
├── models.py                # SQLAlchemy models
├── services/
│   └── gdrive.py            # Google Drive OAuth + upload logic
├── static/                  # Web UI (index.html, playback.html, CSS/JS)
├── tests/                   # Pytest suite (unit + integration, heavily mocked)
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── .env (local, untracked)  # Secrets/config (see Configuration)
```

## Key Features

- **Strict boot validation**: `pydantic-settings` fails fast if `NVIDIA_API_KEY` or `HF_TOKEN` missing.
- **WASAPI Loopback capture**: Captures system audio (what you hear) for online lectures; prefers default speaker loopback device.
- **Dual-source audio**: Simultaneously captures loopback + microphone (best-effort), mixes to 16 kHz mono WAV.
- **Window-targeted capture**: Capture only the lecture window (no need for fullscreen), with perceptual hash deduplication.
- **Resume-friendly**: Checkpoints for transcription/alignment/diarization phases and periodic emergency audio backup during recording.
- **Domain Profiles**: Tailor synthesis output to course context.
- **Interactive Playback Room**: Time-synced review of audio, transcript, and slides.
- **Local-first + optional cloud**: SQLite persistence by default; Google Drive sync opt-in via OAuth.

## Testing

Run the full test suite (all tests mock external I/O):

```bash
# From project root with venv active
pytest -q
```

Current status: **343 passed, 1 warning**. Tests mock WhisperX, LLM, audio backends, screen capture, and Google Drive, so they run offline.

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| **ValidationError on startup** | Missing `NVIDIA_API_KEY` or `HF_TOKEN` in `.env` | Create `.env` with mandatory keys (see Configuration). |
| **No loopback devices found** | WASAPI loopback unavailable / speakers disabled | Enable speakers/output device, ensure Windows audio is active. Check `recorder.get_loopback_devices()` returns entries. |
| **Unicode/encoding errors in console** | Non-ASCII device names on some terminals | Use a UTF-8 capable terminal (PowerShell/Windows Terminal). Codebase now handles encoding safely. |
| **Pyannote model download fails** | Missing/invalid HF token or agreements not accepted | Verify `HF_TOKEN` is correct and accept agreements on Hugging Face for required Pyannote models. |
| **WhisperX slow on CPU** | Default `WHISPER_COMPUTE_TYPE=int8` is CPU-friendly; very long lectures can be slow | Tune `WHISPER_COMPUTE_TYPE` and `WHISPER_BATCH_SIZE` per hardware. Consider smaller Whisper model if available via code paths. |
| **Google Drive auth loop** | Missing `credentials.json` or browser blocks | Ensure `credentials.json` exists in root; complete OAuth in default browser; check `token.json` is created. |
| **FFmpeg not found** | Not in PATH | Install FFmpeg and add to system PATH. |

## Security Notes

- **Local-first by default**: Audio, transcripts, slides, and notes stay on disk under `output/` and in `sqlite_ledger.db`.
- **Secrets**: `.env`, `credentials.json`, and `token.json` contain sensitive credentials — **never commit** them. They are untracked by git.
- **API keys**: `NVIDIA_API_KEY` and `HF_TOKEN` are loaded via validated config; server imports config at top level.
- **Drive scope**: OAuth requests minimal Drive scope needed for upload to specified folder.

## License

This project is provided as-is for educational/research purposes. Review third-party licenses for dependencies (WhisperX, Pyannote, FastAPI, etc.) before redistribution.
