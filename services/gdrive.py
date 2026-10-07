"""Google Drive sync engine for the Lecture Pipeline.

Handles OAuth2 authentication, folder discovery, and Markdown upload so the
pipeline can push a compiled transcript to Drive for ingestion by Google
NotebookLM.

The module is deliberately defensive:
* Missing ``credentials.json`` -> a single ``warning`` and an early return
  (no crash, no partial state).
* Missing ``token.json`` -> the OAuth flow runs once; subsequent calls reuse
  the cached credentials.
* Drive API errors (timeouts, quota exceeded, etc.) are caught so the
  pipeline always completes -- the local files are never touched.
"""

from __future__ import annotations

import logging
import io
import pathlib

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.http
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# ---------------------------------------------------------------------------
# Paths (relative to the project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"
TOKEN_PATH = PROJECT_ROOT / "token.json"

# The name of the Drive folder that will hold all synced transcripts.
DRIVE_FOLDER_NAME = "Academic_Transcripts_Sync"

# The MIME type for a plain text file.
MIME_TYPE_TEXT = "text/plain"

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("gdrive-sync")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _authorize() -> any:
    """Return credentials, or ``None`` on failure.

    The flow:

    1. If ``token.json`` exists and contains valid credentials, use them.
    2. If the credentials are expired but have a refresh token, refresh them.
    3. Otherwise, launch the OAuth consent flow (blocks until the user authorises).
    4. Save the new credentials to ``token.json``.

    If ``credentials.json`` does not exist, the function logs a warning and
    returns ``None``.
    """
    if not CREDENTIALS_PATH.is_file():
        logger.warning(
            "google-drive-sync: credentials.json not found in %s – sync disabled.",
            PROJECT_ROOT,
        )
        return None

    credentials = None
    # Check if token.json exists before attempting to load it
    if TOKEN_PATH.exists():
        try:
            credentials = Credentials.from_authorized_user_file(
                str(TOKEN_PATH), scopes=["https://www.googleapis.com/auth/drive.file"]
            )
        except Exception:  # noqa: BLE001 – corrupt token
            credentials = None

    # Refresh if expired (and a refresh token is available).
    if credentials and credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
        except Exception:  # noqa: BLE001 – network may be unavailable
            logger.warning("google-drive-sync: failed to refresh credentials")
            credentials = None

    # If we don't have valid credentials, run the OAuth flow
    if not credentials or not credentials.valid:
        # Start the OAuth consent flow.
        try:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_PATH),
                scopes=["https://www.googleapis.com/auth/drive.file"],
            )
            credentials = flow.run_local_server(port=0)
        except Exception:  # noqa: BLE001 – user may cancel, network down, etc.
            logger.warning("google-drive-sync: OAuth flow failed")
            return None

        # Persist the new credentials so the next run does not need the flow again.
        try:
            with open(TOKEN_PATH, "w", encoding="utf-8") as token_file:
                token_file.write(credentials.to_json())
        except OSError:  # noqa: BLE001 – disk full, permission denied, etc.
            logger.warning("google-drive-sync: could not write token.json")

    return credentials


def _get_or_create_folder(service) -> str | None:
    """Return the ID of the Academic_Transcripts_Sync folder in Drive root.

    Searches for an existing folder with the expected name. If it does not
    exist, creates it. Returns the folder ID, or ``None`` on failure.
    """
    try:
        response = service.files().list(
            q="mimeType='application/vnd.google-apps.folder' and name='Academic_Transcripts_Sync' and 'root' in parents",
            fields="files(id, name)",
        ).execute()
        files = response.get("files", [])
        if files:
            return files[0]["id"]

        folder_metadata = {
            "name": "Academic_Transcripts_Sync",
            "mimeType": "application/vnd.google-apps.folder",
            "parents": ["root"],
        }
        created = service.files().create(
            body=folder_metadata, fields="id"
        ).execute()
        return created.get("id")
    except Exception:  # noqa: BLE001
        logger.warning("google-drive-sync: failed to find or create folder")
        return None


def sync_markdown(course_code: str, date_str: str, content: str) -> bool:
    """Upload the compiled Markdown as a plain text file to Google Drive.

    Args:
        course_code: Course code used in the filename.
        date_str: Date string used in the filename (e.g., YYYY-MM-DD).
        content: Markdown/string content to upload.

    Returns:
        True on success, False on failure.
    """
    try:
        credentials = _authorize()
        if credentials is None:
            return False

        service = googleapiclient.discovery.build(
            "drive", "v3", credentials=credentials
        )

        folder_id = _get_or_create_folder(service)
        if folder_id is None:
            return False

        file_metadata = {
            "name": f"{course_code}_{date_str}.txt",
            "parents": [folder_id],
        }
        media = googleapiclient.http.MediaIoBaseUpload(
            io.BytesIO(content.encode("utf-8")),
            mimetype=MIME_TYPE_TEXT,
            resumable=True,
        )
        service.files().create(
            body=file_metadata, media_body=media, fields="id"
        ).execute()
        return True
    except Exception:  # noqa: BLE001
        logger.warning("google-drive-sync: upload failed")
        return False
