"""Load human_behaviour/.env into os.environ before other code reads getenv.

Import this module first (e.g. from api.py and database.connection).
Shell environment wins over .env (override=False).
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)
