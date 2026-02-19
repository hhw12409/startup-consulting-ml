"""
📁 scripts/run_server.py
=========================
API 서버 실행 스크립트.

실행: python scripts/run_server.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from config.settings import get_settings


def main():
    s = get_settings()
    uvicorn.run(
        "src.serving.app:app",
        host=s.API_HOST,
        port=s.API_PORT,
        reload=(s.ENV == "development"),
    )


if __name__ == "__main__":
    main()