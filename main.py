#!/usr/bin/env python3
"""
Entry point for the Local Wikipedia RAG Assistant.

Usage:
    python main.py              # launch Streamlit UI (default)
    python main.py --mode cli   # launch terminal chat
    python main.py --mode cli --show-sources --debug
"""

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local Wikipedia RAG Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["ui", "cli"],
        default="ui",
        help="ui = Streamlit browser app (default) | cli = terminal chat",
    )
    args = parser.parse_args()

    if args.mode == "ui":
        app_path = Path(__file__).parent / "ui" / "streamlit_app.py"
        subprocess.run(
            ["streamlit", "run", str(app_path), "--server.headless", "false"],
            check=True,
        )
    else:
        from ui.cli import run_cli
        run_cli()


if __name__ == "__main__":
    main()
