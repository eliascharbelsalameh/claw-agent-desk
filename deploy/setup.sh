#!/usr/bin/env bash
# One-time setup of the desk on the Oracle A1 instance (Ubuntu 22.04/24.04,
# ARM64). Run it as the user that will own the service, from anywhere:
#
#   bash setup.sh            # or: DESK_DIR=/some/path bash setup.sh
#
# It installs Python 3.11+ and git, clones (or updates) the repo, builds the
# virtualenv and creates an empty .env. It never asks for or writes a
# credential: fill .env yourself afterwards (see deploy/README.md).
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/eliascharbelsalameh/claw-agent-desk.git}"
DESK_DIR="${DESK_DIR:-$HOME/claw-agent-desk}"

pick_python() {
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
      echo "$candidate"
      return
    fi
  done
}

sudo apt-get update -y
sudo apt-get install -y git
PYTHON="$(pick_python || true)"
if [ -z "$PYTHON" ]; then
  # Ubuntu 22.04 ships 3.10; the desk needs 3.11+ (it parses Alpaca's
  # nanosecond timestamps with datetime.fromisoformat).
  sudo apt-get install -y python3.11 python3.11-venv
  PYTHON=python3.11
fi
"$PYTHON" -m venv --help >/dev/null 2>&1 || sudo apt-get install -y "${PYTHON}-venv"
echo "Using $("$PYTHON" --version)"

if [ -d "$DESK_DIR/.git" ]; then
  git -C "$DESK_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$DESK_DIR"
fi
cd "$DESK_DIR"

"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
fi
chmod 600 .env
mkdir -p logs state

.venv/bin/python -m pytest -q
echo
echo "Setup done in $DESK_DIR."
echo "Next: fill in the six credentials in $DESK_DIR/.env (nano .env), then follow deploy/README.md."
