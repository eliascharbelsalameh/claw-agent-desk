# Deploying the desk on the Oracle A1 instance

The scheduler (`python -m agents.scheduler`) is the part that runs unattended. It holds no secrets
of its own. It reads the six credentials from the repo's `.env`, which you fill in yourself on the
server. Nothing here puts a credential in a file that gets committed, a unit file, or a command line.

## 1. One-time setup

On the instance (over SSH, e.g. from Termius):

```bash
curl -fsSLO https://raw.githubusercontent.com/eliascharbelsalameh/claw-agent-desk/master/deploy/setup.sh
bash setup.sh
```

If the repo is private, clone it yourself first (`gh auth login` or a deploy key), then run
`bash deploy/setup.sh` from inside the clone. The script installs git and Python 3.11+ (Ubuntu
22.04 ships 3.10, which can't parse Alpaca's nanosecond timestamps). It also creates `.venv`,
installs `requirements.txt`, creates `.env` from `.env.example` with mode 600, and runs the tests.

## 2. Credentials

```bash
cd ~/claw-agent-desk && nano .env
```

Fill in `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `FRED_API_KEY`, `FINNHUB_API_KEY`,
`SEC_EDGAR_USER_AGENT` and `NVIDIA_API_KEY`. Type them on the server. Don't paste them into a
chat or a Claude session. Check that each one is present without printing it:

```bash
.venv/bin/python -c "from ui.support import credential_status; print(credential_status())"
```

## 3. Dry run first

This runs one decision cycle for the next session with log-only orders (nothing is sent to Alpaca):

```bash
.venv/bin/python -m agents.scheduler --once decision
```

It prints each stock's decision, anything deferred, and the orders it would have placed. The full
trail is in `logs/trace-YYYYMMDD.jsonl`, and the state is in `state/desk_state.json`. To start over
from scratch, delete `state/desk_state.json`.

## 4. Run it as a service

```bash
mkdir -p ~/.config/systemd/user
cp deploy/claw-desk-scheduler.service ~/.config/systemd/user/
loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now claw-desk-scheduler
journalctl --user -u claw-desk-scheduler -f
```

The unit runs with `--paper-orders`, so its orders go to the Alpaca **paper** account (the URL in
`.env` is `paper-api.alpaca.markets`). Drop the flag in the unit for log-only.

The scheduler wakes every minute. It runs the day's decision cycle at 08:00 ET, before the open,
and retries deferred stocks every 30 minutes until 15:00 ET. It follows Alpaca's market clock, so
weekends and holidays are skipped whatever the server's time zone is. Restarts are safe: the state
file records what was already decided, and order ids are derived from the decision, so an order is
never placed twice.

## 5. Watch it

The Streamlit app reads the same state and traces. Run it on the server and reach it through an
SSH tunnel. Never open port 8501 to the internet: the app can spend Build credits, and its trace
viewer shows everything the agents saw.

```bash
# on the server
.venv/bin/streamlit run ui/streamlit_app.py
# on your machine
ssh -L 8501:localhost:8501 <user>@<instance-ip>    # then open http://localhost:8501
```

The **Desk state** tab lists positions, deferred stocks, decisions and every pass, including LLM
calls and failures. The **Trace viewer** tab replays any day's trace.

## Updating

```bash
cd ~/claw-agent-desk && git pull --ff-only && .venv/bin/python -m pip install -r requirements.txt
systemctl --user restart claw-desk-scheduler
```
