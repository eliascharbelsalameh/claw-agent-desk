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

Log in as the user that runs the service and type them into `.env` with an editor:

```bash
cd ~/claw-agent-desk && nano .env
```

Fill in `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `FRED_API_KEY`, `FINNHUB_API_KEY`,
`SEC_EDGAR_USER_AGENT` and `NVIDIA_API_KEY`: each value right after its `=`, with no spaces around
the `=`. The User-Agent contains a space, so quote it: `SEC_EDGAR_USER_AGENT="Your Name you@example.com"`.
Save with Ctrl+O and Enter, then exit with Ctrl+X.

Don't set them with `export` or `echo ... >> .env`, because anything typed on the command line ends up
in your shell history. Don't paste them into a chat or a Claude session either (section 6 keeps a
Claude session on this server away from them). Keep the file readable by you alone, and check that
each credential is present without printing it:

```bash
chmod 600 .env
.venv/bin/python -c "from ui.support import credential_status; print(credential_status())"
```

## 3. Dry run first

A quick check that the server can reach every source and model: two stocks, log-only orders (nothing
is sent to Alpaca), a few minutes:

```bash
.venv/bin/python -m agents.scheduler --once decision --watchlist AAPL MSFT
```

It prints each stock's decision, anything deferred, and the orders it would have placed. The full
trail is in `logs/trace-YYYYMMDD.jsonl`. Drop `--watchlist` for the full 22-stock cycle (about an hour).

A log-only run keeps its own state file, `state/desk_state-log-only.json`, separate from the service's
`state/desk_state.json`. So the service never treats a session as already decided, and never acts on
positions a log-only run only pretended to open. The scheduler refuses a state file that belongs to the
other mode. To start a mode over from scratch, delete its state file.

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
SSH tunnel. Never open port 8501 to the internet: the app makes Build calls on your key, and its trace
viewer shows everything the agents saw.

```bash
# on the server
.venv/bin/streamlit run ui/streamlit_app.py
# on your machine
ssh -L 8501:localhost:8501 <user>@<instance-ip>    # then open http://localhost:8501
```

The **Desk state** tab lists positions, deferred stocks, decisions and every pass, including LLM
calls and failures. The **Trace viewer** tab replays any day's trace.

## 6. A Claude Code session on the server

A Claude Code session can read every file its Unix user can read and run anything that user can run,
so permission rules inside Claude Code aren't enough to keep the keys out of it. Run it as a separate
user that has no sudo and can't read `.env`:

```bash
# as your admin user (the one that runs the service), once
sudo adduser --disabled-password --gecos "" claude   # no password, not in the sudo group
sudo apt-get install -y acl
# claude may pass through your home and the repo, and read logs/ and state/ - nothing else
setfacl -m u:claude:x ~ ~/claw-agent-desk
setfacl -R -m u:claude:rX ~/claw-agent-desk/logs ~/claw-agent-desk/state
setfacl -m d:u:claude:rX ~/claw-agent-desk/logs ~/claw-agent-desk/state   # files created later too
sudo -u claude cat ~/claw-agent-desk/.env   # must fail with "Permission denied"
```

Then start sessions as that user, in its own clone of the repo:

```bash
sudo -iu claude
curl -fsSL https://claude.ai/install.sh | bash   # Claude Code for this user only; then log in
git clone https://github.com/eliascharbelsalameh/claw-agent-desk.git && cd claw-agent-desk && claude
```

The session sees the code, plus the live traces and state in your `~/claw-agent-desk/logs` and
`~/claw-agent-desk/state`. It can't see the keys, the data cache or anything else in your home, and it
can't restart the service or use sudo; do those yourself. Never start `claude` from your admin account.

The desk also keeps the keys out of everything it writes. FRED and Finnhub take their keys in the URL,
so error texts are redacted before they become data gaps (which the analysts see) or trace lines.

## Updating

```bash
cd ~/claw-agent-desk && git pull --ff-only && .venv/bin/python -m pip install -r requirements.txt
systemctl --user restart claw-desk-scheduler
```
