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
`bash deploy/setup.sh` from inside the clone. If you cloned it somewhere else, such as
`~/repos/claw-agent-desk`, point the script at that clone and link it to `~/claw-agent-desk`, which
is where the service unit and the commands below look:

```bash
cd ~/repos/claw-agent-desk && DESK_DIR=$PWD bash deploy/setup.sh
ln -s ~/repos/claw-agent-desk ~/claw-agent-desk
```

The script installs git and Python 3.11+ (Ubuntu
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
in your shell history. Don't paste them into a chat or a Claude session either. Keep the file
readable by you alone, and check that each credential is present without printing it:

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

The unit runs with `--paper-orders`, so its orders go to the Alpaca **paper** account
(`ALPACA_TRADING_BASE_URL`, which defaults to `paper-api.alpaca.markets`). Drop the flag in the unit for
log-only.

The scheduler wakes every minute. It runs the day's decision cycle at 07:30 ET, before the open,
and retries deferred stocks every 30 minutes until 15:00 ET. It follows Alpaca's market clock, so
weekends and holidays are skipped whatever the server's time zone is. Restarts are safe: the state
file records what was already decided, and order ids are derived from the decision, so an order is
never placed twice.

Check that linger is on. Without it, the service stops when your last SSH session closes:

```bash
loginctl show-user "$USER" --property=Linger   # must print Linger=yes
```

**The service doesn't need you connected.** Closing Termius or shutting down your PC only ends what runs
inside those sessions, such as the app and the tunnel. The scheduler keeps going on its own:

- it runs the cycles and places the orders;
- systemd restarts it within 30 seconds if it crashes (`Restart=always`);
- it starts again by itself if the server reboots.

Reconnecting doesn't trigger anything; you only look at what happened (section 5):

```bash
systemctl --user status claw-desk-scheduler                      # running, and since when
journalctl --user -u claw-desk-scheduler --since "3 hours ago"   # what it logged meanwhile
```

Orders and positions also show on the Alpaca paper dashboard, from any device.

## 5. Watch it

The Streamlit app reads the same state and traces. Run it on the server and reach it from your PC
through an SSH tunnel. On the server it only listens to the machine itself (`.streamlit/config.toml`),
so nothing is exposed. Never open port 8501 to the internet: the app makes Build calls on your key,
and its trace viewer shows everything the agents saw.

With Termius on your PC, let Termius make the tunnel, so the key never leaves it:

1. *Port Forwarding* → new rule → *Local*: local port `8502`, your instance as the host, destination
   `localhost`, destination port `8501`. Start the rule.
2. In a Termius terminal on the server:
   ```bash
   cd ~/claw-agent-desk && .venv/bin/streamlit run ui/streamlit_app.py
   ```
3. Open http://localhost:8502 on your PC.

To keep the app running after you disconnect, start it inside `tmux`:

1. Run `tmux new -s app`, then the `streamlit run` command.
2. Press Ctrl+B then D to leave it running.
3. Later, `tmux attach -t app` gets you back to it.

The port-forward rule is then all you need to reopen it.

With a key file on your PC instead, run `ssh -i <key file> -L 8502:localhost:8501 ubuntu@<public-ip>`
and the same `streamlit run` in that session. The `ubuntu@instance-...` in the server's prompt is the
user and the server's own name, not its address. The public IP is in Termius's host entry or the
Oracle console. Local port 8502 keeps it apart from an app running on your PC (8501). Closing the
session stops the app; the scheduler keeps running.

- **Desk state** lists positions, deferred stocks, decisions and every pass, including LLM calls and
  failures and when the next pass is due. It stays empty until the service's first decision cycle (07:30 ET), because the scheduler
  only writes its state file during a pass. Log-only runs keep theirs in
  `state/desk_state-log-only.json`. To see it, start the app with
  `CLAW_DESK_STATE=state/desk_state-log-only.json` in front of the command.
- **Trace viewer** replays any day's trace. Log-only runs and the service write to the same daily file.
- Don't press **Run** in the server's app while the service runs. It would start a separate desk run on
  the same key, and its calls would mix into the day's trace and the scheduler's call counts.

From any SSH session:

```bash
systemctl --user status claw-desk-scheduler    # is it running, and since when
journalctl --user -u claw-desk-scheduler -f    # one line per pass, with when the next is due; Build retry warnings
```

## 6. Copy the traces to your PC

The analysis happens on your PC; the server only runs the desk. With Termius's SFTP (open the host,
then *SFTP*), download the day's trace and the state into the repo's `logs/server/` folder on your
PC:

- `~/claw-agent-desk/logs/trace-YYYYMMDD.jsonl` (the date is in UTC)
- `~/claw-agent-desk/state/desk_state.json`

`logs/` is gitignored, and the separate folder keeps them apart from your PC's own traces, which have
the same file names. To browse them in your local app, start it from a new PowerShell window:

```powershell
$env:CLAW_DESK_LOG_DIR = "logs\server"; $env:CLAW_DESK_STATE = "logs\server\desk_state.json"
.venv\Scripts\streamlit run ui\streamlit_app.py
```

The traces contain no keys. FRED and Finnhub take their keys in the URL, so error texts are redacted
before they become data gaps or trace lines. They do contain everything the agents saw and said, so
keep them out of git.

## Updating

```bash
cd ~/claw-agent-desk && git pull --ff-only && .venv/bin/python -m pip install -r requirements.txt
systemctl --user restart claw-desk-scheduler
```

Restart between two passes, not during one. A retry pass saves each decision as it goes but sends its
orders at the end, so a restart in the middle loses the orders for buys it already decided. The
passes are in the trace:

```bash
grep -o '"ts": "[^"]*", "agent": "scheduler", "event": "cycle_[a-z]*", "kind": "[a-z]*"' \
  logs/trace-$(date -u +%Y%m%d).jsonl | tail -2   # safe to restart if the last line is a cycle_end
```
