#!/usr/bin/env bash
# Offload hw2 training to a remote GPU box, and rebuild the box from scratch.
#
#   ./remote.sh bootstrap        install uv, sync code, sync env, fetch MNIST
#   ./remote.sh push             sync code from this laptop to the box
#   ./remote.sh run NAME CMD...  start CMD in a detached tmux session NAME
#   ./remote.sh status           tmux sessions, GPU, and last log lines
#   ./remote.sh pull             copy outputs/ and results back to this laptop
#   ./remote.sh ssh              open a shell on the box in the project dir
#
# Config (environment variables):
#   HW2_HOST      ssh host alias (default: genai, from ~/.ssh/config)
#   HW2_DIR       project dir on the box, relative to $HOME (default: hw2)
#
# Design notes:
#   - Code moves by rsync, not git: the repo has no git remote.
#   - The box gets its own environment (uv sync on Linux + CUDA), never the
#     macOS .venv. requirements-lock.txt is macOS-only and is not used here.
#   - The box writes its own results/runs.csv. pull saves it as
#     results/runs.remote.csv so you merge rows by hand.
#   - Runs go in tmux, so they survive a dropped ssh connection.
#   - Idempotent: re-running bootstrap on a configured box is safe.

set -euo pipefail

HOST="${HW2_HOST:-genai}"
REMOTE_DIR="${HW2_DIR:-hw2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Remote shells are non-interactive, so ~/.local/bin (where uv goes) is not
# on PATH unless we add it. Every remote command starts with this.
REMOTE_ENV='export PATH="$HOME/.local/bin:$PATH"'

die() { echo "remote.sh: $*" >&2; exit 1; }

# Files that must never travel to the box. Matches .gitignore plus local-only
# tooling and symlinks that would dangle on the remote side.
EXCLUDES=(
  --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ --exclude='*.pyc'
  --exclude=.pytest_cache/ --exclude=.pi/ --exclude=.DS_Store --exclude='*.egg-info/'
  --exclude=data/ --exclude=outputs/ --exclude=results/ --exclude='*.pt'
  --exclude=huffman --exclude=hw2-analyzed --exclude=hw2-vibed --exclude=hw2-ref
  --exclude=curr-school --exclude=old-school --exclude=cloud
)

push() {
  [[ -f "$HERE/uv.lock" ]] || die "uv.lock missing locally; run 'uv lock' first"
  rsync -rpt --delete "${EXCLUDES[@]}" "$HERE/" "$HOST:$REMOTE_DIR/"
}

bootstrap() {
  echo "==> ensuring uv is installed on $HOST"
  ssh "$HOST" bash -s <<'EOF'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version
EOF

  echo "==> pushing code to $HOST:~/$REMOTE_DIR"
  ssh "$HOST" "mkdir -p \"\$HOME/$REMOTE_DIR\"/{outputs,results,logs} && $REMOTE_ENV"
  push

  echo "==> syncing environment (locked, Linux, CUDA wheels from PyPI)"
  ssh "$HOST" "$REMOTE_ENV && cd \"\$HOME/$REMOTE_DIR\" && uv sync --locked"

  echo "==> fetching MNIST (skipped if already present)"
  ssh "$HOST" "$REMOTE_ENV && cd \"\$HOME/$REMOTE_DIR\" && uv run python data.py"

  echo "==> checking CUDA"
  ssh "$HOST" "$REMOTE_ENV && cd \"\$HOME/$REMOTE_DIR\" && uv run python -c \
    'import torch; ok = torch.cuda.is_available(); \
     print(\"torch\", torch.__version__, \"cuda\", ok, \
     torch.cuda.get_device_name(0) if ok else \"\")'"
  echo "==> bootstrap complete"
}

run() {
  [[ $# -ge 2 ]] || die "usage: remote.sh run NAME CMD [ARGS...]"
  local name="$1"; shift
  [[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || die "NAME must be letters, digits, _ or -"

  # Quote each argument so the command reaches the box exactly as typed.
  local cmd
  printf -v cmd '%q ' "$@"

  # Build the runner script here, then send it over stdin. Nothing is
  # expanded on the box except what the runner itself uses.
  local runner
  runner="$(cat <<RUNNER
set -uo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
cd "\$HOME/$REMOTE_DIR"
echo "start: \$(date -Iseconds)"
uv run $cmd
rc=\$?
echo "exit: \$rc \$(date -Iseconds)"
RUNNER
)"

  local logdir="\$HOME/$REMOTE_DIR/logs"
  ssh "$HOST" "$REMOTE_ENV && mkdir -p \"$logdir\" && cat > \"$logdir/$name.sh\"" <<<"$runner"

  ssh "$HOST" "if tmux has-session -t $name 2>/dev/null; then echo \"tmux session $name already exists\" >&2; exit 1; fi; \
    tmux new-session -d -s $name 'bash $logdir/$name.sh 2>&1 | tee -a $logdir/$name.log; exec bash' && echo \"started tmux session $name (log: $REMOTE_DIR/logs/$name.log)\""
}

status() {
  ssh "$HOST" bash -s "$REMOTE_DIR" <<'EOF'
set -uo pipefail
dir="$1"
echo "== tmux"; tmux ls 2>/dev/null || echo "(no sessions)"
echo "== gpu"; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(no gpu)"
echo "== logs"
for f in "$HOME/$dir"/logs/*.log; do
  [[ -e "$f" ]] || { echo "(no logs)"; break; }
  echo "-- $(basename "$f")"; tail -n 5 "$f"
done
EOF
}

pull() {
  echo "==> pulling outputs/ from $HOST"
  mkdir -p "$HERE/outputs"
  rsync -rpt "$HOST:$REMOTE_DIR/outputs/" "$HERE/outputs/"
  echo "==> pulling results/runs.csv as runs.remote.csv (merge by hand)"
  mkdir -p "$HERE/results"
  rsync -pt "$HOST:$REMOTE_DIR/results/runs.csv" "$HERE/results/runs.remote.csv" \
    || echo "(no remote runs.csv yet)"
}

cmd="${1:-}"; [[ $# -gt 0 ]] && shift
case "$cmd" in
  bootstrap) bootstrap ;;
  push)      push ;;
  run)       run "$@" ;;
  status)    status ;;
  pull)      pull ;;
  ssh)       exec ssh -t "$HOST" "cd \"\$HOME/$REMOTE_DIR\" && exec \$SHELL -l" ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
