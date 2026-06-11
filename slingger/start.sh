#!/bin/bash
# =============================================================================
# slingger/start.sh — Slingger V6 Persistent Daemon Startup Script
# =============================================================================
# This script is the Railway container entrypoint.
# It runs on EVERY container restart — must be idempotent and fast.
#
# Execution order:
#   1. Health check: verify Python and dependencies
#   2. Directory setup: ensure logs/ and models/candidates/ exist
#   3. Model check: v6_production.pkl present? If not, attempt auto-train.
#   4. Daemon launch: auto_resolver + auto_retrain in background
#   5. Main bot: python main.py --mode dry-run (FOREGROUND — Railway PID 1 child)
#
# Architecture note:
#   Railway monitors the FOREGROUND process (main.py).
#   Daemon PIDs are children of this bash script.
#   On SIGTERM, bash will send SIGTERM to all child PIDs via trap.
# =============================================================================

set -uo pipefail

# ── Resolve paths ─────────────────────────────────────────────
SLINGGER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SLINGGER_DIR")"

MODELS_DIR="$SLINGGER_DIR/models"
CANDIDATES_DIR="$SLINGGER_DIR/models/candidates"
LOGS_DIR="$SLINGGER_DIR/logs"
DATA_DIR="$SLINGGER_DIR/data"
PRODUCTION_MODEL="$MODELS_DIR/v6_production.pkl"
LOG_FILE="$LOGS_DIR/startup.log"

# ── Timestamp helper ──────────────────────────────────────────
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

# ── Logging helper ────────────────────────────────────────────
log() {
    local level="$1"
    local msg="$2"
    local line="$(ts) [${level}] [start.sh] ${msg}"
    echo "$line"
    # Also append to startup log (best effort)
    echo "$line" >> "$LOG_FILE" 2>/dev/null || true
}

# ── Trap: graceful shutdown on SIGTERM/SIGINT ─────────────────
# When Railway sends SIGTERM, propagate to all child daemons
_DAEMON_PIDS=()
cleanup() {
    log "INFO" "SIGTERM received — shutting down daemons..."
    for pid in "${_DAEMON_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log "INFO" "Sending SIGTERM to PID $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    wait "${_DAEMON_PIDS[@]}" 2>/dev/null || true
    log "INFO" "All daemon processes stopped."
}
trap cleanup SIGTERM SIGINT

# =============================================================================
# STEP 1 — Directory setup
# =============================================================================
log "INFO" "=== Slingger V6 Container Startup ==="
log "INFO" "SLINGGER_DIR : $SLINGGER_DIR"
log "INFO" "PROJECT_ROOT : $PROJECT_ROOT"

mkdir -p "$LOGS_DIR" "$MODELS_DIR" "$CANDIDATES_DIR" "$DATA_DIR"
log "INFO" "Directories ready."

# =============================================================================
# STEP 2 — Python environment check
# =============================================================================
log "INFO" "Python: $(python --version 2>&1)"

# =============================================================================
# STEP 3 — Model seeding
# =============================================================================
if [ -f "$PRODUCTION_MODEL" ]; then
    log "INFO" "v6_production.pkl found — skipping auto-train."
    # Quick sanity: check the meta file too
    if [ -f "$MODELS_DIR/v6_production_meta.json" ]; then
        OOF_AUC=$(python -c "
import json
with open('$MODELS_DIR/v6_production_meta.json') as f:
    m = json.load(f)
print(m.get('oof_auc_mean', 'N/A'))
" 2>/dev/null || echo "N/A")
        log "INFO" "Production model AUC: $OOF_AUC"
    fi
else
    log "WARN" "v6_production.pkl NOT found. Attempting auto-train..."

    DATASET="$DATA_DIR/dry_run_shadow_2026-06-06_154404_resolved.csv"
    if [ -f "$DATASET" ]; then
        log "INFO" "Training dataset found: $DATASET"
        log "INFO" "Running train_v6_mvp.py..."
        cd "$SLINGGER_DIR" && python train_v6_mvp.py
        if [ $? -eq 0 ]; then
            log "INFO" "Auto-train successful. v6_production.pkl created."
        else
            log "ERROR" "Auto-train FAILED. Daemons will start but auto_retrain will wait for data."
        fi
    else
        log "WARN" "Training dataset not found at: $DATASET"
        log "WARN" "Daemons will start in standby mode (no model yet)."
    fi
fi

# =============================================================================
# STEP 4 — Launch daemons in background
# =============================================================================
log "INFO" "Launching auto_resolver daemon..."
cd "$SLINGGER_DIR" && python scripts/auto_resolver.py --daemon >> "$LOGS_DIR/resolver_stdout.log" 2>&1 &
RESOLVER_PID=$!
_DAEMON_PIDS+=($RESOLVER_PID)
log "INFO" "auto_resolver started (PID: $RESOLVER_PID)"

log "INFO" "Launching auto_retrain daemon..."
cd "$SLINGGER_DIR" && python scripts/auto_retrain.py --daemon >> "$LOGS_DIR/retrain_stdout.log" 2>&1 &
RETRAIN_PID=$!
_DAEMON_PIDS+=($RETRAIN_PID)
log "INFO" "auto_retrain started (PID: $RETRAIN_PID)"

# Brief pause to let daemons initialize before main.py starts
sleep 2

# Verify daemons are still alive (not immediately crashed)
for pid in "${_DAEMON_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        log "INFO" "Daemon PID $pid is alive."
    else
        log "WARN" "Daemon PID $pid has already exited. Check daemon logs."
    fi
done

# =============================================================================
# STEP 5 — Launch main bot (FOREGROUND)
# =============================================================================
log "INFO" "=== Starting main Slingger engine (foreground) ==="
log "INFO" "Command: python main.py --mode dry-run"
log "INFO" "=================================================="

cd "$PROJECT_ROOT"
# exec replaces this bash process — SIGTERM goes directly to main.py
# Daemons are already detached with their own PIDs
exec python main.py --mode dry-run
