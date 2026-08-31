#!/bin/bash
# WP1b orchestrator: wait for an uncontended GPU, then run the full sweep +
# analysis on the d077902 worktree. Launched with Bash run_in_background;
# notifies on exit. Progress -> orchestrate.log (watched by a persistent Monitor).
set -u
JOB="$CLAUDE_JOB_DIR/tmp/wp1b"
LOG="$JOB/orchestrate.log"
ROUGH_PID=636653          # the Unitree-H1_2-Rough training to wait out
MEM_FREE_MIB=2500         # GPU counts as free below this
STABLE_NEEDED=3           # consecutive 60s polls that must all be free
MAX_WAIT_H=48

exec >>"$LOG" 2>&1
echo "=== orchestrator start $(date '+%F %T') ==="
echo "waiting for: pid $ROUGH_PID gone AND gpu mem < ${MEM_FREE_MIB}MiB AND no python compute-app, stable ${STABLE_NEEDED}min"

deadline=$(( $(date +%s) + MAX_WAIT_H*3600 ))
stable=0
while :; do
  now=$(date +%s); [ "$now" -ge "$deadline" ] && { echo "TIMEOUT after ${MAX_WAIT_H}h waiting for GPU"; exit 3; }
  # "rough alive" only if the pid exists AND still looks like a train.py job
  rough_alive=0
  if kill -0 "$ROUGH_PID" 2>/dev/null; then
    tr -d '\0' < "/proc/$ROUGH_PID/cmdline" 2>/dev/null | grep -q "train.py" && rough_alive=1
  fi
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  napps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -ci python || true)
  if [ "$rough_alive" -eq 0 ] && [ "${mem:-99999}" -lt "$MEM_FREE_MIB" ] && [ "${napps:-1}" -eq 0 ]; then
    stable=$((stable+1))
    echo "$(date '+%T') free (mem=${mem}MiB, python-apps=${napps})  stable ${stable}/${STABLE_NEEDED}"
    [ "$stable" -ge "$STABLE_NEEDED" ] && break
  else
    [ "$stable" -ne 0 ] && echo "$(date '+%T') busy again (rough=${rough_alive} mem=${mem}MiB python-apps=${napps}) -> reset"
    stable=0
    # quieter heartbeat: only every 10 min
    [ $(( now % 600 )) -lt 60 ] && echo "$(date '+%T') still waiting (rough=${rough_alive} mem=${mem}MiB python-apps=${napps})"
  fi
  sleep 60
done

echo "=== GPU FREE $(date '+%F %T') -- launching sweep ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitree_mjlab_h1_2_rl
PY=$(which python)
echo "python: $PY"

"$PY" "$JOB/run_sweep_final.py"; rc=$?
echo "run_sweep_final.py exit $rc"
if [ "$rc" -ne 0 ]; then echo "=== SWEEP FAILED (rc=$rc) -- stopping ==="; exit "$rc"; fi

echo "=== sweep done -- analysis $(date '+%T') ==="
cd "$CLAUDE_JOB_DIR/tmp/wt_v3"
"$PY" "$JOB/analyze.py" > "$JOB/analysis_output.txt" 2>&1; echo "analyze.py exit $?"
"$PY" "$JOB/make_figures.py" >> "$JOB/analysis_output.txt" 2>&1; echo "make_figures.py exit $?"

touch "$JOB/ORCHESTRATOR_DONE"
echo "=== ORCHESTRATOR DONE $(date '+%F %T') ==="
