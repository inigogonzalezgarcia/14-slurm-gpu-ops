#!/usr/bin/env bash
# End-to-end test against the running lab cluster (docker compose up -d first).
# Every command runs on the controller container, the way an operator would run it.
set -euo pipefail

RUN=${RUN-docker compose exec -T slurmctld}   # RUN= (empty) runs against a local Slurm
SHARED=${SHARED:-/shared}
JOBS=${JOBS:-/opt/gpuops/jobs}
OUT=${OUT:-e2e-out}
mkdir -p "$OUT"

pass=0
ok()   { pass=$((pass + 1)); echo "  ok  $*"; }
fail() { echo "FAIL  $*" >&2; $RUN sinfo -N -o "%N %T %E" >&2 || true; exit 1; }
c()    { $RUN "$@"; }
state()  { c sinfo -h -N -n "$1" -o "%T"; }
reason() { c sinfo -h -N -n "$1" -o "%E"; }
wait_for() {  # wait_for <seconds> <description> <command...>
  local t=$1 what=$2; shift 2
  for _ in $(seq "$t"); do "$@" >/dev/null 2>&1 && { ok "$what"; return 0; }; sleep 1; done
  fail "$what (after ${t}s)"
}
expect() {    # expect <description> <command...>
  local what=$1; shift
  if "$@"; then ok "$what"; else fail "$what"; fi
}
eq()           { [[ "$1" == "$2" ]]; }
is_state()     { [[ "$(state "$1")" == "$2" ]]; }
reason_has()   { [[ "$(reason "$1")" == *"$2"* ]]; }
job_state()    { c squeue -h -j "$1" -o "%T" 2>/dev/null || true; }
job_nodes()    { c squeue -h -j "$1" -o "%N"; }
job_running()  { [[ "$(job_state "$1")" == RUNNING ]]; }
job_gone()     { [[ -z "$(job_state "$1")" ]]; }
checkpoints()  { c cat "$SHARED/runs/$1/events.jsonl" 2>/dev/null | grep -c checkpoint_end || true; }
ckpt_after()   { (( $(checkpoints "$1") >= $2 )); }
expand()       { c scontrol show hostnames "$1"; }
not_in()       { local x=$1; shift; for y in "$@"; do [[ "$x" != "$y" ]] || return 1; done; }
no_reservations() { ! c scontrol show res -o | grep -q ReservationName; }

echo "== 1. cluster and GPUs"
wait_for 120 "4 nodes idle" bash -c "[[ \$($RUN sinfo -h -N -t idle -o %N | wc -l) == 4 ]]"
expect "every node has gres gpu:lab:4" eq "$(c sinfo -h -N -o %G | sort -u)" "gpu:lab:4"
v=$(c srun -N1 --gres=gpu:4 printenv CUDA_VISIBLE_DEVICES)
expect "a job asking for 4 GPUs gets CUDA_VISIBLE_DEVICES=$v" eq "$v" "0,1,2,3"
v=$(c srun -N1 --gres=gpu:lab:2 printenv CUDA_VISIBLE_DEVICES)
expect "a job asking for 2 GPUs gets two of them ($v)" eq "$v" "0,1"

echo "== 2. application XIDs do not drain, unknown ones are only recorded"
c gpuops sim inject gpu-4 --gpu 2 --xid 13
c gpuops sim inject gpu-4 --gpu 3 --xid 94
wait_for 30 "XID 94 recorded as 'watch' in the health log" c grep -q '"action": "watch"' "$SHARED/gpu-health/gpu-4.jsonl"
expect "gpu-4 still idle after XID 13 and XID 94" is_state gpu-4 idle
c gpuops sim repair gpu-4

echo "== 3. a node drained by an admin keeps the admin's reason"
c scontrol update NodeName=gpu-1 State=DRAIN Reason="maintenance window"
c gpuops sim inject gpu-1 --gpu 0 --xid 79
wait_for 30 "health check noticed the fault on gpu-1" c grep -q "already drained by someone else" "$SHARED/gpu-health/gpu-1.jsonl"
expect "reason unchanged: maintenance window" reason_has gpu-1 "maintenance window"
c gpuops sim repair gpu-1
c scontrol update NodeName=gpu-1 State=RESUME
wait_for 10 "gpu-1 back to idle" is_state gpu-1 idle

echo "== 4. training job: two GPU faults, two requeues, finishes from its checkpoints"
J=$(c sbatch --parsable --output="$SHARED/runs/%x-%j.out" "$JOBS/train.sbatch")
echo "  job $J"
wait_for 60 "job running" job_running "$J"
wait_for 60 "first checkpoint written" ckpt_after "$J" 1
mapfile -t N1 < <(expand "$(job_nodes "$J")")
c gpuops sim inject "${N1[1]}" --gpu 2 --xid 79
wait_for 30 "${N1[1]} drained for XID 79" reason_has "${N1[1]}" "gpu-health:drain XID 79 on GPU 2"
wait_for 30 "job requeued" c grep -q "\"requeued\": \[\"$J\"\]" "$SHARED/gpu-health/${N1[1]}.jsonl"
wait_for 120 "job running again (after the requeue delay)" job_running "$J"
mapfile -t N2 < <(expand "$(job_nodes "$J")")
expect "restarted on ${N2[*]}, not on the drained node" not_in "${N1[1]}" "${N2[@]}"
before=$(checkpoints "$J")
wait_for 60 "a checkpoint written after the restart" ckpt_after "$J" $((before + 1))
c gpuops sim inject "${N2[0]}" --gpu 1 --row-remap-failure --xid 64
wait_for 30 "${N2[0]} quarantined for XID 64" reason_has "${N2[0]}" "gpu-health:quarantine"
wait_for 120 "job running a third time" job_running "$J"
wait_for 240 "job finished" job_gone "$J"
expect "all 150 steps done" c grep -q '"event": "done", "step": 150' "$SHARED/runs/$J/events.jsonl"
if [[ -z "${NO_ACCOUNTING:-}" ]]; then
c env SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S TZ=UTC sacct -j "$J" -D -X -P -n -o JobIDRaw,JobName,State,Start,End,NNodes,NodeList,AllocTRES,ExitCode | tee "$OUT/sacct.txt"
expect "sacct keeps all three runs" eq "$(cut -d'|' -f3 "$OUT/sacct.txt" | tr '\n' ' ')" "REQUEUED REQUEUED COMPLETED "

echo "== 5. goodput from sacct, with repo 12's analyzer"
c gpuops events "$J" > "$OUT/events.jsonl"
cat "$OUT/events.jsonl"
c cat "$SHARED/runs/$J/events.jsonl" > "$OUT/job-events.jsonl"
c bash -c "cat $SHARED/gpu-health/*.jsonl" > "$OUT/gpu-health.jsonl"
if [[ -n "${GOODPUT_REPO:-}" ]]; then
  (cd "$GOODPUT_REPO" && python3 -m goodput analyze "$OLDPWD/$OUT/events.jsonl") | tee "$OUT/goodput.txt"
  (cd "$GOODPUT_REPO" && python3 -m goodput analyze --format json "$OLDPWD/$OUT/events.jsonl") > "$OUT/goodput.json"
  expect "repo 12: 2 interruptions, causes gpu_xid79 and gpu_xid64" python3 - "$OUT/goodput.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["interruptions"] == 2, d["interruptions"]
assert set(d["by_cause"]) == {"gpu_xid79", "gpu_xid64"}, d["by_cause"]
assert 0 < d["goodput"] < 1
EOF
fi
fi

echo "== 6. validation puts repaired nodes back, and only them"
set +e
c gpuops validate "${N2[0]}"; rc=$?
set -e
expect "quarantined ${N2[0]} refused (exit $rc)" eq "$rc" 3
expect "${N2[0]} still drained" is_state "${N2[0]}" drained
set +e
c gpuops validate "${N1[1]}" --burnin-seconds 5; rc=$?
set -e
expect "${N1[1]} not repaired yet: validation fails (exit $rc)" eq "$rc" 1
wait_for 30 "${N1[1]} drained again" is_state "${N1[1]}" drained
c gpuops sim repair "${N1[1]}"
expect "${N1[1]} repaired and validated" c gpuops validate "${N1[1]}" --burnin-seconds 5
expect "${N1[1]} idle" is_state "${N1[1]}" idle
c gpuops sim repair "${N2[0]}"
set +e
c gpuops validate "${N2[0]}" --burnin-seconds 5 --force; rc=$?
set -e
expect "a GPU reset does not fix a row remap failure: validation fails (exit $rc)" eq "$rc" 1
c gpuops sim replace "${N2[0]}"
wait_for 30 "${N2[0]} drained before the replacement validation" is_state "${N2[0]}" drained
expect "${N2[0]} validated after the board was replaced" c gpuops validate "${N2[0]}" --burnin-seconds 5 --force
expect "no reservation left behind" no_reservations

echo "== 7. too hot: cordon, no requeue, no automatic return"
c gpuops sim inject gpu-3 --gpu 0 --temp 93
wait_for 30 "gpu-3 cordoned" reason_has gpu-3 "gpu-health:cordon temperature on GPU 0"
c gpuops sim repair gpu-3
sleep 15
expect "cooled down, still drained: the health check never resumes a node" is_state gpu-3 drained
expect "gpu-3 validated" c gpuops validate gpu-3 --burnin-seconds 5

echo "== 8. end state"
wait_for 30 "all 4 nodes idle" bash -c "[[ \$($RUN sinfo -h -N -t idle -o %N | wc -l) == 4 ]]"
c gpuops status | tee "$OUT/status.txt"
echo "PASSED: $pass checks"
