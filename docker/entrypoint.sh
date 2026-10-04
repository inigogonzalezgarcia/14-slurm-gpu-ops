#!/bin/bash
# entrypoint.sh slurmdbd|slurmctld|slurmd
set -euo pipefail
role=${1:?role}

wait_port() {  # wait_port host port
  for _ in $(seq 120); do
    (exec 3<>"/dev/tcp/$1/$2") 2>/dev/null && return 0
    sleep 1
  done
  echo "timeout waiting for $1:$2" >&2
  exit 1
}

runuser -u munge -- /usr/sbin/munged
mkdir -p /shared/runs /shared/gpu-health /shared/logs /var/lib/gpu-sim

case "$role" in
  slurmdbd)
    wait_port mariadb 3306
    exec /usr/sbin/slurmdbd -D -v
    ;;
  slurmctld)
    wait_port slurmdbd 6819
    # Register the cluster and an account for the lab's jobs (idempotent).
    sacctmgr -i add cluster lab >/dev/null 2>&1 || true
    sacctmgr -i add account lab Cluster=lab Description="lab jobs" >/dev/null 2>&1 || true
    sacctmgr -i add user root Account=lab DefaultAccount=lab >/dev/null 2>&1 || true
    exec /usr/sbin/slurmctld -D -v
    ;;
  slurmd)
    # cgroup v2 without systemd (see etc/slurm/cgroup.conf). slurmd puts its steps under
    # system.slice/<node>_slurmstepd.scope and expects the controllers to be delegated down to
    # it. A cgroup with processes cannot delegate controllers, so move every process of the
    # container out of the root into a leaf first (the usual trick for cgroups in containers).
    if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
      mkdir -p /sys/fs/cgroup/init /sys/fs/cgroup/system.slice
      xargs -rn1 < /sys/fs/cgroup/cgroup.procs > /sys/fs/cgroup/init/cgroup.procs 2>/dev/null || true
      read -ra controllers < /sys/fs/cgroup/cgroup.controllers
      for c in "${controllers[@]}"; do
        echo "+$c" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null || true
        echo "+$c" > /sys/fs/cgroup/system.slice/cgroup.subtree_control 2>/dev/null || true
      done
    fi
    # Four "GPUs": character devices with no driver behind them (Slurm only checks the file type).
    for i in 0 1 2 3; do
      [ -e "/dev/labgpu$i" ] || mknod -m 666 "/dev/labgpu$i" c 1 3
    done
    wait_port slurmctld 6817
    exec /usr/sbin/slurmd -D -v
    ;;
  *)
    exec "$@"
    ;;
esac
