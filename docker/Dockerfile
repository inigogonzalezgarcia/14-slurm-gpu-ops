# One image for every role: slurmdbd, slurmctld and the GPU nodes (slurmd).
# Slurm and MUNGE from the Ubuntu 24.04 archive (Slurm 23.11).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends slurm-wlm slurmdbd munge python3 \
 && rm -rf /var/lib/apt/lists/*

# Lab shortcut: the MUNGE key is created once, in the image, so every container shares it.
# In a real cluster the key is generated once and distributed securely, never baked into an image.
RUN test -s /etc/munge/munge.key || mungekey --create --keyfile=/etc/munge/munge.key \
 && chown munge:munge /etc/munge/munge.key && chmod 400 /etc/munge/munge.key \
 && mkdir -p /run/munge /var/spool/slurmctld /var/spool/slurmd /var/log/slurm \
 && chown munge:munge /run/munge && chown slurm:slurm /var/spool/slurmctld /var/log/slurm

COPY etc/slurm/ /etc/slurm/
RUN chown slurm:slurm /etc/slurm/slurmdbd.conf && chmod 600 /etc/slurm/slurmdbd.conf

COPY gpuops/ /opt/gpuops/gpuops/
COPY jobs/ /opt/gpuops/jobs/
COPY bin/gpu-health bin/gpuops /usr/local/sbin/
COPY docker/gpu-health.env /etc/default/gpu-health
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 755 /usr/local/sbin/gpu-health /usr/local/sbin/gpuops /usr/local/bin/entrypoint.sh \
 && ln -s /usr/local/sbin/gpuops /usr/local/bin/gpuops
ENV PYTHONPATH=/opt/gpuops

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
