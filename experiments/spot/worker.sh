#!/bin/bash
# Runs at every boot of a spot GPU machine (a systemd service that spot.py's cloud-init installs).
# First boot: fetch code and data from R2 and check them. Every boot: work through this machine's share of the
# phase's jobs. A finished run is skipped; a run cut off by a preemption carries on from its checkpoint.
# Needs /home/zl/spot.env, pushed by spot.py: the bucket-scoped R2 key, R2_S3_ENDPOINT, PHASE, WORKER, WORKERS.
set -uo pipefail
until [ -f /home/zl/spot.env ]; do sleep 10; done
set -a; . /home/zl/spot.env; set +a
export AWS_DEFAULT_REGION=auto HOME=/root PATH=/root/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True UV_PYTHON_PREFERENCE=only-managed  # uv's own Python has the C headers torch.compile builds against
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
r2() { uvx --from awscli aws --endpoint-url "$R2_S3_ENDPOINT" s3 "$@" --only-show-errors; }
S3=s3://zhagaram-data/tokenizer
cd /root
if [ ! -f ready ]; then
  mkdir -p uraikal logs
  r2 cp $S3/code.tgz . && tar xzf code.tgz -C uraikal || exit 1
  for d in bpb bpb-c tamil_tokenizer hf_tokenizers; do r2 sync $S3/$d uraikal/data/$d || exit 1; done
  r2 cp $S3/SHA256SUMS uraikal/data/ && (cd uraikal/data && sha256sum -c --quiet SHA256SUMS) || exit 1
  touch ready
fi
# torch falls back to the CPU without a word if the driver is too old for its CUDA: never pay for an idle GPU
(cd uraikal && uv run --extra tokenizers --extra prep --with torch python -c \
  "import torch; assert torch.cuda.is_available(), 'torch sees no GPU'; f = torch.compile(lambda x: x * 2); assert f(torch.ones(2, device='cuda')).sum() == 4; print(torch.__version__, torch.cuda.get_device_name(), 'compile ok')") \
  > logs/gpu-check.log 2>&1 || { r2 cp logs/gpu-check.log $S3/logs/$PHASE/failed-$WORKER; shutdown -h now; exit 1; }
i=0
while read -r tok seed extra; do  # extra: per-job flags, e.g. --micro 4 for 250k+ vocabularies
  [ $((i++ % WORKERS)) -eq "$WORKER" ] || continue
  base=$(basename "${tok%.json}")
  [[ " $extra " =~ \ --name\ ([^ ]+) ]] && base=${BASH_REMATCH[1]}  # as tokenizer_bpb.py names it
  data=bpb; [[ " $extra " =~ \ --data\ ([^ ]+) ]] && data=${BASH_REMATCH[1]}
  name=$base; [ "$seed" = 0 ] || name=$base-s$seed
  result=uraikal/results/experiments/$data/$name.json weights=uraikal/data/$data/models/$name.pt
  if [ ! -f "$result" ]; then
    (cd uraikal && uv run --extra tokenizers --extra prep --with torch python experiments/tokenizer_bpb.py train "$tok" --seed "$seed" $extra) >> logs/$name.log 2>&1 \
      || { echo "$name" >> logs/failed; r2 cp logs/$name.log $S3/logs/$PHASE/; r2 cp logs/failed $S3/logs/$PHASE/failed-$WORKER; shutdown -h now; exit 1; }  # stop, don't burn through the list
  fi
  [ -f "$result" ] && { r2 cp "$weights" $S3/models/$PHASE/ && r2 cp "$result" $S3/results/$PHASE/ || exit 1; }  # never report done with a result missing from R2; systemd retries in 60 s
  r2 cp logs/$name.log $S3/logs/$PHASE/
done < uraikal/experiments/spot/jobs_$PHASE.txt
echo done | r2 cp - $S3/done/$PHASE/worker-$WORKER
shutdown -h now  # a finished or failed machine powers off, so an idle GPU never bills while the Mac sleeps
