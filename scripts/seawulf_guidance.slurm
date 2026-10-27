#!/bin/bash
#SBATCH --job-name=guidance
#SBATCH --partition=a100-large
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# Serves an open model with vLLM across 4 A100s, then runs the extraction against it.
# Usage:  sbatch scripts/seawulf_guidance.slurm
#         MODEL=Qwen/Qwen2.5-72B-Instruct sbatch scripts/seawulf_guidance.slurm
#         ARGS="--sample data/labels/labels.csv" sbatch scripts/seawulf_guidance.slurm
# Finished releases are cached, so resubmitting after a timeout continues the run.

set -euo pipefail
MODEL=${MODEL:-meta-llama/Llama-3.3-70B-Instruct}
ARGS=${ARGS:-}
export HF_HOME=${HF_HOME:-/gpfs/scratch/$USER/hf}   # model weights (~140 GB) go to scratch, not home

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

vllm serve "$MODEL" --tensor-parallel-size 4 --max-model-len 16384 --port 8000 \
  > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

until curl -sf localhost:8000/health > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "vLLM exited; see logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  sleep 15
done
echo "server ready: $MODEL"

nowcast guidance --base-url http://localhost:8000/v1 --model "$MODEL" --workers 32 $ARGS
