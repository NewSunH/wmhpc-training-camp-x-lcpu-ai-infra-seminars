#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=pdl_launch_probe_%j.log
#SBATCH --error=pdl_launch_probe_%j.err
set -euo pipefail
cd <B300_REPO>/assignment02/team/c1_flashkda/challenge
nvcc -O3 -std=c++17 -arch=sm_103a pdl_launch_probe.cu -o pdl_launch_probe
RESULTS_DIR=../results
LOG_FILE="${RESULTS_DIR}/r18_pdl_launch_probe_${SLURM_JOB_ID}.log"
JSON_FILE="${RESULTS_DIR}/r18_pdl_launch_probe_${SLURM_JOB_ID}.jsonl"
{
  ./pdl_launch_probe 96 128 1048576 2000 1000 5 15
  ./pdl_launch_probe 96 128 1048576 10000 1000 5 15
  ./pdl_launch_probe 96 128 1048576 2000 10000 5 15
  ./pdl_launch_probe 96 128 1048576 10000 10000 5 15
  ./pdl_launch_probe 96 128 262144 2000 1000 5 15
} | tee "${LOG_FILE}"
grep '^{' "${LOG_FILE}" > "${JSON_FILE}"
