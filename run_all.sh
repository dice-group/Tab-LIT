#!/bin/bash
# Every result in the README, from scratch: the backbones on the shipped splits,
# the MrAP rerun, the language model's answers, the answer-column runs, then the
# tables.
#
#   MRAP_DIR=/path/to/literal-embeddings/MrAP ./run_all.sh
#
# BACKBONES narrows the main panel. Existing result files are skipped, so a run
# that stops can be restarted. On one RTX 3090, TabPFN-3.5, TabICLv2,
# Kumo-Tabular and GBT take about 25 minutes together; Nori-1.5 takes hours,
# most of it on YAGO15k, whose largest attributes fall back to the CPU. The
# llm/ answer files ship with the package; deleting them makes the llm step ask
# Qwen3-8B again (a 16 GB download and about an hour of GPU time).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
BACKBONES="${BACKBONES:-tabpfn tabicl kumo nori}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" \
       MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

[ -f llm/DB15K_answers.csv ] && [ -f results/DB15K_llm_direct.csv ] || "$PYTHON" -u -m src.main llm

"$PYTHON" -u -m src.main experiment --backbones $BACKBONES gbt --conversion full
"$PYTHON" -u -m src.main experiment --backbones kumo tabpfn gbt --conversion full_llm

: "${MRAP_DIR:?set MRAP_DIR to a literal-embeddings/MrAP checkout (paper-reproduction branch)}"
[ -f results/DB15K_mrap.csv ] || "$PYTHON" -u -m src.main mrap

"$PYTHON" -u -m src.main report --backbones $BACKBONES --readme
