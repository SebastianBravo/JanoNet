#!/bin/bash
# Ejemplo genérico de entrenamiento en un clúster con Slurm. Ajusta la partición, el
# entorno y las rutas a tu clúster: aquí no hay nada específico del HPC donde se
# entrenaron los modelos publicados.
#
#   sbatch docs/slurm_example.sh rs_ca_tr
#
# El entrenamiento guarda checkpoints cada 500 pasos y al final de cada época, y atiende
# SIGTERM: si Slurm corta el trabajo por tiempo, basta con reenviarlo y retoma solo
# (--resume auto). --signal pide a Slurm que avise 5 minutos antes del límite.

#SBATCH --job-name=janonet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=TERM@300
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
MODEL="${1:-rs_ca_tr}"

# Entorno (conda u otro) con `pip install -e ".[train]"` y torch con CUDA.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate janonet

# La caché de 32 cuadros (≈ 423 GB) conviene tenerla en el disco local del nodo; los
# checkpoints, en un sistema de ficheros persistente.
export JANONET_CACHE="${JANONET_CACHE:-/ruta/a/frames_cache}"
RUN_DIR="${RUN_DIR:-runs/${MODEL}}"

srun python scripts/03_train.py --model "${MODEL}" --run-dir "${RUN_DIR}" --num-workers "${SLURM_CPUS_PER_TASK}"
