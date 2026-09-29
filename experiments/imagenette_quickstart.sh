#!/bin/bash
# Minimal CPU-friendly run: train one Matryoshka SAE on CLIP-B/32 final image embeddings
# (post_projection, -1) using Imagenette, then visualize the top-activating images per neuron.
# Usage: bash experiments/imagenette_quickstart.sh   (run from anywhere; DATA_ROOT defaults to ~/data)
# Training is tracked in Weights & Biases (run `wandb login` once). WANDB_MODE=offline or disabled to skip.

set -euo pipefail
export TQDM_DYNAMIC_NCOLS=1
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-$HOME/data}"
DATASET_PATH="${DATA_ROOT}/imagenette2-160"
MODEL_NAME="clip-vit-base-patch32"
POINT="post_projection"
LAYER="-1"
EXPANSION_FACTOR=8
K=20
STEPS=2000
GROUP_FRACTIONS=(0.0625 0.125 0.25 0.5625)
NUM_WORKERS="${NUM_WORKERS:-2}"
ACT_BATCH_SIZE="${ACT_BATCH_SIZE:-64}"
WANDB_PROJECT="${WANDB_PROJECT:-sae-for-vlm}"

RAW_DIR="./activations_dir/raw/imagenette"
SAE_ACTS_DIR="./activations_dir/matroyshka_batch_top_k_${K}_x${EXPANSION_FACTOR}/imagenette_train"
CKPT_DIR="./checkpoints_dir/imagenette"
SAE_PATH="${CKPT_DIR}/train_matroyshka_batch_top_k_${K}_x${EXPANSION_FACTOR}/trainer_0/ae.pt"

# A step is skipped only if its output dir has a .done marker (written after the step succeeds)
is_done() { [ -f "$1/.done" ]; }
mark_done() { touch "$1/.done"; }
reset_dir() { rm -rf "$1"; mkdir -p "$1"; }

# 0. Download Imagenette (160px, ~100MB): train/ and val/ with one folder per class
if [ ! -d "${DATASET_PATH}" ]; then
  mkdir -p "${DATA_ROOT}"
  wget -q --show-progress -O "${DATA_ROOT}/imagenette2-160.tgz" \
    https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
  tar xzf "${DATA_ROOT}/imagenette2-160.tgz" -C "${DATA_ROOT}"
  rm "${DATA_ROOT}/imagenette2-160.tgz"
fi

# 1. Save original activations (one vector per image)
# "inat" just means a plain ImageFolder at <data_path>/<split>
for SPLIT in "train" "val"; do
  if ! is_done "${RAW_DIR}/${SPLIT}"; then
    reset_dir "${RAW_DIR}/${SPLIT}"
    python save_activations.py \
      --batch_size "${ACT_BATCH_SIZE}" \
      --model_name "${MODEL_NAME}" \
      --attachment_point "${POINT}" \
      --layer "${LAYER}" \
      --dataset_name "inat" \
      --split "${SPLIT}" \
      --data_path "${DATASET_PATH}" \
      --num_workers "${NUM_WORKERS}" \
      --output_dir "${RAW_DIR}/${SPLIT}" \
      --save_every 1000
    mark_done "${RAW_DIR}/${SPLIT}"
  fi
done

# 2. Train SAE (trainer warmup is 1000 steps, so decay_start must be in (1000, STEPS))
if [ ! -f "${SAE_PATH}" ]; then
  python sae_train.py \
    --sae_model "matroyshka_batch_top_k" \
    --activations_dir "${RAW_DIR}/train" \
    --val_activations_dir "${RAW_DIR}/val" \
    --checkpoints_dir "${CKPT_DIR}" \
    --expansion_factor "${EXPANSION_FACTOR}" \
    --steps "${STEPS}" \
    --save_steps 1000 \
    --log_steps 50 \
    --batch_size 1024 \
    --k "${K}" \
    --auxk_alpha 0.03 \
    --decay_start $((STEPS - 1)) \
    --group_fractions "${GROUP_FRACTIONS[@]}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_name "imagenette_${MODEL_NAME}_${POINT}_matryoshka_x${EXPANSION_FACTOR}_k${K}"
fi

# 3. Save SAE activations on the train split
if ! is_done "${SAE_ACTS_DIR}"; then
  reset_dir "${SAE_ACTS_DIR}"
  python save_activations.py \
    --batch_size "${ACT_BATCH_SIZE}" \
    --model_name "${MODEL_NAME}" \
    --attachment_point "${POINT}" \
    --layer "${LAYER}" \
    --dataset_name "inat" \
    --split "train" \
    --data_path "${DATASET_PATH}" \
    --num_workers "${NUM_WORKERS}" \
    --output_dir "${SAE_ACTS_DIR}" \
    --save_every 1000 \
    --sae_model "matroyshka_batch_top_k" \
    --sae_path "${SAE_PATH}"
  mark_done "${SAE_ACTS_DIR}"
fi

# 4. Top-16 activating images per neuron, saved as grids under ${SAE_ACTS_DIR}/tree/
python find_hai_indices.py \
  --activations_dir "${SAE_ACTS_DIR}" \
  --dataset_name "inat" \
  --data_path "${DATASET_PATH}" \
  --split "train" \
  --k 16 \
  --chunk_size 1000

python visualize_neurons.py \
  --output_dir "${SAE_ACTS_DIR}" \
  --top_k 16 \
  --dataset_name "inat" \
  --data_path "${DATASET_PATH}" \
  --split "train" \
  --group_fractions "${GROUP_FRACTIONS[@]}" \
  --hai_indices_path "${SAE_ACTS_DIR}/hai_indices_16.npy"

echo "Done. Neuron grids: ${SAE_ACTS_DIR}/tree/"
