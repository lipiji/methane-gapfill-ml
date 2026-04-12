#!/bin/bash
set -e

SITES="US-Uaf,US-Los,SE-Deg,FI-Sii,US-Twt,FI-Si2,CA-SCB,NZ-Kop,FI-Lom,JP-Mse,JP-BBY,BR-Npw,US-Tw4,US-WPT,US-Myb,US-Tw1,US-OWC"

echo "[step01] Preprocessing 17 sites..."
python -m fluxgapfill.main preprocess \
  --data_dir data \
  --sites "$SITES" \
  --n_train 10 \
  --eval_frac 0.1 \
  --n_mc 100

echo "[step01] Done."
