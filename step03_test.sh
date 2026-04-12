#!/bin/bash
set -e

SITES="US-Uaf,US-Los,SE-Deg,FI-Sii,US-Twt,FI-Si2,CA-SCB,NZ-Kop,FI-Lom,JP-Mse,JP-BBY,BR-Npw,US-Tw4,US-WPT,US-Myb,US-Tw1,US-OWC"
MODELS="lasso,ann,rf,xgb"
PREDICTORS="fluxgapfill/predictors/temporal.txt,fluxgapfill/predictors/meteorological.txt,fluxgapfill/predictors/knox.txt,fluxgapfill/predictors/all.txt"

echo "[step03] Evaluating models on test set + generating scale.json..."
python -m fluxgapfill.main test \
  --data_dir data \
  --sites "$SITES" \
  --models "$MODELS" \
  --predictors_paths "$PREDICTORS" \
  --split test \
  --distribution laplace \
  --overwrite_results True

echo "[step03] Done. Results: data/results/test.csv"
