#!/bin/bash
set -e

SITES="US-Uaf,US-Los,SE-Deg,FI-Sii,US-Twt,FI-Si2,CA-SCB,NZ-Kop,FI-Lom,JP-Mse,JP-BBY,BR-Npw,US-Tw4,US-WPT,US-Myb,US-Tw1,US-OWC"
MODELS="lasso,ann,rf,xgb"
PREDICTORS="fluxgapfill/predictors/temporal.txt,fluxgapfill/predictors/meteorological.txt,fluxgapfill/predictors/knox.txt,fluxgapfill/predictors/all.txt"

echo "[step02] Training models (4 models x 4 predictor sets x 17 sites x 10 folds)..."
echo "         This may take 1-2 days. Consider running in tmux/screen."
python -m fluxgapfill.main train \
  --data_dir data \
  --sites "$SITES" \
  --models "$MODELS" \
  --predictors_paths "$PREDICTORS" \
  --overwrite_existing_models True

echo "[step02] Done."
