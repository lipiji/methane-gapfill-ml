#!/bin/bash
set -e

echo "========================================"
echo " FluxGapfill Pipeline - Irvin et al. 2021"
echo "========================================"

bash step01_preprocess.sh
bash step02_train.sh
bash step03_test.sh
bash step04_gapfill.sh

echo "========================================"
echo " All steps completed."
echo " Results: data/results/test.csv"
echo " Gapfilled: data/{SiteID}/gapfilled/"
echo "========================================"
