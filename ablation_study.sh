#!/bin/bash

# Ablation study script for LSH parameter tuning
# Tests different combinations of k (hash functions) and bands with multiple seeds

set -e

SEEDS=(1 5 42 112 1011)
INPUT_FILE="data/user_movie_rating.npy"
THRESHOLD=0.5

# Define k and bands combinations to test
# Format: "k:bands"
COMBINATIONS=(
    "60:10"
    "60:12"
    "60:15"
    "60:20"
    "80:10"
    "80:16"
    "80:20"
    "100:10"
    "100:20"
    "100:25"
    "120:10"
    "120:12"
    "120:15"
    "120:20"
    "120:24"
    "140:10"
    "140:14"
    "140:20"
    "160:10"
    "160:16"
    "160:20"
)

echo "========================================"
echo "LSH Ablation Study"
echo "========================================"
echo "Testing ${#COMBINATIONS[@]} parameter combinations"
echo "with ${#SEEDS[@]} seeds each"
echo "Total runs: $((${#COMBINATIONS[@]} * ${#SEEDS[@]}))"
echo "========================================"
echo ""

TOTAL_RUNS=$((${#COMBINATIONS[@]} * ${#SEEDS[@]}))
CURRENT_RUN=0

for COMBO in "${COMBINATIONS[@]}"; do
    K=$(echo $COMBO | cut -d':' -f1)
    BANDS=$(echo $COMBO | cut -d':' -f2)
    ROWS=$((K / BANDS))

    echo "----------------------------------------"
    echo "Testing k=$K, bands=$BANDS (rows=$ROWS)"
    echo "----------------------------------------"

    for SEED in "${SEEDS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        echo "  Run $CURRENT_RUN/$TOTAL_RUNS: seed=$SEED"

        python3 main.py \
            --seed $SEED \
            --input $INPUT_FILE \
            --output "results/result_k${K}_b${BANDS}_s${SEED}.txt" \
            --threshold $THRESHOLD \
            --k $K \
            --bands $BANDS

        echo ""
    done
done

echo "========================================"
echo "Ablation study complete!"
echo "Results logged to runs.txt"
echo "========================================"
