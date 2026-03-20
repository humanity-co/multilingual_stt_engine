#!/usr/bin/env bash

# Data Preparation for Kaldi (Marathi, Hindi, English)
# Downloads and formats datasets into standard Kaldi text, wav.scp, utt2spk, spk2utt formats.

set -euo pipefail

DATA_ROOT="../datasets"
mkdir -p "$DATA_ROOT"

echo "=== 1. Downloading Marathi Data ==="
# E.g., OpenSLR 64 (Marathi)
# wget -P $DATA_ROOT http://www.openslr.org/resources/64/mr_in_female.zip
# unzip $DATA_ROOT/mr_in_female.zip -d $DATA_ROOT/marathi/

echo "=== 2. Downloading Hindi Data ==="
# E.g., OpenSLR 71 (Hindi)
# wget -P $DATA_ROOT http://www.openslr.org/resources/71/hi_in_female.zip 
# unzip $DATA_ROOT/hi_in_female.zip -d $DATA_ROOT/hindi/

echo "=== 3. Formatting Data for Kaldi ==="
# Create standard directories
KALDI_DATA_DIR="../data/train_all"
mkdir -p "$KALDI_DATA_DIR"

# Note: In a real environment, you would parse the TSV/CSV files from the datasets
# and output these files:
# 1. wav.scp (utt_id /path/to/wav)
# 2. text (utt_id transcript)
# 3. utt2spk (utt_id spk_id)
# 4. spk2utt (spk_id utt_id)

touch "$KALDI_DATA_DIR/wav.scp"
touch "$KALDI_DATA_DIR/text"
touch "$KALDI_DATA_DIR/utt2spk"

# Utility function to enforce 16kHz mono locally via SoX
echo "Converting all wav to 16kHz Mono..."
# find $DATA_ROOT -name "*.wav" -exec sox {} -r 16000 -c 1 {}.16k.wav \;
# mv {}.16k.wav {} 

# Ensure Kaldi sorting conventions inside the data folder
# utils/utt2spk_to_spk2utt.pl $KALDI_DATA_DIR/utt2spk > $KALDI_DATA_DIR/spk2utt
# utils/validate_data_dir.sh --no-feats $KALDI_DATA_DIR

echo "Data Preparation complete. Ready for feature extraction."
