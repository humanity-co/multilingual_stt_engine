#!/usr/bin/env bash

# KenLM Language Model Training Script for Multilingual Kaldi

set -euo pipefail
source ../training/path.sh

KENLM_BIN="/Users/devsmac/Documents/multilingual_stt_engine/kaldi/tools/kenlm/build/bin" 
CORPUS_DIR="../data/train_all"
LM_OUT_DIR="../data/local/lm"
KALDI_UTILS_DIR="../kaldi/egs/wsj/s5/utils"

mkdir -p $LM_OUT_DIR

TEXT_FILE="$CORPUS_DIR/text"

echo "=== Extracting Raw Text from Kaldi Format ==="
cut -d' ' -f2- "$TEXT_FILE" > "$LM_OUT_DIR/text_raw.txt"

echo "=== Training Unpruned 4-gram KenLM ==="
if [ ! -f "$KENLM_BIN/lmplz" ]; then
    echo "Warning: KenLM binaries not found at $KENLM_BIN. Creating generic ARPA for testing."
    # We will build a dummy G.fst if kenlm is missing on this specific environment test
    cat << 'ARPA' > $LM_OUT_DIR/lm_4gram.arpa
\data\
ngram 1=3
ngram 2=3
ngram 3=2
\1-grams:
-0.4771212  <unk>   0
-0.4771212  <s>     -0.30103
-0.4771212  </s>    0
\2-grams:
-0.30103    <s> <unk>   0
-0.30103    <unk> </s>  0
\3-grams:
-0.30103    <s> <unk> </s>
\end\
ARPA
else
    $KENLM_BIN/lmplz -o 4 --text "$LM_OUT_DIR/text_raw.txt" --arpa $LM_OUT_DIR/lm_4gram.arpa
    $KENLM_BIN/build_binary -q 8 -b 8 trie $LM_OUT_DIR/lm_4gram.arpa $LM_OUT_DIR/lm_4gram.binary
fi

echo "=== Converting ARPA to Kaldi G.fst ==="
if command -v arpa2fst &> /dev/null; then
  # Requires Kaldi tools in PATH
  arpa2fst --disambig-symbol=#0 --read-symbol-table=../data/lang/words.txt $LM_OUT_DIR/lm_4gram.arpa $LM_OUT_DIR/G.fst
else
  echo "arpa2fst not in PATH. Will compile G.fst during acoustic training graph build."
fi

echo "LM generation complete. Saved in $LM_OUT_DIR"
