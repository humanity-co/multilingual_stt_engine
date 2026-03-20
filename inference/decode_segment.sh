#!/bin/bash
# Decode a single WAV file using the trained tri1 GMM model.
# Usage: decode_segment.sh <wav_path>
# Outputs: "segment1 word_id1 word_id2 ..." to stdout

WAV_PATH="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

MODEL="$BASE_DIR/models/acoustic_model/final.mdl"
GRAPH="$BASE_DIR/models/acoustic_model/HCLG.fst"
WORDS="$BASE_DIR/models/lang/words.txt"
MFCC_CONF="$BASE_DIR/models/conf/mfcc.conf"
PITCH_CONF="$BASE_DIR/models/conf/pitch.conf"

# Feature extraction: MFCC + Pitch + Deltas = 129 dims
# Then decode and extract best word ID path
compute-mfcc-feats --config="$MFCC_CONF" \
    "scp,p:echo segment1 $WAV_PATH |" ark:- | \
paste-feats --length-tolerance=2 ark:- \
    "ark:compute-and-process-kaldi-pitch-feats --config=$PITCH_CONF 'scp,p:echo segment1 $WAV_PATH |' ark:- |" \
    ark:- | \
add-deltas ark:- ark:- | \
gmm-latgen-faster --allow-partial=true \
    "$MODEL" "$GRAPH" ark:- ark:- 2>/dev/null | \
lattice-best-path ark:- ark,t:- 2>/dev/null
