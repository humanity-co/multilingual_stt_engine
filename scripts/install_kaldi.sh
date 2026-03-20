#!/usr/bin/env bash

# Kaldi Compilation Script for macOS (Apple Silicon / Intel)
# This will take a VERY long time and requires system dependencies.

set -euo pipefail

cd /Users/devsmac/.gemini/antigravity/scratch/multilingual_stt_engine

echo "1. Checking/Installing System Dependencies..."
# Ensure you have Homebrew installed!
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Please install it first from https://brew.sh/"
    exit 1
fi

brew install cmake wget sox subversion automake autoconf libtool
brew install openblas  # Essential for fast math on CPU

echo "2. Cloning Kaldi Repository..."
if [ ! -d "kaldi" ]; then
    git clone https://github.com/kaldi-asr/kaldi.git
fi
cd kaldi

echo "3. Compiling Tools (OpenFST, etc)..."
cd tools
# This script sometimes needs adjustments on arm64 macs, but Kaldi tries to handle it
make -j $(sysctl -n hw.ncpu)
cd ..

echo "4. Compiling Kaldi Source (src)..."
cd src
# Configure with OpenBLAS
./configure --shared --mathlib=OPENBLAS --openblas-root=$(brew --prefix openblas)
make clean -j $(sysctl -n hw.ncpu)
make depend -j $(sysctl -n hw.ncpu)
make -j $(sysctl -n hw.ncpu)

echo "Kaldi compilation finished! Binaries are now in kaldi/src/"
