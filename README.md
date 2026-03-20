# Multilingual STT Kaldi System (Offline & CPU Optimized)

This repository contains a clean, modular, production-ready multilingual Speech-to-Text (STT) system built directly on the **Kaldi Hybrid DNN-HMM architecture** (without external inference wrappers like Vosk). It is optimized for Marathi, Hindi, and English (Indian Accents).

## Key Features
- **Offline First**: Entirely local recognition engine.
- **CPU Efficient**: Designed to run optimally on standard CPUs without GPU reliance for inference.
- **Low Latency Streaming**: Integrated Silero VAD and Kaldi online decoding for <300ms latency.
- **TDNN-F Acoustic Model**: Uses Kaldi's LF-MMI trained factorized TDNNs for high accuracy and small footprint.
- **KenLM Language Models**: Highly compressed 4-gram LMs pruned for efficiency.
- **Multilingual Support**: Supports code-mixed scenarios and multiple Indian languages using a shared acoustic space.

## Repository Structure
```
multilingual_stt_engine/
├── ARCHITECTURE.md      # Detailed system architecture and design choices
├── PERFORMANCE.md       # Target metrics, pruning, and optimization guide
├── datasets/            # Audio and text corpora
├── models/
│   ├── acoustic_model/  # Trained TDNN-F chain model
│   └── language_models/ # Pruned KenLM 4-gram LMs
├── lexicon/             # G2P generation and pronunciation dictionaries
├── scripts/             # Data preparation tools
├── training/            # Kaldi training pipeline scripts
├── language_models/     # LM training wrappers
├── inference/           # Python-based streaming inference
│   ├── vad_silero.py    # Silero VAD integration
│   └── streaming_stt.py # Real-time custom Kaldi STT Python logic
├── requirements.txt
└── README.md
```

## Setup & Installation

**Prerequisites:**

1. **Kaldi Speech Recognition Toolkit**: Must be installed separately. 
   - [NEW] The `Kaldi/` source/build directory is **excluded** from this repository due to its size (>4GB). 
   - Follow the [official Kaldi installation guide](https://kaldi-asr.org/doc/install.html) to compile it.
   - Ensure the compiled binaries (like `online2-wav-nnet3-latgen-faster`) are in your `$PATH`.
2. **KenLM**: Required for language model training and inference.
3. **Python 3.8+**: Recommended for the streaming inference wrapper.
4. **Git LFS**: This repository uses Git LFS for models and datasets. Run `git lfs pull` after cloning.

**Python Environment:**
> [!NOTE]
> The `venv/` directory is **excluded**. You must create your own local virtual environment.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

1. **Data Prep:** Read through `scripts/prepare_data.sh` to download and structure the corpora.
2. **Lexicon Gen:** Run `lexicon/generate_lexicon.py` to handle Grapheme-to-Phoneme mappings.
3. **Training:** Follow `training/run_kaldi.sh` (which integrates all steps from monophone to chain model).
4. **LM Training:** Compile LMs using `language_models/train_lm.sh`.
5. **Inference:** Test real-time microphone decoding via:
   ```bash
   python inference/streaming_stt.py
   ```

## Target Metrics
- Real-time factor (RTF) < 0.5 on CPU
- Streaming latency < 300 ms
- Model foot-print (AM + LM) < 150 MB

See `ARCHITECTURE.md` and `PERFORMANCE.md` for in-depth technical breakdowns.
