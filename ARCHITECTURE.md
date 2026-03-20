# System Architecture

The STT engine is based on the **Kaldi Hybrid DNN-HMM architecture**. It features a modern, factorized TDNN (TDNN-F) acoustic model trained with LF-MMI (Chain modeling) and a heavily pruned KenLM 4-gram Language Model. 

The inference pipeline is custom-built in Python, leveraging Kaldi binaries via subprocess/wrappers for maximum flexibility and performance without relying on high-level wrappers like Vosk.

## 1. Inference Data Flow (Real-Time)

```text
Microphone Input (16kHz, Mono)
      ↓
Silero Voice Activity Detection (VAD)  --> Drops silence, forwards speech chunks
      ↓
Audio Preprocessing                    --> Amplitude normalization, DC offset removal
      ↓
Feature Extraction                     --> Kaldi MFCC (40 dim) + Pitch + CMVN
      ↓
TDNN-F Acoustic Model                  --> Frame-level acoustic scoring
      ↓
HMM Decoder (Kaldi Online LatGen)      --> Graph search over HCLG.fst
      ↓
KenLM 4-gram Language Model            --> Language level rescoring
      ↓
Final Text Output
```

### Voice Activity Detection (Silero VAD)
- **Role:** Analyzes incoming audio chunks (e.g., 512 samples) and returns a probability of speech.
- **Benefit:** Prevents the Kaldi decoder from processing silence, saving massive amounts of CPU and reducing hallucinated words.

### Preprocessing & Feature Extraction
- **Audio:** Re-sampled (if needed) to 16 kHz Mono WAV format in memory.
- **Features:** 40-dimensional High-Resolution MFCCs. Pitch features are essential for Indic tonal aspects. Cepstral Mean and Variance Normalization (CMVN) is applied per-utterance (or via sliding window online) to handle volume variations.

### Acoustic Model (TDNN-F Chain)
- **Architecture:** Factorized Time Delay Neural Networks (TDNN-F) reduce the parameter count significantly compared to standard TDNNs by factoring the weight matrices using Singular Value Decomposition (SVD).
- **Context Window:** The network looks at a wide context (e.g., -1.5s to +1.5s relative to the current frame) but processes it sparsely through dilated convolutions.
- **Training:** LF-MMI (Lattice-Free Maximum Mutual Information) optimizes sequence-level criteria, drastically dropping Word Error Rate (WER) and supporting frame sub-sampling (decoding at 33ms instead of 10ms frame rates, tripling speed).

### Language Model & Graph (HCLG)
Kaldi compiles the various models into a single Finite State Transducer (FST) called `HCLG.fst`:
- **H:** HMM definitions (transition probabilities).
- **C:** Context-dependency (triphone trees).
- **L:** Lexicon (words to phonemes).
- **G:** Grammar (KenLM smaller 3-gram or 4-gram).

## 2. Multilingual Strategy

We adopt the **Shared Acoustic Model + Separate Language Models** approach, augmented with a single fallback code-mixed LM.

**Why Shared Acoustic Model?**
- Marathi and Hindi share the Devanagari script and heavily overlap in their phonatory structure.
- Indian English is highly influenced by the native phonetic substrate.
- Pooling training data (Marathi + Hindi + Indian English) yields a much more robust, accent-agnostic acoustic model.
- Prevents the need to load 3 separate 60MB acoustic models into memory.

**Implementation Details:**
- **Phoneset:** A universal, IPA-based or common Indic phoneset maps phonemes across the 3 languages uniquely.
- **Lexicon (L.fst):** The lexicon contains words from all three languages mapped to the shared phoneset.
- **Language Models (G.fst):**
    - The engine maintains separate pruned 4-gram models for specific contexts (e.g., `G_marathi.fst`, `G_hindi.fst`).
    - Alternatively, for seamless code-mixing, a unified interpolated LM covers the vocabulary of all three, leaning heavily on conversational text.

## 3. Data Augmentation (SpecAugment)
During TDNN-F training, **SpecAugment** is applied on the fly:
- **Time Masking:** Chunks of consecutive frames are masked to zero.
- **Frequency Masking:** Blocks of mel-frequency bins are zeroed out.
- **Benefits:** Models become extremely robust against background noise, channel distortion, and partial phonetic drops, which are very common in real-world Indian applications.
