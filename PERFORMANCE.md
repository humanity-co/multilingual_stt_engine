# Performance & Optimization Guide

This document outlines the optimization techniques applied to achieve the target metrics for our pure Kaldi STT engine.

## Target Metrics
- **Latency:** < 300 ms (from speech end to text generation).
- **Real-Time Factor (RTF):** < 0.5 on a standard i5 CPU core.
- **Total Payload Size:** < 150 MB (Acoustic + Language Model + Graph).

## 1. Acoustic Model Optimization
**Model Compression via Factoring:**
- The **TDNN-F** architecture natively compresses the network by factoring weight matrices into two smaller matrices with a bottleneck linear layer.
- **Target Size:** The raw acoustic model `.mdl` should be ~60MB to 100MB.
- **Sub-sampling:** The LF-MMI chain models operate on a 30ms or 33ms frame shift (instead of the traditional 10ms). This means the decoder evaluates 1/3 as many frames per second, fundamentally improving the RTF.

## 2. Language Model Pruning (KenLM)
Language models grow exponentially with n-gram size. To keep the 4-gram LM under 100MB:

**Pruning Strategy:**
- Use KenLM's `lmplz` to build the unpruned model.
- Prune heavily using `build_binary -q 8 -b 8` (8-bit quantization for probabilities and backoff weights) and trie data structures.
- Apply entropy-based pruning to remove higher-order n-grams that do not significantly lower the perplexity on a validation set.
- Command example: `build_binary -q 8 -b 8 trie unpruned.arpa pruned.binary`

## 3. Decoding Beam Optimization
Kaldi's online decode uses graph search over the HCLG transducer. Expanding the search graph takes CPU time.

**Tuning Parameters (`kaldi_decode.conf`):**
- **`--beam=10.0` or `12.0`:** (Default is often 15). Lowering the beam restricts the number of active paths. If confidence is high, tighter beams drop the RTF drastically.
- **`--lattice-beam=6.0`:** Determines how deeply the lattice is searched before extracting the 1-best path. Lowering this speeds up endpointing.
- **`--max-active=3000`:** Hard cap on the number of states active at any frame. Default is ~7000. Capping at 3000 prevents CPU spikes during noisy audio segments where the graph branches wildly.

## 4. Streaming Latency Reduction (Silero VAD)
- **Buffering:** Standard Kaldi online decoding waits for chunks. By pairing it with Silero VAD, we only feed verified voice data into the Kaldi pipe, bypassing silence.
- **Chunk Size:** Provide 200–300ms chunks to the decoder rather than feeding sample-by-sample, which reduces the IPC (inter-process communication) overhead.
- **Endpointing:** Configure Kaldi’s endpointing logic (`--endpoint.silence-hits-max`) to eagerly conclude the utterance as soon as trailing silence is verified by Silero.
