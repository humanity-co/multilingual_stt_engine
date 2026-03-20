import sys
import queue
import json
import os
import wave
import tempfile
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from vad_silero import SileroVAD

# Configuration
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 32
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# Vosk Model Paths
VOSK_HINDI_PATH = "models/vosk/vosk-model-small-hi-0.22"
VOSK_ENGLISH_PATH = "models/vosk/vosk-model-small-en-in-0.4"

# Whisper Model for Marathi (small = best results so far)
WHISPER_MARATHI_MODEL = "small"

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy().flatten())


def is_hallucination(text):
    """Detect Whisper hallucinations (repetitive text, gibberish)."""
    if not text or len(text) < 2:
        return True
    # Check for repeating patterns (e.g. "वेवेवेवे")
    for pattern_len in range(1, min(6, len(text) // 3 + 1)):
        pattern = text[:pattern_len]
        if text == pattern * (len(text) // len(pattern)) + pattern[:len(text) % len(pattern)]:
            return True
    # Check if any single character takes up more than 50% of the text
    from collections import Counter
    counts = Counter(text.replace(" ", ""))
    if counts and counts.most_common(1)[0][1] > len(text.replace(" ", "")) * 0.5:
        return True
    return False


class WhisperMarathiDecoder:
    """
    Uses OpenAI Whisper (via faster-whisper) for Marathi STT.
    Segment-based: collects speech, then transcribes the full segment.
    """
    def __init__(self):
        from faster_whisper import WhisperModel
        print(f"  Loading Whisper '{WHISPER_MARATHI_MODEL}' for Marathi...")
        self.model = WhisperModel(WHISPER_MARATHI_MODEL, device="cpu", compute_type="int8")
        print("  ✅ Whisper Marathi ready!")
    
    def decode_segment(self, pcm_float32):
        """Decode a numpy float32 audio segment to Marathi text."""
        pcm_int16 = (pcm_float32 * 32767).astype(np.int16)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_path = tf.name
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm_int16.tobytes())
            
            segments, info = self.model.transcribe(
                tmp_path,
                language="mr",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                temperature=0.0,
            )
            
            # Collect segments and filter hallucinations
            parts = []
            for seg in segments:
                t = seg.text.strip()
                if t and not is_hallucination(t):
                    parts.append(t)
            return " ".join(parts)
        except Exception as e:
            print(f"\n  Whisper error: {e}", flush=True)
            return ""
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except:
                    pass


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Multilingual Kaldi STT Engine (Hindi/English/Marathi) ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    print("\nInitializing models:")
    
    # Load Vosk Hindi
    print("\n  Loading Hindi model...")
    hi_model = Model(VOSK_HINDI_PATH)
    hi_recognizer = KaldiRecognizer(hi_model, SAMPLE_RATE)
    hi_recognizer.SetWords(True)
    print("  ✅ Hindi (हिंदी) ready!")
    
    # Load Vosk English
    print("\n  Loading English model...")
    en_model = Model(VOSK_ENGLISH_PATH)
    en_recognizer = KaldiRecognizer(en_model, SAMPLE_RATE)
    en_recognizer.SetWords(True)
    print("  ✅ English (Indian) ready!")
    
    # Load Whisper Marathi
    print()
    mr_decoder = WhisperMarathiDecoder()
    
    # Load VAD
    vad = SileroVAD(threshold=0.5)

    # Language state
    LANG_HI, LANG_EN, LANG_MR = "hi", "en", "mr"
    current_lang = LANG_HI
    vosk_recognizers = {LANG_HI: hi_recognizer, LANG_EN: en_recognizer}
    lang_names = {
        LANG_HI: "Hindi (हिंदी)",
        LANG_EN: "English (Indian)",
        LANG_MR: "Marathi (मराठी) [Whisper]",
    }
    lang_flags = {LANG_HI: "🇮🇳", LANG_EN: "🇬🇧", LANG_MR: "🟠"}

    print(f"\n🌐 Current language: {lang_names[current_lang]}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Press 1 = Hindi | 2 = English | 3 = Marathi")
    print("  Press Ctrl+C to stop")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("\n🎧 Listening...\n")

    in_speech_state = False
    speech_buffer = []  # For Marathi segment-based decoding

    import select, tty, termios
    old_settings = termios.tcgetattr(sys.stdin)
    
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                            blocksize=CHUNK_SIZE, callback=audio_callback):
            while True:
                # Check for language switch (non-blocking)
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    new_lang = {"1": LANG_HI, "2": LANG_EN, "3": LANG_MR}.get(key)
                    if new_lang and new_lang != current_lang:
                        current_lang = new_lang
                        speech_buffer = []
                        in_speech_state = False
                        print(f"\n\n🌐 Switched to: {lang_names[current_lang]}\n")
                
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                prob = vad.get_speech_probability(chunk)
                
                if current_lang == LANG_MR:
                    # ---- Marathi: Segment-based Whisper decoding ----
                    if prob > 0.35:
                        if not in_speech_state:
                            print(f"\n{lang_flags[current_lang]} ", end="", flush=True)
                            in_speech_state = True
                        speech_buffer.append(chunk)
                        print(".", end="", flush=True)
                    else:
                        if in_speech_state and len(speech_buffer) > 30:
                            # Pad with 0.3s silence on both sides for context
                            pad = np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32)
                            print(" [Transcribing...]", flush=True)
                            full_audio = np.concatenate([pad] + speech_buffer + [pad])
                            text = mr_decoder.decode_segment(full_audio)
                            if text:
                                print(f"   ✅ {text}", flush=True)
                            else:
                                print("   (no speech detected)", flush=True)
                            speech_buffer = []
                            in_speech_state = False
                        elif not in_speech_state:
                            speech_buffer = []
                else:
                    # ---- Hindi/English: Vosk streaming decoding ----
                    recognizer = vosk_recognizers[current_lang]
                    pcm_data = (chunk * 32767).astype(np.int16).tobytes()
                    
                    if prob > 0.35:
                        if not in_speech_state:
                            print(f"\n{lang_flags[current_lang]} ", end="", flush=True)
                            in_speech_state = True
                        
                        if recognizer.AcceptWaveform(pcm_data):
                            result = json.loads(recognizer.Result())
                            text = result.get("text", "")
                            if text:
                                print(f"\n   ✅ {text}", flush=True)
                        else:
                            partial = json.loads(recognizer.PartialResult())
                            partial_text = partial.get("partial", "")
                            if partial_text:
                                print(f"\r   💬 {partial_text}   ", end="", flush=True)
                    else:
                        if in_speech_state:
                            in_speech_state = False
                            result = json.loads(recognizer.FinalResult())
                            text = result.get("text", "")
                            if text:
                                print(f"\n   ✅ {text}", flush=True)
                        else:
                            recognizer.AcceptWaveform(pcm_data)

    except KeyboardInterrupt:
        if current_lang != LANG_MR and current_lang in vosk_recognizers:
            result = json.loads(vosk_recognizers[current_lang].FinalResult())
            text = result.get("text", "")
            if text:
                print(f"\n   ✅ Final: {text}")
        print("\n\n👋 Stopping transcription. Goodbye!")
    except queue.Empty:
        pass  # Normal timeout, just continue
    except Exception as e:
        import traceback
        print(f"\nError: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    main()
