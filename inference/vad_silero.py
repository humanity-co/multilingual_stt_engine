import torch
import numpy as np
import torchaudio

class SileroVAD:
    """
    Wrapper for Silero VAD.
    Responsible for identifying speech within an audio stream
    to reduce latency and unnecessary decoding.
    """
    def __init__(self, threshold=0.5, sampling_rate=16000):
        # Load model from torch hub perfectly offline
        print("Loading Silero VAD model...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        (self.get_speech_timestamps, _, self.read_audio, self.VADIterator, _) = self.utils
        
        # VADIterator acts as a stateful detector for streaming chunks
        self.vad_iterator = self.VADIterator(self.model, threshold=self.threshold)
        print("Silero VAD initialized.")

    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Receives an audio chunk (float32 numpy array, 16kHz mono).
        Returns True if the chunk triggered or maintained a speech state.
        Returns False if the chunk represents silence.
        """
        # Convert to torch tensor
        tensor = torch.from_numpy(audio_chunk).float()
        
        speech_dict = self.vad_iterator(tensor, return_seconds=False)
        
        if speech_dict is not None:
            # We either got a 'start' or an 'end' trigger
            if 'start' in speech_dict:
                return True
            if 'end' in speech_dict:
                return False
                
        # If None, stay with current state context.
        # However, for pure gating, we ask the model raw probability periodically or rely on iterator
        # For simple implementations, if we haven't seen an 'end', we assume speech continues if 'start' happened.
        pass

    def get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """
        Calculates raw speech probability for a given window.
        """
        tensor = torch.from_numpy(audio_chunk).float()
        # Ensure we feed expected chunk size (e.g. 512 samples)
        if len(tensor) < 512:
            return 0.0
        prob = self.model(tensor, self.sampling_rate).item()
        return prob
