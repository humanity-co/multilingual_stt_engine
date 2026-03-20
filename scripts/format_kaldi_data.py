import os
import glob
import pandas as pd

DATA_DIR = "../datasets/marathi"
KALDI_OUT = "../data/train_all"

os.makedirs(KALDI_OUT, exist_ok=True)

import os
import glob
import pandas as pd

import os
import glob
import pandas as pd

# The script runs from the scripts directory, so we point to the parent dir's datasets
DATA_DIR = "../datasets/marathi"
KALDI_OUT = "../data/train_all"

os.makedirs(KALDI_OUT, exist_ok=True)

def parse_open_slr():
    tsv_path = os.path.join(DATA_DIR, "line_index.tsv")
    
    if not os.path.exists(tsv_path):
        print("Error: line_index.tsv not found.")
        return False
        
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=["id", "text"])
    
    # We must sort Kaldi files by utterance ID
    df = df.sort_values(by="id")
    
    with open(os.path.join(KALDI_OUT, "wav.scp"), 'w', encoding='utf-8') as f_wav, \
         open(os.path.join(KALDI_OUT, "text"), 'w', encoding='utf-8') as f_text, \
         open(os.path.join(KALDI_OUT, "utt2spk"), 'w', encoding='utf-8') as f_utt2spk, \
         open(os.path.join(KALDI_OUT, "spk2utt"), 'w', encoding='utf-8') as f_spk2utt:
         
         speaker_dict = {}
         
         for _, row in df.iterrows():
             utt_id = row['id'].strip()
             text = row['text'].strip()
             speaker_id = utt_id[:3] # Usually first few characters
             
             if speaker_id not in speaker_dict:
                 speaker_dict[speaker_id] = []
             speaker_dict[speaker_id].append(utt_id)
                 
             possible_wav = os.path.abspath(os.path.join(DATA_DIR, f"{utt_id}.wav"))
             
             # Kaldi expects: utt_id command
             sox_pipe = f"sox {possible_wav} -t wav -r 16000 -c 1 - |"
             
             f_wav.write(f"{utt_id} {sox_pipe}\n")
             f_text.write(f"{utt_id} {text}\n")
             f_utt2spk.write(f"{utt_id} {speaker_id}\n")
             
         for spk in sorted(speaker_dict.keys()):
             utts = " ".join(sorted(speaker_dict[spk]))
             f_spk2utt.write(f"{spk} {utts}\n")
             
    print(f"Kaldi data files generated in {KALDI_OUT}")
    return True

if __name__ == "__main__":
    parse_open_slr()

if __name__ == "__main__":
    parse_open_slr()
