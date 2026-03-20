import os
import re

def normalize_text(text):
    """
    Normalizes Hindi/Marathi script and English text.
    Handles basic punctuation removal.
    """
    text = text.lower()
    text = re.sub(r'[!?,.:;"\'-]', '', text)
    return text.strip()

def naive_indic_g2p(word):
    """
    A very naive grapheme to phoneme mapping purely for demonstration.
    In production, use Phonetisaurus, espeak-ng, or dict based approaches
    like epitran to map Hindi/Marathi -> IPA/Indic Phoneset.
    """
    phones = " ".join(list(word))
    return phones

def create_lexicon(corpus_file, output_lexicon):
    """
    Reads a corpus file, extracts all unique words, and generates
    a Kaldi compatible lexicon.txt (word phoneme_sequence)
    """
    vocab = set()
    
    # 1. Extract vocabulary
    print(f"Reading corpus {corpus_file}")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            # Normally Kaldi text files are "utt_id word1 word2..."
            parts = line.strip().split(maxsplit=1)
            if len(parts) > 1:
                sentence = normalize_text(parts[1])
                for word in sentence.split():
                    vocab.add(word)

    # 2. Add structural tokens like SIL (silence), UNK (unknown cutoff)
    vocab.add("<SPOKEN_NOISE>")
    vocab.add("<UNK>")
    
    # 3. Write to lexicon
    print(f"Writing {len(vocab)} words to {output_lexicon}")
    with open(output_lexicon, 'w', encoding='utf-8') as out:
        out.write(f"!SIL SIL\n")
        out.write(f"<SPOKEN_NOISE> SPN\n")
        out.write(f"<UNK> SPN\n")
        for word in sorted(list(vocab)):
            if word not in ["!SIL", "<SPOKEN_NOISE>", "<UNK>"]:
                phones = naive_indic_g2p(word)
                out.write(f"{word} {phones}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True, help="Input Kaldi text file")
    parser.add_argument("--lexicon", type=str, required=True, help="Output lexicon.txt path")
    
    args = parser.parse_args()
    create_lexicon(args.corpus, args.lexicon)
