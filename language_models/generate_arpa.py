import math
import collections
import os

input_file = "../data/train_all/text"
output_file = "../data/local/lm/lm_2gram.arpa"

unigrams = collections.Counter()
bigrams = collections.Counter()

with open(input_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) > 1:
            words = ["<s>"] + parts[1:] + ["</s>"]
            for w in words:
                unigrams[w] += 1
            for i in range(len(words)-1):
                bigrams[(words[i], words[i+1])] += 1

unigrams["<unk>"] = 1 # Backoff padding

total_unigrams = sum(unigrams.values())
total_bigrams = sum(bigrams.values())

with open(output_file, "w") as f:
    f.write("\\data\\\n")
    f.write(f"ngram 1={len(unigrams)}\n")
    f.write(f"ngram 2={len(bigrams)}\n\n")
    
    f.write("\\1-grams:\n")
    for word, count in unigrams.items():
        prob = math.log10(count / total_unigrams)
        # simplistic backoff weight
        backoff = -0.30103 if word != "</s>" else 0.0
        f.write(f"{prob:.6f}\t{word}\t{backoff}\n")
        
    f.write("\n\\2-grams:\n")
    for (w1, w2), count in bigrams.items():
        prob = math.log10(count / unigrams[w1])
        f.write(f"{prob:.6f}\t{w1} {w2}\n")
        
    f.write("\n\\end\\\n")
print(f"Generated {output_file} successfully.")
