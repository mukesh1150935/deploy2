import os
import sys
import random
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from torchtext.vocab import GloVe

# usage: python get_phrases2.py PATH_DATA SRC_LANG TGT_LANG FREQ_THRE
path_data = sys.argv[1]
src_lang = sys.argv[2]
tgt_lang = sys.argv[3]
FREQ_THRE = int(sys.argv[4]) if len(sys.argv) > 4 else 2

# Load train, test, and validation sets with UTF-8 encoding
with open(os.path.join(path_data, f"train.tok.true.{tgt_lang}"), "r", encoding="utf-8") as file:
    tgt = file.readlines()
with open(os.path.join(path_data, f"train.tok.true.{src_lang}"), "r", encoding="utf-8") as file:
    src = file.readlines()
with open(os.path.join(path_data, "train.align"), "r", encoding="utf-8") as file:
    align = file.readlines()

tgt_tokens = " ".join(tgt).split()

# Step 1: Extract MWEs using NLTK N-grams
bigram_measures = nltk.collocations.BigramAssocMeasures()
bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(tgt_tokens)
bigram_finder.apply_freq_filter(FREQ_THRE)

trigram_measures = nltk.collocations.TrigramAssocMeasures()
trigram_finder = nltk.collocations.TrigramCollocationFinder.from_words(tgt_tokens)
trigram_finder.apply_freq_filter(FREQ_THRE)

# Create dictionaries for MWEs
bigram_dict = {"_".join(words): score for words, score in bigram_finder.score_ngrams(bigram_measures.pmi)}
trigram_dict = {"_".join(words): score for words, score in trigram_finder.score_ngrams(trigram_measures.pmi)}

def align_to_dict(a):
    alignment = {}
    for pair in a.split():
        s, t = map(int, pair.split('-'))
        alignment.setdefault(s, []).append(t)
    return alignment

def extract_phrases(src, tgt, align):
    phrases = {}
    for src_sent, tgt_sent, a in zip(src, tgt, align):
        src_tokens, tgt_tokens = src_sent.split(), tgt_sent.split()
        a_dict = align_to_dict(a)
        for src_idx, tgt_idx in a_dict.items():
            if len(tgt_idx) > 1:
                tgt_idx.sort()
                tgt_mwe = "_".join([tgt_tokens[i] for i in tgt_idx])
                phrases.setdefault(tgt_mwe, {'freq': 0, 'src': {}})
                phrases[tgt_mwe]['freq'] += 1
                phrases[tgt_mwe]['src'][src_tokens[src_idx]] = phrases[tgt_mwe]['src'].get(src_tokens[src_idx], 0) + 1
    return phrases

phrases = extract_phrases(src, tgt, align)

# Step 2: Filter phrases based on PMI and frequency threshold
def filter_phrases(phrases, bigram_dict, trigram_dict):
    filtered_phrases = {}
    for phrase, value in phrases.items():
        n = len(phrase.split("_"))
        pmi = bigram_dict.get(phrase, 0) if n == 2 else trigram_dict.get(phrase, 0)
        if value['freq'] >= FREQ_THRE and pmi > 0:
            filtered_phrases[phrase] = {'freq': value['freq'], 'pmi': pmi, 'src': value['src']}
    return filtered_phrases

filtered_phrases = filter_phrases(phrases, bigram_dict, trigram_dict)

# Step 3: Load pre-trained GloVe embeddings using torchtext
glove = GloVe(name="6B", dim=100)

# Step 4: Generate embeddings for MWEs
mwe_embeddings = {}
for mwe in filtered_phrases:
    tokens = mwe.split("_")
    embedding = sum(glove[token] for token in tokens if token in glove) / len(tokens)
    mwe_embeddings[mwe] = embedding.tolist()

# Save MWEs and embeddings to files
with open(os.path.join(path_data, 'mwe_list.txt'), 'w', encoding="utf-8") as file:
    file.write("\n".join(mwe_embeddings.keys()))
with open(os.path.join(path_data, 'mwe_list.mwe.vec'), 'w', encoding="utf-8") as file:
    for mwe, embedding in mwe_embeddings.items():
        file.write(f"{mwe} {' '.join(map(str, embedding))}\n")

# Step 5: Process validation and test sets
def concat_mwe_phrases(tgt, align, src):
    tgt_processed = []
    align_processed = []
    for tgt_sent, align_line, src_sent in zip(tgt, align, src):
        tgt_tokens = tgt_sent.split()
        align_dict = align_to_dict(align_line)
        processed_sent = []
        new_align = {}
        for i, word in tgt_tokens:
            if i in align_dict:
                indices = align_dict[i]
                phrase = "_".join([tgt_tokens[j] for j in indices])
                if phrase in mwe_embeddings:
                    processed_sent.append(phrase)
                    new_align[len(processed_sent) - 1] = indices
                else:
                    processed_sent.append(word)
                    new_align[len(processed_sent) - 1] = [i]
            else:
                processed_sent.append(word)
                new_align[len(processed_sent) - 1] = [i]
        tgt_processed.append(" ".join(processed_sent))
        align_processed.append(" ".join([f"{k}-{v}" for k, v in new_align.items()]))
    return tgt_processed, align_processed

# Load validation and test data with UTF-8 encoding
with open(os.path.join(path_data, f"valid.tok.true.{src_lang}"), "r", encoding="utf-8") as file:
    src_val = file.readlines()
with open(os.path.join(path_data, f"test.tok.true.{src_lang}"), "r", encoding="utf-8") as file:
    src_test = file.readlines()
with open(os.path.join(path_data, f"test.tok.true.{tgt_lang}"), "r", encoding="utf-8") as file:
    tgt_test = file.readlines()
with open(os.path.join(path_data, f"valid.tok.true.{tgt_lang}"), "r", encoding="utf-8") as file:
    tgt_val = file.readlines()
with open(os.path.join(path_data, "test.align"), "r", encoding="utf-8") as file:
    align_test = file.readlines()
with open(os.path.join(path_data, "valid.align"), "r", encoding="utf-8") as file:
    align_val = file.readlines()

# Apply MWE processing
tgt_train_mwe, new_align_train = concat_mwe_phrases(tgt, align, src)
tgt_test_mwe, new_align_test = concat_mwe_phrases(tgt_test, align_test, src_test)
tgt_val_mwe, new_align_val = concat_mwe_phrases(tgt_val, align_val, src_val)

# Write processed MWE files
with open(os.path.join(path_data, f"train.tok.true.{tgt_lang}_mwe"), "w", encoding="utf-8") as file:
    file.write("\n".join(tgt_train_mwe))
with open(os.path.join(path_data, "train.align_mwe"), "w", encoding="utf-8") as file:
    file.write("\n".join(new_align_train))
