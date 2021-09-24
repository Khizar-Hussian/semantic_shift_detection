import pandas as pd
import sys
import pickle
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial import distance
import plotly
import plotly.graph_objs as go
import numpy as np
from scipy.stats.stats import pearsonr
import argparse
import os
import csv
from tqdm import tqdm


def readTarget(input_path):
  lines = None
  with open(input_path, "r") as f:
    lines = f.readlines()
  data = [line.replace("\n", "") for line in lines]
  return data


def get_cos_dist(words, path, years):

    cds = []
    shifts = []
    word_list = []
    new_terms = []

    with open(path, 'rb') as f:
        vocab_vectors = pickle.load(f)
        print(f"# Keys in vocab vector: {len(vocab_vectors.keys())}")
    
    # try:
    for w in tqdm(words):
        words_emb = []

        for year in years:
            year_word = f"{w}_{year}"
            # print(year_word)
            if year_word in vocab_vectors:
                words_emb.append(vocab_vectors[year_word])

        if len(words_emb) != 2:
          print(f"Skipping word: {w}  <- Embedding doesn't exist for both contexts.")
          new_terms.append(w)
          continue

        cos_dist = distance.cosine(words_emb[0], words_emb[1]) #[0][0]
        cds.append(cos_dist)
        word_list.append(w)
    return cds, word_list, new_terms

def write_to_file(cds, words, path):
  with open(path, "w", newline='') as f:
    print("writing results to file.")
    writer = csv.writer(f)
    writer.writerow(["word", "cosine_distance"])
    for idx, word in tqdm(enumerate(words)):
      writer.writerow([word, cds[idx]])

def dump_new_terms(words, path):
    with open(path, "w", newline='') as f:
        for word in words:
            f.write(f"{word}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_path', type=str,
                        help='Path to output time embeddings',
                        default='embeddings/liverpool.pickle')
    parser.add_argument('--target_words', type=str,
                        help='Path to target words txt file',
                        default='./target_words.txt')
    parser.add_argument("--month_a", type=str, help="shorthand month and year eg mar-21")
    parser.add_argument("--month_b", type=str, help="shorthand month and year eg mar-21")
    args = parser.parse_args()


    src_month_a = args.month_a
    src_month_b = args.month_b
    years = [src_month_a[-4:], src_month_b[-4:]]

    target = readTarget(args.target_words)
    print(f"# of target words: {len(target)}")
    cds, words, new_words = get_cos_dist(target, args.embeddings_path, years)
    print(f"size of cos distance list: {len(cds)}")
    print(f"size of words list: {len(words)}")
    write_to_file(cds, words, f"/content/semantic_shift_detection/result.csv")
    dump_new_terms(new_words, f"/content/semantic_shift_detection/new_terms.txt")





