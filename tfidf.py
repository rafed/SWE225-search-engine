import os
import math
import collections
from tqdm import tqdm

class TFIDFCalculator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.num_docs = 0
        self.df = collections.defaultdict(int)  # Document Frequency
        self.tfidf = {}

    def _tokenize(self, text):
        return text.lower().split()  # Simple space-based tokenizer

    def compute_tf(self, filename):
        with open(filename, 'r', encoding="utf-8") as f:
            words = self._tokenize(f.read())
            total_terms = len(words)
            tf = collections.Counter(words)  # Raw term count
            return {word: count / total_terms for word, count in tf.items()}  # Normalize TF

    def compute_df(self):
        for file in tqdm(os.listdir(self.data_dir), "calculating df"):
            file_path = os.path.join(self.data_dir, file)
            if not os.path.isfile(file_path):
                continue

            self.num_docs += 1
            words_seen = set()
            with open(file_path, 'r', encoding="utf-8") as f:
                words = self._tokenize(f.read())
                words_seen.update(words)

            for word in words_seen:
                self.df[word] += 1

    def compute_tfidf(self):
        for file in tqdm(os.listdir(self.data_dir), "calculating idf"):
            file_path = os.path.join(self.data_dir, file)
            if not os.path.isfile(file_path):
                continue

            tf = self.compute_tf(file_path)
            tfidf_scores = {
                word: tf[word] * math.log(self.num_docs / (self.df[word] + 1))  # IDF smoothing
                for word in tf
            }
            self.tfidf[file] = tfidf_scores

    def process(self):
        self.compute_df()  # First pass: compute document frequencies
        self.compute_tfidf()  # Second pass: compute TF-IDF for each file

        return self.tfidf  # Dictionary {filename: {word: tfidf_score}}

data_directory = "./processed_files"  # Change this to your folder containing 30,000 files
tfidf_calculator = TFIDFCalculator(data_directory)
tfidf_scores = tfidf_calculator.process()



sample_file = list(tfidf_scores.keys())[0]
print(f"Top words in {sample_file}:")
for word, score in sorted(tfidf_scores[sample_file].items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{word}: {score:.4f}")
