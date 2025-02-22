import os
import json
import pickle
import collections
from distdict import DistDict
from collections import defaultdict
import math
from tqdm import tqdm

db = DistDict('mydb_rafed')

def document_generator(folder_path):
    global urls
    urls = defaultdict(int)
    id = 0
    for filename in sorted(os.listdir(folder_path)): 
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                url = data.get("url", "")
                urls[url] = id
                id = id + 1 
                title_text = " ".join(data.get("title", []))
                bold_text = " ".join(data.get("bold", []))
                other_text = " ".join(data.get("other_text", []))
                headings_text = " ".join(data.get("h1", []) + data.get("h2", []) + data.get("h3", []))
                yield id, " ".join([title_text, bold_text, headings_text, other_text])

def compute_df(document_generator, chunk_size=1000):
    df_counts = defaultdict(int)
    total_docs = 0

    for doc_tuple in document_generator():
        doc = doc_tuple[1] 
        
        total_docs += 1
        for word in set(doc.split()):
            df_counts[word] += 1

    # print(df_counts)
    return df_counts, total_docs

def compute_tf(content):
    words = content.split()
    total_terms = len(words)
    tf = collections.Counter(words)  # Raw term count
    return {word: count / total_terms for word, count in tf.items()}
        
def compute_tf_idf(document_generator, df_counts, total_docs):
    for doc_tuple in tqdm(document_generator(), desc="computing tf-idf", total=total_docs):
        doc_id = doc_tuple[0]
        content = doc_tuple[1]
        tf = compute_tf(content)
        tfidf_scores = {
            word: tf[word] * math.log(total_docs / (df_counts[word] + 1))
            for word in tf
        }

        create_inverted_index(doc_id, tfidf_scores)

def create_inverted_index(doc_id, tfidf_scores):
    for key, value in tfidf_scores.items():
        if value > 0:
            db.put(key, doc_id, f"{value:.4f}")


if __name__ == '__main__':
    folder_path = "processed_files"
    global urls

    df_counts, total_docs = compute_df(lambda: document_generator(folder_path), chunk_size=1000)
    compute_tf_idf(lambda: document_generator(folder_path), df_counts, total_docs)

    db.flush()

    print(urls)
    with open('url_mapping.pkl', 'wb') as pkl_file:
        pickle.dump(urls, pkl_file)