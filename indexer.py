import os
import json
# from distdict import DistDict
from diskdict import DiskDict
from collections import defaultdict
import math
from tqdm import tqdm
import orjson
from pathlib import Path

db = DiskDict()

WEIGHTS = {
    "title": 3.0,
    "h1": 2.5,
    "h2": 2.5,
    "h3": 2.5,
    "h4": 2.5,
    "h5": 2.5,
    "h6": 2.5,
    "bold": 2.0,
    "other_text": 1.0
}

def document_generator(folder_path):
    global urls
    urls = defaultdict(int)
    doc_id = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                url = data.get("url", "")
                urls[url] = doc_id
                
                sections = {
                    "title": data.get("title", []),
                    "bold": data.get("bold", []),
                    "h1": data.get("h1", []),
                    "h2": data.get("h2", []),
                    "h3": data.get("h3", []),
                    "h4": data.get("h4", []),
                    "h5": data.get("h5", []),
                    "h6": data.get("h6", []),
                    "other_text": data.get("other_text", [])
                }

                yield doc_id, sections
                doc_id += 1 

def compute_df_idf(document_generator):
    df_counts = defaultdict(int)
    total_docs = 0
    idf_dict = {}

    for doc_id, sections in document_generator():
        unique_terms = set()
        
        for section, terms in sections.items():
            unique_terms.update(terms)
        
        for term in unique_terms:
            df_counts[term] += 1
        
        total_docs += 1

    for term, df in df_counts.items():
        idf_dict[term] = math.log(total_docs / df) + 1

    Path('data/idf_values.json').write_bytes(orjson.dumps(idf_dict))

    return idf_dict, total_docs

def compute_tf(sections):
    weighted_tf = defaultdict(float)
    total_weighted_terms = 0

    for section, terms in sections.items():
        weight = WEIGHTS.get(section, 1.0)  # Default to 1.0 if section not listed
        
        for term in terms:
            weighted_tf[term] += weight
            total_weighted_terms += weight 

    return {term: freq / total_weighted_terms for term, freq in weighted_tf.items()}

def compute_tf_idf(document_generator, idf_dict, total_docs):
    doc_norms = defaultdict(float)
    for doc_id, sections in tqdm(document_generator(), desc="Computing TF-IDF", total=total_docs):
        tf = compute_tf(sections)

        tfidf_scores = {
            term: tf[term] * idf_dict.get(term, 0)
            for term in tf
        }

        create_inverted_index(doc_id, tfidf_scores)

        doc_norms[str(doc_id)] = math.sqrt(sum(score ** 2 for score in tfidf_scores.values()))
    
    d = {str(k):v for k,v in doc_norms.items()}
    Path('data/doc_norms.json').write_bytes(orjson.dumps(d))

def create_inverted_index(doc_id, tfidf_scores):
    for term, value in tfidf_scores.items():
        if value > 0:
            db.put(term, (doc_id, f"{value:.4f}"))

if __name__ == '__main__':
    folder_path = "data/processed_files"
    global urls

    idf_dict, total_docs = compute_df_idf(lambda: document_generator(folder_path))
    compute_tf_idf(lambda: document_generator(folder_path), idf_dict, total_docs)

    db.close()

    Path('data/url_mapping.json').write_bytes(orjson.dumps(urls))
