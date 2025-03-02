from text_processor import tokenize, stem_words
from distdict import DistDict
import math
from collections import defaultdict
import time
from pathlib import Path
import orjson

db = DistDict()
urls = defaultdict(list, orjson.loads(
    Path('data/url_mapping.json').read_bytes()
))

def compute_query_tf(query_terms):
    term_counts = defaultdict(int)
    total_terms = len(query_terms)
    
    for term in query_terms:
        term_counts[term] += 1
    
    return {term: count / total_terms for term, count in term_counts.items()}

def compute_query_tfidf(query_terms, idf_dict):
    tf = compute_query_tf(query_terms)
    return {term: tf[term] * idf_dict.get(term, 0) for term in tf}

def cosine_similarity(query_tfidf, doc_dot_product, doc_norm, query_norm):
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return doc_dot_product / (query_norm * doc_norm)

def search(query_terms, idf_dict, doc_norms, top_k=10):
    query_tfidf = compute_query_tfidf(query_terms, idf_dict)
    query_norm = math.sqrt(sum(val ** 2 for val in query_tfidf.values()))
    
    doc_scores = defaultdict(float)

    for term in query_tfidf.keys():
        postings = db.get(term)
        for (doc_id, tfidf_score) in postings:
            doc_scores[doc_id] += query_tfidf[term] * float(tfidf_score)
    
    results = []
    for doc_id in doc_scores:
        similarity = cosine_similarity(query_tfidf, doc_scores[doc_id], doc_norms[doc_id], query_norm)
        url = [u for u, d in urls.items() if d == doc_id]
        results.append((similarity, doc_id, url))
    
    results.sort(reverse=True)
    return results[:top_k]


if __name__ == '__main__':
    idf_dict = orjson.loads(Path('data/idf_values.json').read_bytes())
    doc_norms = orjson.loads(Path('data/doc_norms.json').read_bytes())
    doc_norms = defaultdict(int, {int(k): v for k, v in doc_norms.items()})

    query = input("Please enter your query: ")
    start_time = time.time()
    
    tokens = tokenize(query)
    stemmed_tokens = stem_words(tokens)

    top_results = search(stemmed_tokens, idf_dict, doc_norms)
    end_time = time.time()
    
    print("\nTop results:")
    for score, doc_id, url in top_results:
        print(f"Doc ID: {doc_id}, URL: {url}, Similarity: {score:.4f}")
    
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
