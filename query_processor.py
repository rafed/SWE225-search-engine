from text_processor import tokenize, stem_words
from diskdict import DiskDict
import math
from collections import defaultdict, Counter
import time
from pathlib import Path
import orjson
from concurrent.futures import ThreadPoolExecutor
from heapq import heappush, heappop
import pagerank as pr

# Initialize global data structures
db = DiskDict()
db.load_top_k_words_in_cache()
urls = defaultdict(list, orjson.loads(Path('data/url_mapping_with_pagerank.json').read_bytes()))
#urls = {v: k for k, v in urls.items()}
urls={value[0]: [key, value[1]] for key, value in urls.items()}


idf_dict = orjson.loads(Path('data/idf_values.json').read_bytes())
doc_norms = orjson.loads(Path('data/doc_norms.json').read_bytes())
doc_norms = defaultdict(int, {int(k): v for k, v in doc_norms.items()})

pageRanks = pr.getPageRanks()

def compute_query_tf(query_terms):
    term_counts = Counter(query_terms)
    total_terms = len(query_terms)
    return {term: count / total_terms for term, count in term_counts.items()}

def compute_query_tfidf(query_terms):
    tf = compute_query_tf(query_terms)
    return {term: tf[term] * idf_dict.get(term, 0) for term in tf}

def cosine_similarity(doc_dot_product, doc_norm, query_norm):
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return doc_dot_product / (query_norm * doc_norm)

def compute_doc_scores(term, query_tfidf):
    postings = db.get(term)
    term_score = query_tfidf[term]
    local_scores = defaultdict(float)
    
    for doc_id, tfidf_score in postings:
        if urls[doc_id][1] >= 0.000010712589241379107 :
            local_scores[doc_id] += term_score * float(tfidf_score)

    return local_scores

def estimate_lower_bound(doc_score, remaining_terms, query_tfidf, doc_norm):
    """Estimate a lower bound on the final score for pruning."""
    max_remaining = sum(
        query_tfidf[term] * doc_norm
        for term in remaining_terms
    )
    return doc_score + max_remaining

def search(query, top_k=10):
    tokens = tokenize(query)
    stemmed_tokens = stem_words(tokens)

    query_tfidf = compute_query_tfidf(stemmed_tokens)
    query_norm = math.sqrt(sum(val ** 2 for val in query_tfidf.values()))
    
    doc_scores = defaultdict(float)  # Accumulate dot products
    processed_terms = set()
    best_scores = {}  # {doc_id: (similarity, url)} for tracking best score per doc

    tf_idf_weight = 0.8
    page_rank_weight = 0.2

    # Process terms in parallel
    with ThreadPoolExecutor() as executor:
        term_results = executor.map(
            lambda term: (term, compute_doc_scores(term, query_tfidf)),
            query_tfidf.keys()
        )

        for term, local_scores in term_results:
            processed_terms.add(term)
            remaining_terms = set(query_tfidf.keys()) - processed_terms

            # Update document scores
            for doc_id, score in local_scores.items():
                doc_scores[doc_id] += score  # Accumulate partial dot product

                # Prune based on lower bound
                lower_bound = estimate_lower_bound(
                    doc_scores[doc_id], remaining_terms, query_tfidf, doc_norms[doc_id]
                )
                # We'll use min_score later, after building best_scores
                # For now, just compute similarity
                similarity = cosine_similarity(
                    doc_scores[doc_id], doc_norms[doc_id], query_norm
                )
                weightedSimilarity =  ((similarity * tf_idf_weight) + (urls[doc_id][1] * page_rank_weight )) / (tf_idf_weight + page_rank_weight)

                #url = urls.get(doc_id, [])
                url = urls.get(doc_id, [])[0] or []

                # Update best score for this doc_id
                if doc_id not in best_scores or weightedSimilarity > best_scores[doc_id][0]:
                    best_scores[doc_id] = (weightedSimilarity, url)

    # Build the top-k heap from best_scores
    heap = []
    min_score = 0.0
    for doc_id, (weightedSimilarity, url) in best_scores.items():
        if len(heap) < top_k:
            heappush(heap, (weightedSimilarity, doc_id, url))
            if len(heap) == top_k:
                min_score = heap[0][0]
        elif weightedSimilarity > min_score:
            heappop(heap)
            heappush(heap, (weightedSimilarity, doc_id, url))
            min_score = heap[0][0]

    # Extract results from heap
    results = []
    while heap:
        results.append(heappop(heap))
    results.sort(reverse=True)
    return results


if __name__ == '__main__':
    print("Starting query processor...")
    try:
        while True:
            query = input("Please enter your query: ")

            start_time = time.time()
            top_results = search(query, top_k=5)
            end_time = time.time()
            
            print("\nTop results:")
            for score, doc_id, url in top_results:
                print(f"Doc ID: {doc_id}, URL: {url}, Similarity: {score:.4f}, pageRank:{urls[doc_id][1]:.12f}")
            
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time * 1000:.2f} ms")

    except KeyboardInterrupt:
        print("Exiting...")
        db.close()