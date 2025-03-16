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
import indexer as indx

# Initialize global data structures
db = DiskDict()
db.load_top_k_words_in_cache()
urls = defaultdict(list, orjson.loads(Path('data/url_mapping_with_pagerank.json').read_bytes()))
#urls = {v: k for k, v in urls.items()}
urls={value[0]: [key, value[1]] for key, value in urls.items()}


idf_dict = orjson.loads(Path('data/idf_values.json').read_bytes())
#doc_norms = orjson.loads(Path('data/doc_norms.json').read_bytes())
#doc_norms = defaultdict(int, {int(k): v for k, v in doc_norms.items()})
doc_len_norm_bm25 = orjson.loads(Path('data/doc_len_norm_bm25.json').read_bytes())

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
        if urls[doc_id][1] > 0 :
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
    best_scores = {}  # {doc_id: (weightedSimilarity, url)} for tracking best score per doc

    tf_idf_weight = 0.7
    page_rank_weight = 0.3

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

                # Prune based on upper bound
                upper_bound = estimate_lower_bound(
                    doc_scores[doc_id], remaining_terms, query_tfidf, doc_norms[doc_id]
                )
                # Note: We only use the bound for pruning later when heap is full

                # Compute exact similarity
                similarity = cosine_similarity(
                    doc_scores[doc_id], doc_norms[doc_id], query_norm
                )
                weightedSimilarity = ((similarity * tf_idf_weight) + (urls[doc_id][1] * page_rank_weight))

                url = urls.get(doc_id, [])[0] or []

                # Update best score for this doc_id
                if doc_id not in best_scores or weightedSimilarity > best_scores[doc_id][0]:
                    best_scores[doc_id] = (weightedSimilarity, url)

    # Build the top-k heap from best_scores with pruning
    heap = []
    min_score = 0.0
    for doc_id, (weightedSimilarity, url) in best_scores.items():
        # Prune here using the final score against min_score
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

def compute_bm25(allPostings, stemmed_token, doc_id):
    """
    Compute the BM25 score for a document with respect to a given query.
    """
    score = 0
    for term in stemmed_token:
        #print(term)
        if term not in idf_dict:
            print("term not found")
            continue

        posting = allPostings[term]
        #print(posting)
        #print(f"doc: {doc_id}")
        term_score=0
        for id,bm_25_score in posting:
            if(id == doc_id):
                term_score = float(bm_25_score)
                break

        #print(f"vale is {f}")
        
        #idf = idf_dict[term]  # Compute the IDF for the term
        #print(f"idf: {idf}")
        
        #doc_len_norm = doc_len_norm_bm25[str(doc_id)][1]  # Length of the document

        # Compute the BM25 component for the term
        #term_score = idf * ((f * (k1 + 1)) / (f + (k1 * doc_len_norm)))
        score += term_score  # Add the term score to the total score

    return score

def search_using_BM25(query,top_k=10):
    """
    Retrieve BM25 scores for the query, and return the top N relevant documents.
    """
    num_docs = len(doc_len_norm_bm25) - 1
    print(num_docs)
    tokens = tokenize(query)
    stemmed_tokens = stem_words(tokens)
    
    num_threads = 10
    chunk_size = num_docs // num_threads
    chunks = [range(i * chunk_size, (i + 1) * chunk_size) for i in range(num_threads)]

    # Handle the last chunk if it has fewer documents
    if chunks[-1][-1] != num_docs - 1:
        chunks[-1] = range(chunks[-1].start, num_docs)

    allPostings = {}
    for term in stemmed_tokens:
        allPostings[term] = db.get(term)

    # Function to compute BM25 score for a chunk of documents
    def process_chunk(chunk):
        results = []
        for doc_id in chunk:
            if urls[doc_id][1] > 0.0002977095588546717 :
                # Compute BM25 score for each document in the chunk
                bm25_score = compute_bm25(allPostings, stemmed_tokens, doc_id)
                #weighted_bm25score = ((bm25_score*0.7) + (urls[doc_id][1] * 0.3)) 
                results.append((doc_id, urls[doc_id][0], bm25_score))
        return results
    
    # Flatten the sorted and truncated sublists
    all_scores = []
    # Process chunks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_chunk, chunks))

        # Sort each sublist by BM25 score in descending order and keep only top 10
        for sublist in results:
            sublist.sort(key=lambda x: x[2], reverse=True)  # Sort each sublist by BM25 score

        for sublist in results:
            all_scores.extend(sublist)  # Add the top 10 documents from each sublist

    # Optionally, sort all scores again 
    all_scores.sort(key=lambda x: x[2], reverse=True)  # Sort by BM25 score in descending order
    
    # Return top N documents based on BM25 score
    return all_scores[:top_k]


if __name__ == '__main__':
    print("Starting query processor...")
    try:
        while True:
            query = input("Please enter your query: ")

            start_time = time.time()
            #top_results = search(query)
            top_results = search_using_BM25(query)
            end_time = time.time()
            
            print("\nTop results:")
            for doc_id, url, score in top_results:
                print(f"Doc ID: {doc_id}, URL: {url}, Similarity: {score:.4f}, pageRank:{urls[doc_id][1]:.12f}")
            
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time * 1000:.2f} ms")

    except KeyboardInterrupt:
        print("Exiting...")
        db.close()