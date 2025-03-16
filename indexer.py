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
    "title": 5.0,
    "h1": 4.0,
    "h2": 4.0,
    "h3": 4.0,
    "h4": 3.0,
    "h5": 3.0,
    "h6": 3.0,
    "bold": 2.0,
    "other_text": 1.0
}
#bm25 paramenters
b=0.75
k1=1.5
avg_doc_length=0

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
        #idf_dict[term] = math.log10(total_docs / df)
        idf_dict[term] = math.log10(((total_docs-df+0.5) / (df+0.5))+1)

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

def compute_tf_idf(document_generator, idf_dict, doc_len_norm_bm25, total_docs):
    doc_norms = defaultdict(float)
    

    for doc_id, sections in tqdm(document_generator(), desc="Computing TF-Only_Now", total=total_docs):
        tf = compute_tf(sections)

        #tfidf_scores = {
        #    term: tf[term] * idf_dict.get(term, 0)
        #    for term in tf
        #}

        term_bm25_scores={}
        for term, value in tf.items():
            term_bm25_scores[term] = idf_dict.get(term, 0) * ( (value * (k1+1)) / ( value + ( k1 * doc_len_norm_bm25[str(doc_id)][1])))

        create_inverted_index(doc_id, term_bm25_scores)

        #doc_norms[str(doc_id)] = math.sqrt(sum(score ** 2 for score in tfidf_scores.values()))
    
    #d = {str(k):v for k,v in doc_norms.items()}
    #Path('data/doc_norms.json').write_bytes(orjson.dumps(d))

def create_inverted_index(doc_id, tfidf_scores):
    for term, value in tfidf_scores.items():
        if value > 0:
            db.put(term, (doc_id, f"{value:.4f}"))

def compute_document_length_normalization_bm25(document_generator):
    total_docs = 0
    dlength_normalization_dict = defaultdict(lambda: (0, 0.0))
    sum_doc_length = 0
    x=0
    for doc_id, sections in tqdm(document_generator(), desc="Computing BM25 doc length", total=total_docs):
        doc_length = 0
        for section, term in sections.items():
            doc_length = doc_length +  ( len(term) * WEIGHTS.get(section, 1.0) )
            if(x == 0):
                print(f"\n checking {doc_id} {section} {len(term)} {WEIGHTS.get(section, 1.0)}")   
        dlength_normalization_dict[str(doc_id)]=(doc_length, 0.0)
        sum_doc_length = sum_doc_length + doc_length
        total_docs += 1
        #print(f"{sum_doc_length} {doc_length}")
        x +=1

    avg_doc_length = sum_doc_length / total_docs
    print(f"Avg is: {avg_doc_length} {total_docs}")
    new_dict = defaultdict(lambda: (0, 0.0))
    for doc_id, value in dlength_normalization_dict.items():
        new_dict[str(doc_id)]=(value[0], (1 + b - (b * (value[0]/avg_doc_length))))

    new_dict["avg_doc_length"]=(avg_doc_length,0)


    Path('data/doc_len_norm_bm25.json').write_bytes(orjson.dumps(new_dict))



if __name__ == '__main__':
    folder_path = "data/processed_files"
    global urls

    compute_document_length_normalization_bm25(lambda: document_generator(folder_path))
    idf_dict, total_docs = compute_df_idf(lambda: document_generator(folder_path))
    doc_len_norm_bm25 = orjson.loads(Path('data/doc_len_norm_bm25.json').read_bytes())
    compute_tf_idf(lambda: document_generator(folder_path), idf_dict, doc_len_norm_bm25, total_docs)

    db.close()

    Path('data/url_mapping.json').write_bytes(orjson.dumps(urls))
    
