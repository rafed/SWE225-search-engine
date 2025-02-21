import os
import json
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

def document_generator(folder_path):
    """Yields 'title', 'bold', 'other_text', and combined 'h1+h2+h3' from JSON files."""
    for filename in sorted(os.listdir(folder_path)):  # Ensure consistent order
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                title_text = " ".join(data.get("title", []))
                bold_text = " ".join(data.get("bold", []))
                other_text = " ".join(data.get("other_text", []))
                headings_text = " ".join(data.get("h1", []) + data.get("h2", []) + data.get("h3", []))
                yield filename, title_text, bold_text, other_text, headings_text

def compute_global_df(document_generator, field_index, chunk_size=1000):
    df_counts = defaultdict(int)
    total_docs = 0

    chunk = []
    for doc_tuple in document_generator():
        doc = doc_tuple[field_index + 1]  # Select field: 0-title, 1-bold, 2-other_text, 3-headings
        chunk.append(doc)
        if len(chunk) >= chunk_size:
            for doc in chunk:
                total_docs += 1
                for word in set(doc.split()):  # Count unique words per document
                    df_counts[word] += 1
            chunk = []  # Reset chunk

    if chunk:
        for doc in chunk:
            total_docs += 1
            for word in set(doc.split()):
                df_counts[word] += 1

    print(df_counts)
    return df_counts, total_docs

def compute_tf_idf(document_generator, df_counts, total_docs, field_index, chunk_size=1000):
    count_vectorizer = CountVectorizer(vocabulary=df_counts.keys())

    print(total_docs)
    print(df_counts)
    idf_values = np.array([np.log(total_docs / (df_counts[word] + 1)) for word in df_counts])
    print(len(idf_values))
    print(idf_values)
    vocab_list = list(df_counts.keys())

    chunk = []
    doc_ids = []  # Store document filenames
    for doc_tuple in document_generator():
        doc_id = doc_tuple[0]  # Filename as document ID
        doc = doc_tuple[field_index + 1]  # Select field: 0-title, 1-bold, 2-other_text, 3-headings
        chunk.append(doc)
        doc_ids.append(doc_id)

        if len(chunk) >= chunk_size:
            raw_tf_matrix  = count_vectorizer.fit_transform(chunk)
            num_terms_in_docs = raw_tf_matrix.sum(axis=1).A1
            num_terms_in_docs[num_terms_in_docs == 0] = 1
            tf_matrix = raw_tf_matrix / num_terms_in_docs[:, np.newaxis]
            tfidf_matrix = tf_matrix.multiply(idf_values)
            
            print_tfidf_scores(doc_ids, tfidf_matrix, vocab_list)
            chunk, doc_ids = [], []

    if chunk:
        raw_tf_matrix = count_vectorizer.fit_transform(chunk)
        print(count_vectorizer.get_feature_names_out())
        print(raw_tf_matrix.shape)
        print(raw_tf_matrix)
        num_terms_in_docs = raw_tf_matrix.sum(axis=1).A1
        num_terms_in_docs[num_terms_in_docs == 0] = 1
        tf_matrix = raw_tf_matrix / num_terms_in_docs[:, np.newaxis]
        print(tf_matrix.shape)
        print(tf_matrix)
        tfidf_matrix = tf_matrix.multiply(idf_values)
        print(tfidf_matrix.shape)
        print(tfidf_matrix)

        print_tfidf_scores(doc_ids, tfidf_matrix, vocab_list)

def print_tfidf_scores(doc_ids, tfidf_matrix, vocab_list):
    tfidf_matrix = tfidf_matrix.tocsr()  # Convert sparse matrix to CSR format

    for doc_index, doc_id in enumerate(doc_ids):
        print(f"\nDocument: {doc_id}")
        tfidf_scores = tfidf_matrix[doc_index].toarray().flatten()  # Now it's subscriptable
        sorted_indices = np.argsort(-tfidf_scores)  # Sort words by TF-IDF score (descending)
        
        for idx in sorted_indices[:]:
            if tfidf_scores[idx] > 0:
                print(f"  {vocab_list[idx]}: {tfidf_scores[idx]:.4f}")


if __name__ == '__main__':
    folder_path = "processed_files"

    fields = ["title", "bold", "other_text", "headings (h1+h2+h3)"]

    for i, field_name in enumerate(fields):
        print(f"\nProcessing field: {field_name}")
        df_counts, total_docs = compute_global_df(lambda: document_generator(folder_path), field_index=i, chunk_size=1000)
        compute_tf_idf(lambda: document_generator(folder_path), df_counts, total_docs, field_index=i, chunk_size=1000)
