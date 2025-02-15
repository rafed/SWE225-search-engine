from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
from rank_bm25 import BM25Okapi

def getProcessedDocuments():
    allDocuments = []
    folder_path="processed_files/"
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
        
            # Open and read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                document = ''
                document += " ".join(data['title'])
                document += " ".join(data['h1'])
                document += " ".join(data['h2'])
                document += " ".join(data['h3'])
                document += " ".join(data['bold'])
                document += " ".join(data['other_text'])
                allDocuments.append(document)

    return allDocuments

def calculate_tf_idf_of_all_documents(documents):
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the corpus into a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Get feature names (terms/words)
    terms = vectorizer.get_feature_names_out()

    custom_term_index = {}
    for index, term in enumerate(terms):
        custom_term_index[term] = index

    custom_doc_index = {}
    print(len(documents))
    for i in range(len(documents)):
        doc=str(i)+'_processed'
        custom_doc_index[doc] = i

    # Tf-idf of a specific term in specific document is -
    tfidf_value = tfidf_matrix[custom_doc_index['1_processed'], custom_term_index['subscript']]
    print("value is "+str(tfidf_value))
    # Convert the matrix to a dense array and print it
    dense_matrix = tfidf_matrix.todense()
    print("TF-IDF Matrix (Dense Form):\n", dense_matrix)


def get_bm25_score_for_each_document(documents, query):
    bm25 = BM25Okapi(documents)

    # Compute BM25 scores for the query
    bm25_scores = bm25.get_scores(query)

    # Output BM25 scores for each document in the corpus
    for i, score in enumerate(bm25_scores):
        print(f"Document {i + 1} BM25 Score: {score:.4f}")



calculate_tf_idf_of_all_documents(getProcessedDocuments())

# Query to search for
query = ["this",
        "is",
        "a",
        "paragraph"]
get_bm25_score_for_each_document(getProcessedDocuments(),query)


            
            


