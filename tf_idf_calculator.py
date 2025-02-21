from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
from rank_bm25 import BM25Okapi

def getProcessedDocuments():
    allDocuments = {}
    folder_path="processed_files/"
    counter=0
    for filename in os.listdir(folder_path):
        #print(filename)
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            counter+=1
        
            # Open and read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                document = ''
                if data['title'] is not None and len(data['title']) > 0:
                    document += " ".join(data['title'])
                if data['h1'] is not None and len(data['h1']) > 0:
                    document += " ".join(data['h1'])
                if data['h2'] is not None and len(data['h2']) > 0:
                    document += " ".join(data['h2'])
                if data['h3'] is not None and len(data['h3']) > 0:
                    document += " ".join(data['h3'])
                if data['bold'] is not None and len(data['bold']) > 0:
                    document += " ".join(data['bold'])
                if data['other_text'] is not None and len(data['other_text']) > 0:
                    document += " ".join(data['other_text'])
                allDocuments[filename]=document
    print(f"counter is {counter}")
    return allDocuments

def calculate_tf_idf_of_all_documents(documents):
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words=None)

    # Fit and transform the corpus into a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(documents.values())

    # Get feature names (terms/words)
    terms = vectorizer.get_feature_names_out()

    custom_term_index = {}
    for index, term in enumerate(terms):
        custom_term_index[term] = index

    custom_doc_index = {}
    for i, (key, value) in enumerate(documents.items()):
        custom_doc_index[key] = i

    # Tf-idf of a specific term in specific document is -
    tfidf_value = tfidf_matrix[custom_doc_index['0a1a7e7eb0c0851ba9124227b935f8e80d9fbd1d55688e1de47d7b9775dd2ec6_processed.json'], custom_term_index['name']]
    print("value is "+str(tfidf_value))
    # Convert the matrix to a dense array and print it
    dense_matrix = tfidf_matrix.todense()
    #print("TF-IDF Matrix (Dense Form):\n", dense_matrix)


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


            
            


