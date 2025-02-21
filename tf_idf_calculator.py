import json
import os
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

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

def create_inverted_index(documents):
    vectorizer = TfidfVectorizer()

    vectorizer.fit(documents)

    terms = vectorizer.get_feature_names_out()

    for doc_idx, doc in enumerate(documents):
        print(f"\nProcessing Document {doc_idx}:")

        # Transform the single document into a sparse TF-IDF matrix
        row = vectorizer.transform([doc])

        inverted_index = defaultdict(list)
        for word_idx in row.nonzero()[1]:
            word = terms[word_idx]
            tfidf_value = row[0, word_idx]
            inverted_index[word].append((doc_idx, round(tfidf_value, 4)))

    return inverted_index



print(create_inverted_index(getProcessedDocuments()))



            
            

