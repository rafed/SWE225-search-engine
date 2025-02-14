from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus (list of documents)
documents = [
    "Cats are great pets.",
    "Dogs are loyal pets.",
    "I love my cat."
]

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the corpus into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names (terms/words)
terms = vectorizer.get_feature_names_out()

# Convert the matrix to a dense array and print it
dense_matrix = tfidf_matrix.todense()

# Display the TF-IDF matrix and feature names
print("TF-IDF Matrix (Dense Form):\n", dense_matrix)
print("\nTerms (Feature Names):", terms)

# Optionally, convert to DataFrame for better visualization
import pandas as pd
df = pd.DataFrame(dense_matrix, columns=terms)
print("\nDataFrame Representation of TF-IDF Matrix:\n", df)
