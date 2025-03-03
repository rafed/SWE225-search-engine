from text_processor import tokenize, stem_words
from distdict import DistDict

db = DistDict()

query = input("Please enter your query: ")
tokens = tokenize(query)
stemmed_tokens = stem_words(tokens)


