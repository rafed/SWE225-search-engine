import re
from nltk.stem import PorterStemmer, SnowballStemmer

def tokenize(text):
    return [word.lower() for word in re.findall(r"[a-zA-Z0-9]+", text)]


def stem_words(words):
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(word) for word in words]


if __name__ == '__main__':
    text_1 = """This is a text to check Stemming. Whether this actually works or not! Testing123 Ph.D in S.E., UC irvine"""
    text_2 = """Running through the forest, the children’s laughter echoed. Quickly, they’d jumped over fallen logs, enjoying nature’s beauty. However, one tripped—falling, fallen, falls. “Ouch!” he cried. Does stemming recognize ‘running’ and ‘run’ as the same? What about ‘quickly’ vs. ‘quick’? Testing, tested, tester, tests."""
    words = tokenize(text_1)
    stemmed_words = stem_words(words)
    print(stemmed_words)

    words = tokenize(text_2)
    stemmed_words = stem_words(words)
    print(stemmed_words)