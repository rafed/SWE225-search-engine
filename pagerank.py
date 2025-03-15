import os
from urllib.parse import urlparse, urlunparse, quote
import warnings

import networkx as nx
import orjson as oj
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning, GuessedAtParserWarning
from tqdm import tqdm
from pathlib import Path
import json
from collections import defaultdict
import orjson
from decimal import Decimal

def getPageRanks():
    with open('data/url_mapping_with_pagerank.json', 'r') as infile:
        pageRanks = json.load(infile)
        return pageRanks

def standardize_url(url: str) -> str:
    try:
        p = urlparse(url)
    except:
        return None
    return urlunparse((
        p.scheme.lower(),
        p.netloc.lower(),
        quote(p.path, safe="/%"),
        quote(p.params, safe=";"),
        quote(p.query, safe="&="),
        quote(p.fragment, safe="#")
    ))

def pagerank(i: list[dict[str, str]]):
    allnodes = set()
    G = nx.DiGraph()
    print("Phase 1: Read URLs")
    for e in tqdm(range(len(i))):
        nurl = (i[e]["url"])
        if nurl is None:
            raise ValueError("Error: input URL is invalid, this cannot happen")
        G.add_node(nurl)
        allnodes.add(nurl)
        i[e]["nurl"] = nurl
    print("Phase 2: Read connections")
    for v in tqdm(range(len(i))):
        e = i[v]
        url = e["url"]
        anchors = e["anchor"]
        for a in anchors:
            if not a.startswith("http"):
                continue
            nurl = (a)
            if nurl is None:
                continue
            if not nurl in allnodes:
                continue
            G.add_edge(url, nurl)
    print("Phase 3: Run PageRank")
    R = nx.pagerank(G, 0.85)
    return R
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    warnings.filterwarnings("ignore", category=GuessedAtParserWarning)
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    docs = []
    for r, _, fs in os.walk("data/processed_files"):
        for f in fs:
            with open(os.path.join(r, f), "r", encoding="utf-8") as fp:
                docs.append(oj.loads(fp.read()))
    
    Rdict = pagerank(docs)

    # Apply min-max normalization
    print("Applying min-max normalization...")
    scores = list(Rdict.values())
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score if max_score != min_score else 1  # Avoid division by zero
    
    # Normalize the scores
    normalized_Rdict = {}
    for url, score in Rdict.items():
        normalized_score = (score - min_score) / score_range
        normalized_Rdict[url] = normalized_score

    # Add normalized page rank values to url_mapping_with_pagerank.json
    with open('data/url_mapping.json', 'r') as infile:
        data = json.load(infile)
        dataWithPageRank = defaultdict(lambda: (None, None))
        notFoundUrls = 0

        for doc_url, doc_id in tqdm(data.items(), total=len(data), desc="Processing Items"):
            if doc_url in normalized_Rdict:
                dataWithPageRank[doc_url] = (doc_id, normalized_Rdict[doc_url])
            else:
                notFoundUrls += 1
                dataWithPageRank[doc_url] = (doc_id, 0)
                print(doc_url)

        sorted_data = dict(sorted(dataWithPageRank.items(), key=lambda item: float(item[1][1]), reverse=True))
        print(f"threshold is: {list(sorted_data.items())[-15000][1]}")
        
        Path('data/url_mapping_with_pagerank.json').write_bytes(orjson.dumps(sorted_data, option=orjson.OPT_INDENT_2))
        print(f"not found urls: {notFoundUrls}")
        print(f"Min normalized score: {min(normalized_Rdict.values())}")
        print(f"Max normalized score: {max(normalized_Rdict.values())}")