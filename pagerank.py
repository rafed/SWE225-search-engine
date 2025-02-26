import os
from urllib.parse import urlparse, urlunparse, quote
import warnings

import networkx as nx
import orjson as oj
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning, GuessedAtParserWarning
from tqdm import tqdm

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

# Page schema: url=, content=, encoding=,
# Two passes: identify documents, 
# Note: this does modify the input, as it standardizes the URL format
def pagerank(i: list[dict[str, str]]):
    allnodes = set()
    G = nx.DiGraph()
    print("Phase 1: Read URLs")
    for e in tqdm(range(len(i))):
        nurl = standardize_url(i[e]["url"])
        if nurl is None:
            raise ValueError("Error: input URL is invalid, this cannot happen")
        G.add_node(nurl)
        allnodes.add(nurl)
        i[e]["nurl"] = nurl
    print("Phase 2: Read connections")
    for v in tqdm(range(len(i))):
        e = i[v]
        url = e["url"]
        content = BeautifulSoup(e["content"])
        for a in content.find_all("a", href=True):
            v = a["href"]
            if not v.startswith("http"):
                continue
            nurl = standardize_url(a["href"])
            if nurl is None:
                continue
            if not nurl in allnodes:
                continue
            G.add_edge(url, nurl)
    print("Phase 3: Run PageRank")
    R = nx.pagerank(G, 0.85)
    return sorted(R.items(), key=lambda i: i[1], reverse=True)

# For testing purposes only, assumes the files are in ../../Downloads/DEV
if __name__ == "__main__" and os.environ.get("RUN_PAGERANK_TEST"):
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    warnings.filterwarnings("ignore", category=GuessedAtParserWarning)
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    docs = []
    for r, _, fs in os.walk("../../Downloads/DEV"):
        for f in fs:
            with open(os.path.join(r, f), "r") as fp:
                docs.append(oj.loads(fp.read()))
    print(pagerank(docs)[:50])
