import numpy as np
from bs4 import BeautifulSoup
from typing import List, Dict

def standardize_url(url: str) -> str:
    """
    Standardizes the URL, for example converting to lowercase.
    You can extend this to handle more URL normalization cases.
    """
    return url.lower()

def extract_links(content: str) -> List[str]:
    """
    Extracts all the hyperlinks (href) from the HTML content using BeautifulSoup.
    """
    soup = BeautifulSoup(content, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return links

def build_adjacency_matrix(docs: List[Dict[str, str]], chunk_size: int = 10000) -> np.ndarray:
    """
    Builds the adjacency matrix where `adj_matrix[i, j] = 1` if document `j` links to document `i`.
    This function should be broken into chunks to handle large datasets.
    """
    n = len(docs)
    adj_matrix = np.zeros((n, n))

    # Create URL to index mapping
    url_to_index = {doc['nurl']: idx for idx, doc in enumerate(docs)}

    # Process documents in chunks to avoid memory overload
    for idx, doc in enumerate(docs):
        url = doc['nurl']
        content = doc['content']
        
        # Extract links from the document
        links = extract_links(content)
        
        for link in links:
            # Only add links that are part of the documents
            if link in url_to_index:
                adj_matrix[url_to_index[link], idx] = 1  # Directed edge from link to current document

    return adj_matrix, url_to_index

def pagerank(docs: List[Dict[str, str]], alpha: float = 0.85, tol: float = 1.0e-6, chunk_size: int = 10000, max_iter: int = 100) -> List[tuple]:
    """
    Implements the PageRank algorithm for a collection of documents.
    It handles large datasets by processing documents in chunks.
    """
    n = len(docs)
    
    # Step 1: Standardize URLs and build adjacency matrix
    for doc in docs:
        doc['nurl'] = standardize_url(doc['url'])
    
    adj_matrix, url_to_index = build_adjacency_matrix(docs, chunk_size)
    
    # Step 2: Initialize the rank vector
    ranks = np.ones(n) / n  # Initial rank, each document starts with equal probability

    # Step 3: Handle dangling nodes (nodes with no out-links)
    out_degree = np.sum(adj_matrix, axis=0)  # Number of out-links for each document
    dangling_nodes = np.where(out_degree == 0)[0]
    
    for node in dangling_nodes:
        adj_matrix[node, :] = 1 / n  # Treat dangling nodes as if they link to all other nodes
    
    # Step 4: Apply the PageRank algorithm (Power Iteration)
    for iteration in range(max_iter):
        new_ranks = np.ones(n) * (1 - alpha) / n  # Start with the (1 - alpha) / N factor
        new_ranks += alpha * np.dot(adj_matrix, ranks / out_degree)  # Apply the link structure

        # Check for convergence (using L1 norm)
        if np.linalg.norm(new_ranks - ranks, 1) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break
        
        ranks = new_ranks  # Update the ranks for the next iteration

    # Step 5: Sort URLs by their PageRank scores in descending order
    sorted_ranks = sorted(zip(url_to_index.keys(), ranks), key=lambda x: x[1], reverse=True)

    return sorted_ranks

def main():
    # Example documents: List of dictionaries where each dictionary contains URL and content
    docs = [
        {"url": "http://example1.com", "content": "<html><a href='http://example2.com'>Link</a></html>"},
        {"url": "http://example2.com", "content": "<html><a href='http://example1.com'>Link</a></html>"},
        # Add more document dictionaries here...
    ]
    
    # Run PageRank
    result = pagerank(docs)
    
    # Print top 10 ranked URLs
    for url, score in result[:10]:
        print(f"{url}: {score}")

if __name__ == "__main__":
    main()
