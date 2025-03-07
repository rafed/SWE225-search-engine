# Search Engine Project

## How to run

```bash
python page_processor.py developer/DEV
python indexer.py
python top_k_words.py
python query_processor.py
```

## Overview
This project is to build a complete search engine over the course of three milestones. Groups of 1-3 members will work collaboratively to design and implement the system. Groups with at least one CS or SE student are required to complete the search engine option. The final deliverable will include an indexer and a search component capable of processing queries and ranking results efficiently.

## General Guidelines
- **Use of Code:** You may reuse code from previous projects but cannot use code written by non-group members for this project.
- **Allowed Libraries:** You can use any language and libraries for text processing, such as `nltk`. However, text indexing libraries like Lucene, PyLucene, or ElasticSearch are not allowed.
- **Corpus:** The dataset consists of web pages crawled during Project 2, organized as JSON files in `developer.zip`.
- **Deliverables:** Each milestone has specific deliverables and evaluation criteria.

## Milestones

### Milestone #1: Build an Index
#### Goal
- Create an inverted index for the provided corpus.
- Tokens include all alphanumeric sequences.
- No stopping; include all words.
- Use stemming (e.g., Porter stemming) for better matches.
- Treat important words (bold, headings, titles) as more significant.

#### Deliverables
- A short report (PDF) with the following:
  - Number of documents.
  - Number of unique tokens.
  - Total size of the index (in KB) on disk.

#### Evaluation Criteria
- Timely submission of the report.
- Plausibility of the reported numbers.

### Milestone #2: Develop a Search and Retrieval Component
#### Goal
- Implement a search component that:
  - Accepts queries via a console prompt.
  - Uses tf-idf and cosine similarity for ranking.
  - Returns URLs of relevant pages ranked by relevance.

#### Required Queries for Testing
1. Iftekhar Ahmed
2. Machine Learning
3. ACM
4. Master of Software Engineering

#### Deliverables
- A short report (PDF) with:
  - Top 5 URLs for each query.
  - Screenshot of the search interface in action.

#### Evaluation Criteria
- Timely submission of the report.
- Plausibility of the reported URLs.

### Milestone #3: Complete Search Engine
#### Goal
- Improve the search engine by optimizing for:
  - Ranking performance.
  - Runtime performance.
- Evaluate using at least 20 queries, half of which initially perform poorly.

#### Deliverables
- A zip file containing all the programs written.
- A document with test queries and commentary on improvements made.
- Live demonstration of the search engine.

#### Evaluation Criteria
- Does the search engine function as expected?
- How general are the heuristics employed?
- Is the response time under 300ms?
- Can the team demonstrate and explain the implementation?

## Specifications

### Indexer
- Build an inverted index stored on disk.
- Important design considerations:
  - Partial index files: Offload to disk at least 3 times during index construction.
  - Merging: Combine partial indexes at the end.
  - Optional splitting into term-range files after merging.

### Search Component
- Prompt for user queries.
- Rank pages using tf-idf (and optionally additional heuristics).
- Response time: Under 300ms (targeting 100ms or less).

### Ranking
- Use tf-idf and cosine similarity.
- Enhance ranking with:
  - Word importance (e.g., bold, headings, titles).
  - Optional improvements (e.g., PageRank, n-grams).

## Extra Credit Opportunities
- **Duplicate Page Elimination:** +2 points.
- **PageRank Implementation:** +3 points.
- **2-Gram/3-Gram Indexing:** +2 points.
- **Word Positions for Retrieval:** +2 points.
- **Anchor Words Indexing:** +1 point.
- **Web/GUI Interface:** +2 points.

## Dataset Details
- **Structure:**
  - One folder per domain.
  - Files are JSON with two fields: `url` and `content`.
- **Challenges:**
  - Broken or missing HTML tags.
  - Use a parser library that can handle malformed HTML.

## Notes
- Use efficient data structures to balance memory usage and response time.
- Modularize code to aid debugging and teamwork.
- Use GitHub for collaboration.

---

