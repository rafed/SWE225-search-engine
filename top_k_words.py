import heapq
from diskdict import DiskDict

top_n_heap = []
db = DiskDict()
top_k = 5000

unique_words = set()

for word in db.disk_index.keys():
    unique_words.add(word)
    postings = db.get(word)
    count = len(postings)
    
    if len(top_n_heap) < top_k:
        heapq.heappush(top_n_heap, (count, word))
    elif count > top_n_heap[0][0]:
        heapq.heapreplace(top_n_heap, (count, word))

top_words = [word for _, word in sorted(top_n_heap, reverse=True)]
print(len(unique_words))

with open("data/top_k_words.txt", "w") as f:
    for word in top_words:
        f.write(word + "\n")
