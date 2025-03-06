import heapq
from distdict import DistDict
top_n_heap = []
        
NUM_SHARDS = 200
db = DistDict()
top_k = 5000

unique_words = set()

for shard_id in range(NUM_SHARDS):
    shard_dict = db._read_dict_from_disk(shard_id)
    
    for word, postings in shard_dict.items():
        unique_words.add(word)
        count = len(postings)
        
        if len(top_n_heap) < top_k:
            heapq.heappush(top_n_heap, (count, word))
        elif count > top_n_heap[0][0]:
            heapq.heapreplace(top_n_heap, (count, word))
    
    del shard_dict

top_words = [word for _, word in sorted(top_n_heap, reverse=True)]
print(len(unique_words))

with open("data/top_k_words.txt", "w") as f:
    for word in top_words:
        f.write(word + "\n")

