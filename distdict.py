import threading
import hashlib
from pathlib import Path
import os
from collections import defaultdict
import orjson
from functools import lru_cache

NUM_SHARDS = 600
MAX_COUNT_IN_MEM = 500000

class DistDict:
    _lock = threading.Lock()
    _instances = {}

    def __new__(cls, db_name="db"):
        with cls._lock:  # Ensure thread-safe singleton
            if db_name not in cls._instances:
                instance = super().__new__(cls)
                instance._init_db(db_name)
                cls._instances[db_name] = instance
            return cls._instances[db_name]

    def _init_db(self, db_name="db"):
        self.db_path = "data/distdict_" + db_name
        os.makedirs(self.db_path, exist_ok=True)
        self.memory_dict = defaultdict(list)
        self.lock = threading.Lock()
        self.counter = 0
        self._cache_most_common_words()

    def _get_shard_id(self, term, num_shards = NUM_SHARDS):
        return int(hashlib.md5(term.encode()).hexdigest(), 16) % num_shards

    def _read_dict_from_disk(self, shard_id:int):
        shard_path = os.path.join(self.db_path, f"shard_{shard_id}.json")
        if os.path.exists(shard_path):
            json_bytes = Path(shard_path).read_bytes()
            shard_dict = defaultdict(list, orjson.loads(json_bytes))
        else:
            shard_dict = defaultdict(list)
        return shard_dict

    def _write_dict_to_disk(self, shard_id:int, shard_dict:defaultdict):
        shard_path = os.path.join(self.db_path, f"shard_{shard_id}.json")
        json_bytes = orjson.dumps(shard_dict)
        Path(shard_path).write_bytes(json_bytes)

    def _flush_to_disk(self):
        shard_batches = defaultdict(list) # For batch updates

        for word, tuple_list in self.memory_dict.items():
            shard_id = self._get_shard_id(word)
            shard_batches[shard_id].append((word, tuple_list))

        for shard_id, word_tuple_list in shard_batches.items():
            shard_dict = self._read_dict_from_disk(shard_id)
            for word, tuple_list in word_tuple_list:
                for tuple in tuple_list:
                    shard_dict[word].append(tuple)

            self._write_dict_to_disk(shard_id, shard_dict)

        self.memory_dict = defaultdict(list) # fresh memory

    def put(self, word: str, doc_id: int, tfidf: float):
        with self.lock: 
            self.memory_dict[word].append((doc_id, tfidf))
            self.counter += 1

            if self.counter % MAX_COUNT_IN_MEM == 0:
                self._flush_to_disk()
    
    @lru_cache(maxsize=5000)
    def get(self, word: str) -> list:
        # self._flush_to_disk()
        shard_id = self._get_shard_id(word)
        shard_dict = self._read_dict_from_disk(shard_id)
        return shard_dict[word]
    
    def flush(self):
        self._flush_to_disk()

    def _cache_most_common_words(self, n=1000):
        with open("data/top_k_words.txt", "r") as f:
            top_k_words = f.readlines()

        for word in top_k_words[:n]:
            self.get(word)
                