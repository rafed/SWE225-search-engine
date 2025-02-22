import threading
import hashlib
import pickle
import os
from collections import defaultdict

NUM_SHARDS = 10
MAX_COUNT_IN_MEM = 100

class DistDict:
    _shared_state = {}  # Borg shared state
    _lock = threading.Lock()

    # def __init__(self, db_path="./distdict_db"):
    #     self.__dict__ = self._shared_state
    #     if not hasattr(self, "initialized"):
    #         with self._lock:
    #             if not hasattr(self, "initialized"):
    #                 self._init_db(db_path)
    #                 self.initialized = True

    def __init__(self, db_name="db"):
        self.db_path = "distdict_" + db_name
        os.makedirs(self.db_path, exist_ok=True)

        self.lock = threading.Lock()
        self.memory_dict = defaultdict(list)
        self.counter = 0

    def _get_shard_id(self, term, num_shards = NUM_SHARDS):
        return int(hashlib.md5(term.encode()).hexdigest(), 16) % num_shards

    def _read_dict_from_disk(self, shard_id:int):
        shard_path = os.path.join(self.db_path, f"shard_{shard_id}.pkl")
        if os.path.exists(shard_path):
            with open(shard_path, "rb") as f:
                shard_dict = pickle.load(f)
        else:
            shard_dict = defaultdict(list)
        return shard_dict

    def _write_dict_to_disk(self, shard_id:int, shard_dict:defaultdict):
        shard_path = os.path.join(self.db_path, f"shard_{shard_id}.pkl")
        with open(shard_path, "wb") as f:
            pickle.dump(shard_dict, f)

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
    
    def get(self, word: str) -> list:
        self._flush_to_disk()

        shard_id = self._get_shard_id(word)
        shard_path = os.path.join(self.db_path, f"shard_{shard_id}.pkl")

        if os.path.exists(shard_path):
            with open(shard_path, "rb") as f:
                shard_dict = pickle.load(f)
                print("dict before returning: ", shard_dict)
                return shard_dict[word]
        return []