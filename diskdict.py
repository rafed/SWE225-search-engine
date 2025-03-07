import os
import orjson
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import io
from functools import lru_cache

class DiskDict:
    def __init__(self, filename="diskdict"):
        self.data_file_path = "data/" + filename + ".dat"
        self.index_file_path = "data/" + filename + ".idx"
        
        self.data_file_write = open(self.data_file_path, "ab", buffering=io.DEFAULT_BUFFER_SIZE)
        self.data_file_read = open(self.data_file_path, "rb", buffering=io.DEFAULT_BUFFER_SIZE)
        self.disk_index = {}

        if os.path.exists(self.index_file_path):
            self._load_disk_index()

        self.memory_dict = defaultdict(list)
        self.memory_size = 0  
        self.MEMORY_LIMIT = 100 * 1024 * 1024  # 100 MB

    def _load_disk_index(self):
        json_bytes = Path(self.index_file_path).read_bytes()
        self.disk_index = orjson.loads(json_bytes)

    def _save_disk_index(self):
        json_bytes = orjson.dumps(self.disk_index)
        Path(self.index_file_path).write_bytes(json_bytes)

    def put(self, key, value):
        value_bytes = orjson.dumps(value)
        self.memory_dict[key].append(value)
        self.memory_size += len(value_bytes) + len(str(key).encode())

        if self.memory_size >= self.MEMORY_LIMIT:
            self._dump()

    def _dump(self, store_all=False):
        if not self.memory_dict:
            return

        to_delete = []
        with tqdm(self.memory_dict.items(), desc="Storing to disk") as pbar:
            for key, values in pbar:
                if store_all or len(values) > 5:  # Lower threshold for efficiency
                    if key in self.disk_index:
                        offset, length = self.disk_index[key]
                        self.data_file_read.seek(offset)
                        old_values = orjson.loads(self.data_file_read.read(length).decode())
                        values = old_values + values

                    value_bytes = orjson.dumps(values) + b"\n"
                    current_offset = self.data_file_write.tell()
                    self.data_file_write.write(value_bytes)
                    self.disk_index[key] = (current_offset, len(value_bytes))
                    to_delete.append(key)
                    pbar.set_description(f"Storing {key} ({len(values)} items)")

        for key in to_delete:
            del self.memory_dict[key]

        self.data_file_write.flush()
        self._save_disk_index()
        self.memory_size = sum(len(orjson.dumps(v)) for values in self.memory_dict.values() for v in values)

    @lru_cache(maxsize=1000)
    def get(self, key):
        if key not in self.disk_index:
            return []

        offset, length = self.disk_index[key]
        self.data_file_read.seek(offset)
        values = orjson.loads(self.data_file_read.read(length).decode())

        return values

    def _compactize(self):
        self.data_file_write.close()
        self.data_file_read.close()

        new_data_file_path = self.data_file_path + ".tmp"
        new_index = {}

        with open(self.data_file_path, "rb") as old_f, \
             open(new_data_file_path, "wb", buffering=io.DEFAULT_BUFFER_SIZE) as new_f:
            for key, (offset, length) in self.disk_index.items():
                old_f.seek(offset)
                value = old_f.read(length)
                new_offset = new_f.tell()
                new_f.write(value)
                new_index[key] = (new_offset, len(value))

        os.replace(new_data_file_path, self.data_file_path)
        self.disk_index = new_index
        self._save_disk_index()

        self.data_file_write = open(self.data_file_path, "ab", buffering=io.DEFAULT_BUFFER_SIZE)
        self.data_file_read = open(self.data_file_path, "rb", buffering=io.DEFAULT_BUFFER_SIZE)

    def load_top_k_words_in_cache(self, k=800):
        with open("data/top_k_words.txt", "r") as f:
            for i, line in enumerate(f):
                if i == k:
                    break

                word = line.strip()
                self.get(word)

    def close(self):
        self._dump(store_all=True)
        self.data_file_write.close()
        self.data_file_read.close()
