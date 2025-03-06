import os
import orjson
from pathlib import Path
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import io

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
        self.cache = OrderedDict()  # LRU cache for recently accessed items
        self.cache_limit = 1000  # Max cache size

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

        # Stream writes directly to file instead of batching in memory
        to_delete = []
        with tqdm(self.memory_dict.items(), desc="Storing to disk") as pbar:
            for key, values in pbar:
                if store_all or len(values) > 5:  # Lower threshold for efficiency
                    if key in self.disk_index:
                        # Append new values instead of rewriting
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

        # Remove dumped items from memory
        for key in to_delete:
            del self.memory_dict[key]

        self.data_file_write.flush()
        self._save_disk_index()
        self.memory_size = sum(len(orjson.dumps(v)) for values in self.memory_dict.values() for v in values)

    def get(self, key):
        # Check cache first
        if key in self.cache:
            self.cache.move_to_end(key)  # Mark as recently used
            return self.cache[key]

        # Check memory dict
        if key in self.memory_dict:
            return self.memory_dict[key]

        # Check disk
        if key not in self.disk_index:
            return []

        offset, length = self.disk_index[key]
        self.data_file_read.seek(offset)
        values = orjson.loads(self.data_file_read.read(length).decode())

        # Add to cache (evict oldest if full)
        if len(self.cache) >= self.cache_limit:
            self.cache.popitem(last=False)  # Remove least recently used
        self.cache[key] = values
        return values

    def _compactize(self):
        # Incremental compaction: only rewrite if fragmentation is high
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

    def close(self):
        self._dump(store_all=True)
        self.data_file_write.close()
        self.data_file_read.close()

# Example usage
if __name__ == "__main__":
    dd = DiskDict("test", memory_limit_mb=10)  # 10 MB memory limit
    for i in range(1000):
        dd.put(f"key{i}", f"value{i}")
    print(dd.get("key500"))
    dd.close()