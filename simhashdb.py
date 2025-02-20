import threading
from rocksdict import Rdict
import simhash

class SimhashManager:
    _shared_state = {}  # Borg shared state
    _lock = threading.Lock()

    def __init__(self, db_path="./fingerprints"):
        self.__dict__ = self._shared_state
        if not hasattr(self, "initialized"):
            with self._lock:
                if not hasattr(self, "initialized"):
                    self._init_db(db_path)
                    self.initialized = True

    def _init_db(self, db_path):
        self.db_path = db_path
        self.db = Rdict(self.db_path)
        self.db_lock = threading.Lock()
        self.counter = 0

    def close(self):
        self.db.close()

    def flush_db(self):
        self.db.close()
        self.db = Rdict(self.db_path)

    def _hash_token_to_int(self, token):
        return abs(hash(token))

    def _get_fingerprint(self, content):
        tokens = content.split()
        tokens = [self._hash_token_to_int(token) for token in tokens]
        return simhash.compute(tokens)

    def exists_duplicate(self, url, new_content):
        new_finger_print = self._get_fingerprint(new_content)
        
        with self.db_lock: 
            for current_url, current_fingerprint in self.db.items():
                distance = simhash.num_differing_bits(new_finger_print, current_fingerprint)
                if distance <= 2:
                    print(f"Duplicate page: {url} matched with URL: {current_url}, {distance}")
                    return True

            self.db[url] = new_finger_print
            self.counter += 1

            if self.counter % 1000 == 0:
                self.flush_db()
            
        return False