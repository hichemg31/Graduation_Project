import faiss

class LSHIndex:
    def __init__(self):
        self.index = faiss.IndexLSH(1024, 128)

    def add(self, vec):
        self.index.add(vec.reshape(1, -1))

    def search(self, vec, k=1):
        if self.index.ntotal == 0:
            return None
        _, idx = self.index.search(vec.reshape(1, -1), k)
        return idx[0]
