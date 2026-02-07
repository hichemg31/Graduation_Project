class SCRT:
    def __init__(self):
        self.records = []

    def add(self, feature, result, frame):
        self.records.append({
            "feature": feature,
            "result": result,
            "frame": frame,      # stored for SSIM
            "reuse_count": 0
        })

    def get(self, idx):
        return self.records[idx]

    def increment_reuse(self, idx):
        self.records[idx]["reuse_count"] += 1
