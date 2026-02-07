import os
import time
from slcr.preprocess import preprocess
from slcr.feature import FeatureExtractor
from slcr.lsh_index import LSHIndex
from slcr.scrt import SCRT
from slcr.model import InferenceModel
from slcr.slcr_engine import SLCREngine

DATA_DIR = "images"

engine = SLCREngine(
    extractor=FeatureExtractor(),
    lsh=LSHIndex(),
    scrt=SCRT(),
    model=InferenceModel()
)

reuse_count = 0
total = 0

for img_file in sorted(os.listdir(DATA_DIR)):
    img = preprocess(os.path.join(DATA_DIR, img_file))
    start = time.time()
    _, reused = engine.process(img)
    latency = time.time() - start

    reuse_count += int(reused)
    total += 1

    print(f"{img_file}: reused={reused}, latency={latency:.4f}s")

print("Reuse rate:", reuse_count / total)
