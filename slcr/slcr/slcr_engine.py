import time

import cv2
import numpy as np
from config import SSIM_THRESHOLD
from slcr.similarity import compute_ssim


class SLCREngine:
    def __init__(self, extractor, lsh, scrt, model):
        self.extractor = extractor
        self.lsh = lsh
        self.scrt = scrt
        self.model = model
        self.prev_frame = None
        self.last_result = None

    def process(self, raw_frame, proc_frame):
        timing = {}
        t0 = time.perf_counter()

        # 1. Feature extraction
        t = time.perf_counter()
        feat = self.extractor.extract(proc_frame)
        timing["feature"] = (time.perf_counter() - t) * 1000

        # 2. LSH search
        t = time.perf_counter()
        idxs = self.lsh.search(feat, k=1)
        timing["lsh"] = (time.perf_counter() - t) * 1000

        # 3. No candidate → YOLO
        if idxs is None:
            result, infer_time = self._infer_and_store(raw_frame, proc_frame, feat)
            timing["ssim"] = 0.0
            timing["yolo"] = infer_time
            timing["total"] = (time.perf_counter() - t0) * 1000
            return result, False, timing

        # ---- Fast motion check ----
        if self.prev_frame is not None:
            diff = np.mean(np.abs(proc_frame - self.prev_frame))
            if diff < 0.01:
                # Reuse nearest LSH candidate directly
                idx = idxs[0]
                self.scrt.increment_reuse(idx)
                timing["ssim"] = 0.0
                timing["yolo"] = 0.0
                timing["total"] = (time.perf_counter() - t0) * 1000
                self.prev_frame = proc_frame
                return self.scrt.get(idx)["result"], True, timing
            

        # 4. SSIM comparison
        t = time.perf_counter()
        best_idx = None
        best_ssim = -1

        for idx in idxs:
            candidate = self.scrt.get(idx)
            score = compute_ssim(proc_frame, candidate["frame"])
            if score > best_ssim:
                best_ssim = score
                best_idx = idx

        timing["ssim"] = (time.perf_counter() - t) * 1000

        # 5. Decision
        if best_ssim > SSIM_THRESHOLD:
            self.scrt.increment_reuse(best_idx)
            timing["yolo"] = 0.0
            timing["total"] = (time.perf_counter() - t0) * 1000
            return self.scrt.get(best_idx)["result"], True, timing

        # 6. YOLO inference
        result, infer_time = self._infer_and_store(raw_frame, proc_frame, feat)
        timing["yolo"] = infer_time
        timing["total"] = (time.perf_counter() - t0) * 1000
        self.prev_frame = proc_frame
        return result, False, timing

    def _infer_and_store(self, raw_frame, proc_frame, feat):
        t = time.perf_counter()
        result = self.model.infer(raw_frame)
        infer_time = (time.perf_counter() - t) * 1000

        small = cv2.resize(
            cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY),
            (64, 64)
        )
        self.scrt.add(feat, result, small)
      #  self.scrt.add(feat, result, proc_frame)
        self.lsh.add(feat)
        self.last_result = result


        return result, infer_time
