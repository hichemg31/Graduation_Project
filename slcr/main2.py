import cv2
import time

from slcr.preprocess2 import preprocess_frame
from slcr.feature import FeatureExtractor
from slcr.lsh_index import LSHIndex
from slcr.scrt import SCRT
from slcr.model import YOLOInferenceModel
from slcr.slcr_engine import SLCREngine

# Initialize camera
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

engine = SLCREngine(
    extractor=FeatureExtractor(),
    lsh=LSHIndex(),
    scrt=SCRT(),
    model=YOLOInferenceModel("best.pt", conf=0.7)
)

total_frames = 0
reuse_frames = 0

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    img = preprocess_frame(frame)
    result, reused, timing = engine.process(
        raw_frame=frame,
        proc_frame=img
    )

    latency = (time.time() - start) * 1000  # ms

    total_frames += 1
    reuse_frames += int(reused)

    annotated = frame.copy()
    for (x1, y1, x2, y2), (cx, cy) in zip(
        result["boxes"], result["centroids"]
    ):
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.circle(annotated, (cx, cy), 5, (0,0,255), -1)

    y = 30
    print(" ")
    for k in ["feature", "lsh", "ssim", "yolo", "total"]:
        txt = f"{k}: {timing[k]:.1f} ms"
        cv2.putText(
            annotated, txt, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0) if k != "yolo" or not reused else (255, 0, 0),
            2
        )
        print(txt)
        y += 25

    cv2.imshow("SLCR + YOLO", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Reuse rate:", reuse_frames / max(total_frames, 1))
        break

cap.release()
cv2.destroyAllWindows()

