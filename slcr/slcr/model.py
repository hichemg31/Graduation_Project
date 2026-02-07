from ultralytics import YOLO

class YOLOInferenceModel:
    def __init__(self, model_path="best.pt", conf=0.7):
        self.model = YOLO(model_path)
        self.conf = conf

    def infer(self, frame):
        """
        Heavy computation (YOLO inference)
        """
        results = self.model(frame, conf=self.conf)[0]

        boxes = []
        centroids = []
        classes = []
        confidences = []

        for box in results.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            cx = int((x_min + x_max) / 2)
            cy = int((y_min + y_max) / 2)

            boxes.append((x_min, y_min, x_max, y_max))
            centroids.append((cx, cy))
            classes.append(cls)
            confidences.append(conf)

        return {
            "boxes": boxes,
            "centroids": centroids,
            "classes": classes,
            "confidences": confidences
        }
