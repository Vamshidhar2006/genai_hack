from collections import Counter
import cv2
import numpy as np
from ultralytics import YOLO

STREET_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
}


class StreetObjectDetector:
    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        conf_threshold: float = 0.22,
        iou_threshold: float = 0.45,
        imgsz: int = 960,
    ):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

    def detect(
        self,
        bgr_image: np.ndarray,
        street_only: bool = False,
        conf_threshold: float | None = None,
    ) -> dict:
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("Input image is empty.")

        conf = self.conf_threshold if conf_threshold is None else conf_threshold
        results = self.model.predict(
            source=bgr_image,
            imgsz=self.imgsz,
            conf=conf,
            iou=self.iou_threshold,
            verbose=False,
        )
        result = results[0]
        names = result.names

        detections = []
        counts = Counter()
        annotated = bgr_image.copy()

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            label = str(names[cls_id])
            if street_only and label not in STREET_CLASSES:
                continue

            score = round(float(box.conf[0].item()), 3)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            detections.append({
                "label": label,
                "confidence": score,
                "bbox": [x1, y1, x2, y2],
            })
            counts[label] += 1

        detections.sort(key=lambda item: item["confidence"], reverse=True)

        for item in detections:
            x1, y1, x2, y2 = item["bbox"]
            label = item["label"]
            score = item["confidence"]
            tag = f"{label} {score:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (16, 185, 129), 2)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top_y = max(th + 8, y1)
            cv2.rectangle(
                annotated,
                (x1, top_y - th - 6),
                (x1 + tw + 8, top_y),
                (16, 185, 129),
                -1,
            )
            cv2.putText(
                annotated,
                tag,
                (x1 + 4, top_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return {
            "total_detections": len(detections),
            "counts": dict(counts),
            "detections": detections,
            "annotated_image": annotated,
        }