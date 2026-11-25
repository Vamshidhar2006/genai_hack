import cv2
import numpy as np


class PrivacyBlurrer:
    def __init__(
        self,
        blur_kernel: tuple[int, int] = (99, 99),
        padding: float = 0.2,
        passes: int = 2,
    ):
        kx = blur_kernel[0] if blur_kernel[0] % 2 == 1 else blur_kernel[0] + 1
        ky = blur_kernel[1] if blur_kernel[1] % 2 == 1 else blur_kernel[1] + 1
        self.base_kernel = (max(3, min(kx, 151)), max(3, min(ky, 151)))
        self.padding = padding
        self.passes = max(1, passes)

        frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"

        self.frontal_cascade = cv2.CascadeClassifier(frontal_path)
        if self.frontal_cascade.empty():
            raise RuntimeError(f"Failed to load frontal face Haar cascade from {frontal_path}")

        profile_cascade = cv2.CascadeClassifier(profile_path)
        self.profile_cascade = None if profile_cascade.empty() else profile_cascade

    def _expand_box(
        self, x: int, y: int, w: int, h: int, img_w: int, img_h: int
    ) -> tuple[int, int, int, int]:
        pad_x = int(w * self.padding)
        pad_y = int(h * self.padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)
        return x1, y1, x2, y2

    @staticmethod
    def _merge_overlapping_boxes(
        boxes: list[tuple[int, int, int, int]], iou_threshold: float = 0.25
    ) -> list[tuple[int, int, int, int]]:
        if not boxes:
            return []

        merged = list(boxes)
        changed = True

        while changed:
            changed = False
            next_boxes: list[tuple[int, int, int, int]] = []
            used = [False] * len(merged)

            for i in range(len(merged)):
                if used[i]:
                    continue
                ax1, ay1, ax2, ay2 = merged[i]
                area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)

                for j in range(i + 1, len(merged)):
                    if used[j]:
                        continue
                    bx1, by1, bx2, by2 = merged[j]
                    ix1 = max(ax1, bx1)
                    iy1 = max(ay1, by1)
                    ix2 = min(ax2, bx2)
                    iy2 = min(ay2, by2)

                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
                        union = area_a + area_b - inter
                        iou = inter / union if union > 0 else 0.0
                        overlap_smaller = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0

                        if iou >= iou_threshold or overlap_smaller >= 0.5:
                            ax1 = min(ax1, bx1)
                            ay1 = min(ay1, by1)
                            ax2 = max(ax2, bx2)
                            ay2 = max(ay2, by2)
                            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
                            used[j] = True
                            changed = True

                used[i] = True
                next_boxes.append((ax1, ay1, ax2, ay2))

            merged = next_boxes

        return merged

    def _dynamic_kernel(self, roi_w: int, roi_h: int) -> tuple[int, int]:
        kx = max(self.base_kernel[0], int(roi_w * 0.5) | 1)
        ky = max(self.base_kernel[1], int(roi_h * 0.5) | 1)
        kx = min(kx, 151)
        ky = min(ky, 151)
        return (kx, ky)

    def blur_faces(
        self,
        bgr_image: np.ndarray,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (28, 28),
    ) -> dict:
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("Input image is empty.")

        img_h, img_w = bgr_image.shape[:2]
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        raw_boxes: list[tuple[int, int, int, int]] = []

        for (x, y, w, h) in self.frontal_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.15,
            minNeighbors=min_neighbors,
            minSize=min_size,
        ):
            raw_boxes.append(self._expand_box(int(x), int(y), int(w), int(h), img_w, img_h))

        if self.profile_cascade is not None:
            for (x, y, w, h) in self.profile_cascade.detectMultiScale(
                gray_eq,
                scaleFactor=1.15,
                minNeighbors=min_neighbors,
                minSize=min_size,
            ):
                raw_boxes.append(self._expand_box(int(x), int(y), int(w), int(h), img_w, img_h))

        merged_boxes = self._merge_overlapping_boxes(raw_boxes)
        output = bgr_image.copy()
        blurred_regions = []

        for (x1, y1, x2, y2) in merged_boxes:
            if x2 > x1 and y2 > y1:
                roi = output[y1:y2, x1:x2]
                ksize = self._dynamic_kernel(x2 - x1, y2 - y1)
                for _ in range(self.passes):
                    roi = cv2.GaussianBlur(roi, ksize, 0)
                output[y1:y2, x1:x2] = roi
                blurred_regions.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

        return {
            "image": output,
            "blurred_count": len(blurred_regions),
            "regions": blurred_regions,
        }