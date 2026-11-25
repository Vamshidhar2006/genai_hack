import base64
import cv2
import numpy as np
import requests


class VLMSceneAnalyzer:
    def __init__(
        self,
        model_name: str = "urbansight-qwen",
        ollama_url: str = "http://localhost:11434/api/chat",
        timeout: int = 180,
        temperature: float = 0.35,
        repeat_penalty: float = 1.15,
        num_predict: int = 750,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.num_predict = num_predict

    def _encode_bgr_to_base64(self, bgr_image: np.ndarray, max_side: int = 640) -> str:
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("Input image is empty.")

        h, w = bgr_image.shape[:2]
        scale = min(max_side / max(h, w), 1.0)
        resized = (
            cv2.resize(bgr_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else bgr_image
        )

        ok, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            raise RuntimeError("Failed to encode image to JPEG.")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def _format_detection_context(
        self,
        detections_summary: dict | None = None,
        detections_list: list[dict] | None = None,
    ) -> str:
        lines = []
        if detections_summary:
            counts_str = ", ".join(f"{k}: {v}" for k, v in detections_summary.items())
            lines.append(f"YOLO Counts: {counts_str}")
        else:
            lines.append("YOLO Counts: None detected above threshold.")

        if detections_list:
            top_items = detections_list[:15]
            box_notes = [
                f"{d.get('label')} ({d.get('confidence')}) at {d.get('bbox')}"
                for d in top_items
            ]
            lines.append("Top YOLO Boxes [x1,y1,x2,y2]: " + "; ".join(box_notes))

        return "\n".join(lines)

    def fallback_report(
        self,
        model_name: str,
        detections_summary: dict | None = None,
        detections_list: list[dict] | None = None,
        reason: str = "",
    ) -> dict:
        counts_str = (
            ", ".join(f"{k}: {v}" for k, v in detections_summary.items())
            if detections_summary
            else "None detected above threshold."
        )
        person_count = (detections_summary or {}).get("person", 0)
        vehicle_count = sum(
            (detections_summary or {}).get(k, 0)
            for k in ("car", "motorcycle", "bus", "truck", "bicycle", "train")
        )

        report_text = (
            f"1) Scene Overview\n"
            f"Detection-grounded street summary ({model_name} fallback). Detected objects: {counts_str}.\n\n"
            f"2) Traffic & Mobility\n"
            f"Detected {vehicle_count} vehicle/mobility object(s) in the frame.\n\n"
            f"3) Pedestrian & Safety Observations\n"
            f"Detected {person_count} person detection(s); privacy anonymization applied prior to analysis.\n\n"
            f"4) Infrastructure & Hazards\n"
            f"Not visible from detector output alone.\n\n"
            f"5) Actionable Recommendations\n"
            f"Review annotated bounding boxes and verify live VLM runner status ({reason or 'Ollama vision error'})."
        )
        return {
            "model": model_name,
            "report": report_text,
            "fallback": True,
        }

    def analyze(
        self,
        bgr_image: np.ndarray,
        detections_summary: dict | None = None,
        detections_list: list[dict] | None = None,
        focus_prompt: str = "Analyze street traffic, pedestrian safety, road conditions, and urban infrastructure.",
    ) -> dict:
        img_b64 = self._encode_bgr_to_base64(bgr_image)
        yolo_context = self._format_detection_context(detections_summary, detections_list)

        system_prompt = (
            "You are UrbanSight AI. Write a concise, strictly image-grounded street report. "
            "State ONLY what is visibly present in the image and supported by YOLO detections. "
            "Never guess unobserved details or use filler templates. If an item is not visible, say 'Not visible.' "
            "Use these exact headings:\n"
            "1) Scene Overview\n"
            "2) Traffic & Mobility\n"
            "3) Pedestrian & Safety Observations\n"
            "4) Infrastructure & Hazards\n"
            "5) Actionable Recommendations"
        )

        user_prompt = (
            f"Focus: {focus_prompt}\n"
            f"{yolo_context}\n\n"
            "Describe this exact image accurately and concisely."
        )

        payload = {
            "model": self.model_name,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": self.temperature,
                "repeat_penalty": self.repeat_penalty,
                "num_predict": self.num_predict,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [img_b64],
                },
            ],
        }

        response = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        report_text = data.get("message", {}).get("content", "").strip()
        if not report_text:
            raise RuntimeError("Ollama returned an empty response.")

        return {
            "model": self.model_name,
            "report": report_text,
        }