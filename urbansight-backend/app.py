import base64
import os
import tempfile
from typing import Optional

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from detector import StreetObjectDetector
from pdf_generator import PDFReportGenerator
from privacy_blur import PrivacyBlurrer
from vlm_analyzer import VLMSceneAnalyzer

app = FastAPI(title="UrbanSight AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")

blurrer = PrivacyBlurrer(blur_kernel=(99, 99), padding=0.2, passes=2)
detector = StreetObjectDetector(
    model_name=YOLO_MODEL,
    conf_threshold=0.22,
    iou_threshold=0.45,
    imgsz=640,
)
vlm_analyzer = VLMSceneAnalyzer(model_name=OLLAMA_MODEL, timeout=180, temperature=0.1)
pdf_generator = PDFReportGenerator()


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid float value: {value}") from exc


def decode_upload_to_bgr(file_bytes: bytes) -> np.ndarray:
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")
    return img


def decode_base64_to_bgr(b64_str: str) -> np.ndarray:
    if not b64_str:
        raise HTTPException(status_code=400, detail="Empty base64 image string.")
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64_str)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image string.") from exc
    return decode_upload_to_bgr(raw)


def encode_bgr_to_data_url(bgr_image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode output image.")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@app.get("/health")
def health_check():
    ollama_up = False
    models = []
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            ollama_up = True
            models = [m.get("name") for m in resp.json().get("models", []) if m.get("name")]
    except requests.RequestException:
        ollama_up = False

    return {
        "status": "ok",
        "ollama_reachable": ollama_up,
        "vlm_model": OLLAMA_MODEL,
        "yolo_model": YOLO_MODEL,
        "installed_models": models,
    }


@app.post("/api/preprocess-detect")
async def preprocess_and_detect(
    request: Request,
    file: Optional[UploadFile] = File(None),
    enable_blur: Optional[str] = Form(None),
    street_only: Optional[str] = Form(None),
    conf_threshold: Optional[str] = Form(None),
):
    content: Optional[bytes] = None
    form_blur = enable_blur
    form_street = street_only
    form_conf = conf_threshold

    if file is not None:
        content = await file.read()

    content_type = request.headers.get("content-type", "").lower()

    if content is None and "multipart/form-data" in content_type:
        form_data = await request.form()
        for key in ("file", "image", "frame", "upload"):
            candidate = form_data.get(key)
            if isinstance(candidate, UploadFile):
                content = await candidate.read()
                break
            if isinstance(candidate, str) and candidate.strip():
                content = decode_base64_to_bgr(candidate)
                ok, buf = cv2.imencode(".jpg", content)
                content = buf.tobytes() if ok else None
                break

        if form_blur is None and form_data.get("enable_blur") is not None:
            form_blur = str(form_data.get("enable_blur"))
        if form_street is None and form_data.get("street_only") is not None:
            form_street = str(form_data.get("street_only"))
        if form_conf is None and form_data.get("conf_threshold") is not None:
            form_conf = str(form_data.get("conf_threshold"))

    if content is None and "application/json" in content_type:
        body = await request.json()
        b64_val = body.get("image_base64") or body.get("image") or body.get("file")
        if isinstance(b64_val, str) and b64_val.strip():
            bgr_from_json = decode_base64_to_bgr(b64_val)
            ok, buf = cv2.imencode(".jpg", bgr_from_json)
            if ok:
                content = buf.tobytes()
        if form_blur is None and "enable_blur" in body:
            form_blur = str(body["enable_blur"])
        if form_street is None and "street_only" in body:
            form_street = str(body["street_only"])
        if form_conf is None and "conf_threshold" in body:
            form_conf = str(body["conf_threshold"])

    q = request.query_params
    if form_blur is None and "enable_blur" in q:
        form_blur = q["enable_blur"]
    if form_street is None and "street_only" in q:
        form_street = q["street_only"]
    if form_conf is None and "conf_threshold" in q:
        form_conf = q["conf_threshold"]

    if not content:
        raise HTTPException(
            status_code=400,
            detail="No valid image found in request. Send multipart file under 'file' (or 'image') or base64 JSON.",
        )

    bgr_img = decode_upload_to_bgr(content)
    blur_enabled = _parse_bool(form_blur, True)
    only_street = _parse_bool(form_street, False)
    parsed_conf = _parse_float(form_conf)

    try:
        if blur_enabled:
            blur_res = blurrer.blur_faces(bgr_img)
            working_img = blur_res["image"]
            blurred_count = blur_res["blurred_count"]
            blurred_regions = blur_res["regions"]
        else:
            working_img = bgr_img
            blurred_count = 0
            blurred_regions = []

        det_res = detector.detect(
            working_img,
            street_only=only_street,
            conf_threshold=parsed_conf,
        )

        return {
            "blurred_count": blurred_count,
            "blurred_regions": blurred_regions,
            "total_detections": det_res["total_detections"],
            "counts": det_res["counts"],
            "detections": det_res["detections"],
            "blurred_image": encode_bgr_to_data_url(working_img),
            "annotated_image": encode_bgr_to_data_url(det_res["annotated_image"]),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stage 1 processing failed: {exc}") from exc


class VLMAnalyzeRequest(BaseModel):
    image_base64: str
    counts: dict = {}
    detections: list[dict] = []
    focus_prompt: str = "Analyze street traffic, pedestrian safety, road conditions, and urban infrastructure."
    model_name: Optional[str] = None


@app.post("/api/analyze-vlm")
def analyze_scene_vlm(req: VLMAnalyzeRequest):
    bgr_img = decode_base64_to_bgr(req.image_base64)
    target_model = req.model_name or vlm_analyzer.model_name

    analyzer = (
        VLMSceneAnalyzer(
            model_name=target_model,
            timeout=vlm_analyzer.timeout,
            temperature=vlm_analyzer.temperature,
        )
        if target_model != vlm_analyzer.model_name
        else vlm_analyzer
    )

    try:
        result = analyzer.analyze(
            bgr_image=bgr_img,
            detections_summary=req.counts,
            detections_list=req.detections,
            focus_prompt=req.focus_prompt,
        )
        return result
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach Ollama at localhost:11434: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class PDFReportRequest(BaseModel):
    vlm_report: str
    blurred_count: int = 0
    counts: dict = {}
    focus_prompt: Optional[str] = None
    model_name: Optional[str] = None
    image_base64: Optional[str] = None


@app.post("/api/generate-pdf")
def create_pdf_report(req: PDFReportRequest):
    processed_bgr = decode_base64_to_bgr(req.image_base64) if req.image_base64 else None
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        pdf_generator.generate(
            output_path=tmp_path,
            vlm_report=req.vlm_report,
            blurred_count=req.blurred_count,
            object_counts=req.counts,
            focus_prompt=req.focus_prompt,
            model_name=req.model_name or OLLAMA_MODEL,
            processed_image=processed_bgr,
        )
        return FileResponse(
            path=tmp_path,
            media_type="application/pdf",
            filename="urbansight_report.pdf",
        )
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}") from exc