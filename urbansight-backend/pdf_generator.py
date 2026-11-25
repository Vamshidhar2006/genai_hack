import io
import os
from datetime import datetime
import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class PDFReportGenerator:
    def __init__(self, page_size=A4):
        self.page_size = page_size
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            "UrbanSightTitle",
            parent=self.styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=6,
        )
        self.heading_style = ParagraphStyle(
            "UrbanSightHeading",
            parent=self.styles["Heading2"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1E3A8A"),
            spaceBefore=12,
            spaceAfter=6,
        )
        self.body_style = ParagraphStyle(
            "UrbanSightBody",
            parent=self.styles["BodyText"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=6,
        )

    def _bgr_to_rl_image(
        self,
        bgr_image: np.ndarray,
        max_width: float = 6.2 * inch,
        max_height: float = 3.6 * inch,
    ) -> RLImage | None:
        if bgr_image is None or bgr_image.size == 0:
            return None
        ok, buf = cv2.imencode(".jpg", bgr_image)
        if not ok:
            return None
        img_io = io.BytesIO(buf.tobytes())
        img = RLImage(img_io)
        if img.imageWidth and img.imageHeight:
            scale = min(max_width / img.imageWidth, max_height / img.imageHeight, 1.0)
            img.drawWidth = img.imageWidth * scale
            img.drawHeight = img.imageHeight * scale
        return img

    def generate(
        self,
        output_path: str,
        vlm_report: str,
        blurred_count: int = 0,
        object_counts: dict | None = None,
        focus_prompt: str | None = None,
        model_name: str = "urbansight-qwen",
        processed_image: np.ndarray | None = None,
    ) -> str:
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.page_size,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        story = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        story.append(Paragraph("UrbanSight AI — Street Scene Analysis Report", self.title_style))
        story.append(Paragraph(f"Generated: {timestamp} | VLM Model: {model_name}", self.body_style))
        story.append(Spacer(1, 0.1 * inch))

        counts_text = (
            ", ".join(f"{k}: {v}" for k, v in object_counts.items())
            if object_counts
            else "None detected"
        )

        summary_rows = [
            ["Privacy Faces Blurred", str(blurred_count)],
            ["YOLO Detections", counts_text],
            ["Focus Prompt", focus_prompt or "General street scene & safety analysis"],
        ]

        summary_table = Table(summary_rows, colWidths=[1.8 * inch, 4.9 * inch])
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(summary_table)

        rl_img = self._bgr_to_rl_image(processed_image)
        if rl_img is not None:
            story.append(Paragraph("Processed Street Frame", self.heading_style))
            story.append(rl_img)

        story.append(Paragraph("VLM Scene Assessment", self.heading_style))
        for paragraph in vlm_report.strip().split("\n\n"):
            clean_line = paragraph.replace("\n", "<br/>")
            if clean_line.strip():
                story.append(Paragraph(clean_line, self.body_style))

        doc.build(story)
        return output_path