import io
import json
import os
import re

import cv2
import numpy as np
import qrcode
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, Flowable
from reportlab.lib.enums import TA_CENTER


class BubbleOptionsRow(Flowable):
    """Draw OMR-friendly bubble options as real circles followed by labels."""

    def __init__(self, options, font_name="Helvetica", font_size=10, left_indent=10):
        super().__init__()
        self.options = options
        self.font_name = font_name
        self.font_size = font_size
        self.left_indent = left_indent
        self.bubble_radius = 2 * mm
        self.label_gap = 2 * mm
        self.option_gap = 6 * mm
        self.height = max((2 * self.bubble_radius) + (0.8 * mm), self.font_size + 2)

    def wrap(self, avail_width, avail_height):
        self.width = avail_width
        return avail_width, self.height

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFont(self.font_name, self.font_size)
        c.setLineWidth(1)

        x = self.left_indent
        y = self.height / 2.0
        for opt in self.options:
            cx = x + self.bubble_radius
            if cx + self.bubble_radius > self.width:
                break
            c.circle(cx, y, self.bubble_radius, stroke=1, fill=0)
            label_x = cx + self.bubble_radius + self.label_gap
            label = str(opt)
            c.drawString(label_x, y - (self.font_size * 0.35), label)
            label_w = stringWidth(label, self.font_name, self.font_size)
            x = label_x + label_w + self.option_gap

        c.restoreState()


def generate_test_pdf(test, output_path):
    """Generate a printable PDF for the given test."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Heading1"], alignment=TA_CENTER, fontSize=18)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], alignment=TA_CENTER, fontSize=11)
    question_style = ParagraphStyle("Question", parent=styles["Normal"], fontSize=11, spaceAfter=4)

    elements = []

    # Title
    elements.append(Paragraph(test.title, title_style))
    if test.description:
        elements.append(Paragraph(test.description, subtitle_style))
    elements.append(Spacer(1, 6 * mm))

    # Header table: Name + QR code
    template = build_omr_template(test)
    qr_img_buf = _make_qr(f"OMR_TEMPLATE:{template['template_id']}")
    qr_rl = Image(qr_img_buf, width=25 * mm, height=25 * mm)

    header_data = [
        [Paragraph("<b>Nome:</b> ___________________________________", styles["Normal"]),
         Paragraph("<b>Data:</b> _______________", styles["Normal"]),
         qr_rl],
        [Paragraph("<b>Observação:</b> ________________________________________________", styles["Normal"]),
         "", ""],
    ]
    header_table = Table(
        header_data,
        colWidths=[90 * mm, 50 * mm, 30 * mm],
        rowHeights=[12 * mm, 10 * mm],
    )
    header_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
        ("GRID", (0, 0), (-1, 0), 0.5, colors.black),
        ("SPAN", (2, 0), (2, 1)),
        ("SPAN", (0, 1), (1, 1)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 6 * mm))

    # Questions
    for idx, tq in enumerate(test.questions, start=1):
        q = tq.question
        elements.append(Paragraph(f"<b>{idx}.</b> {q.text}", question_style))
        options = q.get_options()
        if options:
            elements.append(BubbleOptionsRow(options))
        elements.append(Spacer(1, 4 * mm))

    doc.build(elements, onFirstPage=lambda canvas, doc_ref: _draw_fiducials(canvas, doc_ref, template))


def _make_qr(data: str) -> io.BytesIO:
    qr = qrcode.QRCode(box_size=4, border=1)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# Minimum fill ratio (0–1) to consider a bubble as marked.
# 40% fill accounts for partial or imprecise shading while avoiding false positives.
OMR_FILL_THRESHOLD = 0.4


def extract_scan_metadata(image_path):
    """
    Try to extract respondent metadata (name/observation) from the scan header.
    OCR is optional; if unavailable, returns empty fields.
    """
    img, source_method = _read_scan_image(image_path)
    if img is None:
        return {"respondent_name": None, "observation": None, "method": source_method}

    try:
        import pytesseract
    except Exception:
        return {"respondent_name": None, "observation": None, "method": "ocr_unavailable"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boosted = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    cleaned = cv2.GaussianBlur(boosted, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    text = ""
    try:
        text = pytesseract.image_to_string(thresh, lang="por+eng")
    except Exception:
        try:
            text = pytesseract.image_to_string(thresh, lang="eng")
        except Exception:
            return {"respondent_name": None, "observation": None, "method": "ocr_failed"}

    respondent_name = _extract_labeled_text(
        text,
        label_patterns=[r"nome", r"name"],
        stop_patterns=[r"data", r"date", r"observa[çc][aã]o", r"observation"],
    )
    observation = _extract_labeled_text(
        text,
        label_patterns=[r"observa[çc][aã]o", r"observation"],
        stop_patterns=[r"nome", r"name", r"data", r"date"],
    )

    return {
        "respondent_name": respondent_name,
        "observation": observation,
        "method": f"ocr_tesseract_{source_method}",
    }


def _extract_labeled_text(text, label_patterns, stop_patterns):
    if not text:
        return None

    label_alt = "|".join(label_patterns)
    stop_alt = "|".join(stop_patterns)
    regex = re.compile(
        rf"(?is)\b(?:{label_alt})\b\s*[:\-]?\s*(.+?)(?=(?:\b(?:{stop_alt})\b\s*[:\-]?)|\n|$)"
    )
    match = regex.search(text)
    if not match:
        return None

    value = match.group(1)
    value = value.replace("_", " ").strip(" .;:-\t\r\n")
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return None
    return value


def process_scan_image(image_path, test, with_confidence=False):
    """
    Basic OMR: detect filled circles/bubbles in the uploaded scan.
    Returns a dict {question_index (1-based): selected_option_text}.
    Falls back to empty dict if detection is unreliable.
    """
    img, _ = _read_scan_image(image_path)
    if img is None:
        return ({}, {}) if with_confidence else {}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold is more robust to uneven lighting than global Otsu alone.
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )
    aligned = _align_with_fiducials(img)
    if aligned is not None:
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
        )

    # Detect circles via HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=15,
        param1=50,
        param2=30,
        minRadius=8,
        maxRadius=20,
    )

    if circles is None:
        return ({}, {}) if with_confidence else {}

    circles = np.round(circles[0, :]).astype("int")

    # Sort circles top-to-bottom, then left-to-right
    circles = sorted(circles, key=lambda c: (c[1] // 20, c[0]))

    questions = test.questions
    results = {}
    confidences = {}
    circle_idx = 0
    for tq in questions:
        q = tq.question
        options = q.get_options()
        if not options:
            continue
        q_circles = circles[circle_idx: circle_idx + len(options)]
        circle_idx += len(options)
        best = None
        best_fill = -1
        for i, (cx, cy, r) in enumerate(q_circles):
            mask = np.zeros(gray.shape, dtype="uint8")
            cv2.circle(mask, (cx, cy), r - 2, 255, -1)
            filled = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            total_px = cv2.countNonZero(mask)
            ratio = filled / total_px if total_px > 0 else 0
            if ratio > OMR_FILL_THRESHOLD and ratio > best_fill:
                best_fill = ratio
                best = options[i]
        if best:
            results[tq.question_id] = best
            confidences[tq.question_id] = round(float(best_fill), 4)
    return (results, confidences) if with_confidence else results


def _read_scan_image(image_path):
    """Read scan input (image or PDF first page) and return (bgr_image_or_none, method)."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".pdf":
        try:
            import pypdfium2 as pdfium
        except Exception:
            return None, "pdf_renderer_unavailable"
        try:
            pdf = pdfium.PdfDocument(image_path)
            if len(pdf) < 1:
                return None, "pdf_empty"
            page = pdf[0]
            pil_img = page.render(scale=2.0).to_pil()
            arr = np.array(pil_img)
            if arr.ndim == 2:
                bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.shape[2] == 4:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return bgr, "pdf_page1_rendered"
        except Exception:
            return None, "pdf_render_failed"

    img = cv2.imread(image_path)
    if img is None:
        return None, "image_unreadable"
    return img, "image_loaded"


def import_questions_from_xlsx(file_path):
    """
    Import questions from an Excel file.
    Expected columns: text, answer_type, correct_answer, custom_options
    Returns list of dicts.
    """
    import openpyxl
    wb = openpyxl.load_workbook(file_path)
    ws = wb.active
    headers = [
        str(cell.value).strip().lower() if cell.value else ""
        for cell in next(ws.iter_rows(min_row=1, max_row=1))
    ]
    questions = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not any(row):
            continue
        data = dict(zip(headers, row))
        q = {
            "text": str(data.get("text") or data.get("pergunta") or "").strip(),
            "answer_type": str(data.get("answer_type") or data.get("tipo") or "yes_no").strip().lower(),
            "correct_answer": str(data.get("correct_answer") or data.get("resposta_correta") or "").strip(),
            "custom_options": str(data.get("custom_options") or data.get("opcoes") or "").strip(),
        }
        if q["text"]:
            if q["answer_type"] not in ("yes_no", "likert", "custom"):
                import warnings
                warnings.warn(
                    f"Tipo de resposta desconhecido '{q['answer_type']}'; usando 'yes_no'.",
                    UserWarning,
                    stacklevel=2,
                )
                q["answer_type"] = "yes_no"
            questions.append(q)
    return questions


def import_questions_from_json(file_obj_or_path):
    """
    Import questions from JSON payload/file.
    Accepted payload:
    {
      "questions": [
        {
          "text": "...",
          "answer_type": "yes_no|likert|custom",
          "correct_answer": "...",
          "custom_options": "A, B, C"
        }
      ]
    }
    """
    if isinstance(file_obj_or_path, str):
        with open(file_obj_or_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        raw = file_obj_or_path.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)

    rows = payload.get("questions", [])
    questions = []
    for row in rows:
        q = {
            "text": str(row.get("text") or row.get("pergunta") or "").strip(),
            "answer_type": str(row.get("answer_type") or row.get("tipo") or "yes_no").strip().lower(),
            "correct_answer": str(row.get("correct_answer") or row.get("resposta_correta") or "").strip(),
            "custom_options": str(row.get("custom_options") or row.get("opcoes") or "").strip(),
        }
        if q["text"]:
            questions.append(q)
    return questions


def export_questions_to_xlsx(questions):
    """Return BytesIO with questions exported to xlsx."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Perguntas"
    headers = ["text", "answer_type", "correct_answer", "custom_options"]
    ws.append(headers)
    for q in questions:
        ws.append([q.text, q.answer_type, q.correct_answer or "", q.custom_options or ""])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


def build_omr_template(test):
    """Build a versioned OMR template metadata descriptor for the test."""
    questions = []
    for order, tq in enumerate(test.questions, start=1):
        q = tq.question
        questions.append(
            {
                "order": order,
                "question_id": q.id,
                "answer_type": q.answer_type,
                "options": q.get_options(),
                "option_count": len(q.get_options()),
            }
        )
    return {
        "template_version": "1.1",
        "template_id": f"test-{test.id}-v1_1",
        "fiducials": [
            {"name": "tl", "x_norm": 0.03, "y_norm": 0.03},
            {"name": "tr", "x_norm": 0.97, "y_norm": 0.03},
            {"name": "bl", "x_norm": 0.03, "y_norm": 0.97},
            {"name": "br", "x_norm": 0.97, "y_norm": 0.97},
        ],
        "questions": questions,
    }


def _draw_fiducials(canvas, doc_ref, template):
    width, height = A4
    size = 5 * mm
    for fid in template["fiducials"]:
        x = fid["x_norm"] * width
        y = fid["y_norm"] * height
        canvas.setFillColorRGB(0, 0, 0)
        canvas.rect(x - (size / 2), y - (size / 2), size, size, stroke=0, fill=1)


def _align_with_fiducials(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 80:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.08 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h) if h else 0
            if 0.6 <= ratio <= 1.4:
                squares.append((area, x, y, w, h))

    if len(squares) < 4:
        return None
    squares = sorted(squares, key=lambda s: s[0], reverse=True)[:8]
    centers = np.array([[x + w / 2.0, y + h / 2.0] for _, x, y, w, h in squares], dtype=np.float32)
    if centers.shape[0] < 4:
        return None

    s = centers.sum(axis=1)
    diff = np.diff(centers, axis=1).reshape(-1)
    tl = centers[np.argmin(s)]
    br = centers[np.argmax(s)]
    tr = centers[np.argmin(diff)]
    bl = centers[np.argmax(diff)]

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    out_w, out_h = 1654, 2339  # ~A4 @ 150 DPI
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (out_w, out_h))
