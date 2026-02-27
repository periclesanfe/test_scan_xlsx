import io
import os
import uuid

import cv2
import numpy as np
import qrcode
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT


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
    option_style = ParagraphStyle("Option", parent=styles["Normal"], fontSize=10, leftIndent=10)

    elements = []

    # Title
    elements.append(Paragraph(test.title, title_style))
    if test.description:
        elements.append(Paragraph(test.description, subtitle_style))
    elements.append(Spacer(1, 6 * mm))

    # Header table: Name + QR code
    qr_img_buf = _make_qr(f"TEST:{test.id}")
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
        option_row = []
        for opt in options:
            option_row.append(f"( ) {opt}")
        if option_row:
            elements.append(Paragraph("    " + "     ".join(option_row), option_style))
        elements.append(Spacer(1, 4 * mm))

    doc.build(elements)


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


def process_scan_image(image_path, test):
    """
    Basic OMR: detect filled circles/bubbles in the uploaded scan.
    Returns a dict {question_index (1-based): selected_option_text}.
    Falls back to empty dict if detection is unreliable.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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
        return {}

    circles = np.round(circles[0, :]).astype("int")

    # Sort circles top-to-bottom, then left-to-right
    circles = sorted(circles, key=lambda c: (c[1] // 20, c[0]))

    questions = test.questions
    results = {}
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

    return results


def import_questions_from_xlsx(file_path):
    """
    Import questions from an Excel file.
    Expected columns: text, answer_type, correct_answer, custom_options
    Returns list of dicts.
    """
    import openpyxl
    wb = openpyxl.load_workbook(file_path)
    ws = wb.active
    headers = [str(cell.value).strip().lower() if cell.value else "" for cell in next(ws.iter_rows(min_row=1, max_row=1))]
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
