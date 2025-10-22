"""
Automated PDF report generation for Horno Prediction Project.
Generates a concise report including metrics and key figures.
"""

import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib import utils


def _draw_image(c, img_path, x, y, width=None, height=None, max_width=16*cm):
    if not os.path.exists(img_path):
        return
    img = utils.ImageReader(img_path)
    iw, ih = img.getSize()
    if width is None and height is None:
        scale = max_width / float(iw)
        width = iw * scale
        height = ih * scale
    c.drawImage(img_path, x, y, width=width, height=height, preserveAspectRatio=True, mask='auto')


def generate_pdf_report(metrics_path='results/metrics.json',
                        training_curves_path='results/training_curves.png',
                        confusion_matrix_path='results/confusion_matrix.png',
                        output_path='results/report.pdf'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, height - 2*cm, "Horno Prediction - Model Report")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, height - 2.7*cm, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Metrics section
    y = height - 4*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Metrics")
    y -= 0.7*cm
    c.setFont("Helvetica", 10)
    if metrics:
        for k in sorted(metrics.keys()):
            if k == 'timestamp':
                continue
            value = metrics[k]
            c.drawString(2.2*cm, y, f"{k}: {value}")
            y -= 0.5*cm
            if y < 3*cm:
                c.showPage()
                y = height - 2*cm
    else:
        c.drawString(2.2*cm, y, "No metrics found.")
        y -= 0.7*cm

    # Training curves
    if y < 12*cm:
        c.showPage()
        y = height - 2*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Training Curves")
    y -= 0.7*cm
    _draw_image(c, training_curves_path, 2*cm, y - 9*cm, max_width=17*cm)
    y -= 10*cm

    # Confusion matrix if exists
    if os.path.exists(confusion_matrix_path):
        if y < 12*cm:
            c.showPage()
            y = height - 2*cm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, y, "Confusion Matrix")
        y -= 0.7*cm
        _draw_image(c, confusion_matrix_path, 2*cm, y - 9*cm, max_width=17*cm)
        y -= 10*cm

    c.showPage()
    c.save()
    return output_path


if __name__ == '__main__':
    path = generate_pdf_report()
    print(f"Report generated at: {path}")



