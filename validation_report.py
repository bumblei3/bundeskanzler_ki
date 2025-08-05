from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import json
import os
import numpy as np

PDF_PATH = "validation_report.pdf"
JSON_PATH = "validation_results.json"
IMG_PATH = "confusion_matrix.png"


def create_validation_report(json_path=JSON_PATH, img_path=IMG_PATH, pdf_path=PDF_PATH):
    if not os.path.exists(json_path):
        print(f"JSON-Datei nicht gefunden: {json_path}")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    accuracy = np.mean([r["is_correct"] for r in results])
    errors = [r for r in results if not r["is_correct"]]
    correct = [r for r in results if r["is_correct"]]
    # PDF erstellen
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height-50, "Modell-Validierungsbericht")
    c.setFont("Helvetica", 12)
    c.drawString(40, height-80, f"Accuracy: {accuracy*100:.2f}%")
    c.drawString(40, height-100, f"Fehleranzahl: {len(errors)}")
    c.drawString(40, height-120, f"Korrekte Vorhersagen: {len(correct)}")
    # Fehlerbeispiele
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height-150, "Top-Fehlerbeispiele:")
    c.setFont("Helvetica", 10)
    for i, r in enumerate(errors[:5]):
        c.drawString(60, height-170-i*15, f"Input: {r['input'][:60]} | Erwartet: {r['expected']} | Vorhergesagt: {r['predicted']} | Prob_richtig: {r['prob_pred']:.4f}")
    # Konfusionsmatrix-Bild
    if os.path.exists(img_path):
        try:
            c.drawImage(img_path, 40, height-400, width=400, height=200)
            c.drawString(40, height-410, "Konfusionsmatrix")
        except Exception as e:
            c.drawString(40, height-410, f"Fehler beim Laden des Bildes: {e}")
    # Abschluss
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(40, 40, "Bericht automatisch erstellt am: "+__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M'))
    c.save()
    print(f"PDF-Validierungsbericht gespeichert: {pdf_path}")

if __name__ == "__main__":
    create_validation_report()
