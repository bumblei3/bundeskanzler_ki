import streamlit as st
import json
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Modell-Validierung", layout="wide")
st.title("Modell-Validierung & Fehleranalyse")

# Lade JSON-Ergebnisse
try:
    with open("validation_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    df = pd.DataFrame(results)
except Exception as e:
    st.error(f"Fehler beim Laden der Ergebnisse: {e}")
    st.stop()

# Metriken anzeigen
st.header("Metriken")
accuracy = df["is_correct"].mean()
f1 = st.text_input("F1-Score (siehe Konsole)", "Berechnung siehe Skript")
precision = st.text_input("Precision (siehe Konsole)", "Berechnung siehe Skript")
recall = st.text_input("Recall (siehe Konsole)", "Berechnung siehe Skript")
st.metric("Accuracy", f"{accuracy*100:.2f}%")

# Konfusionsmatrix anzeigen
st.header("Konfusionsmatrix")
try:
    img = Image.open("confusion_matrix.png")
    st.image(img, caption="Konfusionsmatrix", use_column_width=True)
except Exception as e:
    st.warning(f"Konfusionsmatrix-Bild nicht gefunden: {e}")

# Fehlerbeispiele
st.header("Top-Fehlerbeispiele")
num_errors = st.slider("Anzahl Fehlerbeispiele", 1, 20, 5)
errors = df[df["is_correct"] == False].sort_values("prob_pred").head(num_errors)
st.dataframe(errors[["input", "expected", "predicted", "prob_pred"]])

# Häufigste Verwechslungen
st.header("Häufigste Verwechslungen")
confusions = {}
for _, row in df.iterrows():
    if not row["is_correct"]:
        key = (row["expected"], row["predicted"])
        confusions[key] = confusions.get(key, 0) + 1
confusions_sorted = sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:5]
for (true_cls, pred_cls), count in confusions_sorted:
    st.write(f"{true_cls} → {pred_cls}: {count}x")

# Filter und Download
st.header("Ergebnisse filtern & exportieren")
selected_class = st.selectbox("Nur Klasse anzeigen", ["Alle"] + sorted(df["expected"].unique()))
if selected_class != "Alle":
    filtered = df[df["expected"] == selected_class]
else:
    filtered = df
st.dataframe(filtered)
st.download_button("Download als CSV", filtered.to_csv(index=False), "validation_filtered.csv")
