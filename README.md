# Bundeskanzler KI

Ein KI-gestütztes System zur Verarbeitung und Beantwortung von Fragen zu deutschen Regierungspositionen und politischen Themen.

## 🎯 Projektübersicht

Die Bundeskanzler KI ist ein intelligentes Modell, das auf Basis von Deep Learning politische Fragen und Themen verarbeitet. Das System nutzt eine optimierte LSTM-Architektur mit fortgeschrittenen Regularisierungstechniken, um präzise und kontextrelevante Antworten zu generieren.

## 🚀 Features

- **Intelligente Textverarbeitung**: Verarbeitung natürlicher Sprache für politische Themen
- **Kontextbewusstsein**: Berücksichtigung des Gesprächskontexts bei Antworten
- **Regularisierte Architektur**: Optimierte Modellarchitektur gegen Overfitting
- **Konfigurierbare Pipeline**: Flexible Anpassung von Trainingsparametern
- **Deutschsprachige Basis**: Speziell für den deutschen politischen Kontext optimiert

## 🛠 Installation

1. **Python-Umgebung erstellen**:
   ```bash
   python -m venv bkki_venv
   source bkki_venv/bin/activate  # Linux/Mac
   # oder
   .\bkki_venv\Scripts\activate   # Windows
   ```

2. **Abhängigkeiten installieren**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Verwendung

### Training

```bash
python bundeskanzler_ki.py train --epochs 200 --batch_size 4
```

### Interaktiver Modus

```bash
python bundeskanzler_ki.py interact
```

### Batch-Verarbeitung

```bash
python bundeskanzler_ki.py batch --input fragen.txt --output antworten.csv
```

## 🔧 Konfiguration

Die Konfiguration erfolgt über mehrere Dateien:

- `tf_config.py`: TensorFlow-spezifische Einstellungen
- `config.yaml`: Allgemeine Projekteinstellungen
- `transformer_model.py`: Modellarchitektur und Trainingsparameter

### Wichtige Konfigurationsparameter

```yaml
model:
  embedding_dim: 16
  lstm_units: 8
  dropout_rate: 0.6
  learning_rate: 0.00001

training:
  batch_size: 4
  epochs: 200
  validation_split: 0.2
```

## 📊 Modellarchitektur

Das System verwendet eine optimierte LSTM-basierte Architektur:

1. **Embedding Layer**: 
   - Dimensionalität: 16
   - L1/L2 Regularisierung
   - MaxNorm Constraints

2. **LSTM Layer**:
   - 8 Units
   - Bidirektional
   - Dropout: 0.6
   - Kernel, Recurrent und Bias Regularisierung

3. **Dense Layer**:
   - ReLU Aktivierung
   - Batch Normalization
   - L1/L2 Regularisierung

## 🔄 Training

Das Training ist optimiert für kleine Datensätze und verwendet:

- Early Stopping mit erhöhter Geduld
- Learning Rate Reduction
- Gradient Clipping
- L1/L2 Regularisierung
- Hohe Dropout-Raten

## 📋 Geplante Erweiterungen

- [ ] Erweiterter Trainingskorpus
- [ ] Zero-shot/Few-shot Learning
- [ ] Mehrsprachige Unterstützung
- [ ] Faktenprüfung und Quellenangaben
- [ ] Verbessertes Kontextverständnis

## 🤝 Beitragen

Beiträge sind willkommen! Bitte beachten Sie:

1. Fork des Repositories
2. Feature-Branch erstellen
3. Änderungen committen
4. Push zum Branch
5. Pull Request erstellen

## 📝 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert.

## 🙏 Danksagung

Besonderer Dank gilt allen Mitwirkenden und der Open-Source-Community für ihre wertvollen Beiträge und Unterstützung.

## 📬 Kontakt

Bei Fragen oder Anregungen können Sie ein Issue erstellen oder sich direkt an die Projektbetreuer wenden.

---

**Hinweis**: Dieses Projekt befindet sich in aktiver Entwicklung. Feedback und Verbesserungsvorschläge sind jederzeit willkommen!
