# Marketplace Safety – prioritera misstänkta händelser (ML)

Det här projektet bygger en modell som ger en **riskpoäng** för händelser i en marketplace-app (annonser/meddelanden).  
Målet är att hjälpa Trust & Safety att **prioritera vad som ska granskas först** – inte att få 100% rätt.

## Kravkort (vår grupp)
**Stakeholder:** Ali (Operations Manager)  
Ali vill inte drunkna i flaggningar. Det vi flaggar ska vara **värt att granska** och mängden ska vara **hanterbar** varje dag.

Därför använder vi **Top-X**: vi sorterar efter riskpoäng och skickar bara de X% med högst risk till granskning.

---

## Data
Ligger i `data/`:
- `historical_data.csv` – historisk data med label `is_suspicious`
- `new_data.csv` – ny data utan label (släpps av läraren, läggs i `data/` när den kommer)

---

## Projektstruktur
- `notebooks/1_EDA.ipynb`  
  Dataöversikt, target-fördelning, missing values och figurer.
- `notebooks/2_Model_Comparison.ipynb`  
  Train/test split + pipeline (leakage-säker) + cross-validation och modelljämförelse.
- `notebooks/3_Tuning_Threshhold_Deploy .ipynb`  
  Tuning med GridSearchCV, Top-X-jämförelse och sparar slutlig pipeline.
- `notebooks/4_Deploy_New_Data.ipynb`  
  Kör den sparade pipelinen på `new_data.csv` och skapar prioriteringslista.
- `src/utils.py`  
  Hjälpfunktioner (load, pipeline, top-x, spara/ladda modell).
- `models/final_model.joblib`  
  Sparad pipeline (preprocessing + modell).
- `marketplace_safety_slides.pptx`  
  Presentationen.

---

## Resultat
- Final modell: **Logistic Regression**
- Vi sparar hela pipelinen (preprocessing + modell) i `models/final_model.joblib`
- I drift använder vi **Top-X** för stabil och planeringsbar arbetsmängd

---

## Hur man kör

### 1) Installera dependencies
```bash
pip install -r requirements.txt
```

### 2) Kör notebooks i ordning
1. `notebooks/1_EDA.ipynb`
2. `notebooks/2_Model_Comparison.ipynb`
3. `notebooks/3_Tuning_Threshhold_Deploy .ipynb`

### 3) När `new_data.csv` kommer
- Lägg filen i `data/`
- Kör `notebooks/4_Deploy_New_Data.ipynb`
- Du får en lista med `id` + `risk_score` och kan flagga Top-X (t.ex. Top 5%)

---

## Ansvarsfördelning
- **Rasmus Svensson** – Pipeline, train/test split och utils-implementation  
- **Jan Rajt** – EDA och delar av modelljämförelsen  
- **Salam Lababneh** – Hyperparameter tuning och modelloptimering  
- **Yunus Capar** – Prioritering (Top-X), presentation, pitch och riskanalys  

---

## Teknisk info
- Python: **3.13.7**

---
