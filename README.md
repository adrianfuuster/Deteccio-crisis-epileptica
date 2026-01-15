# Detecció automàtica de crisis epilèptiques amb EEG — CNN vs CNN+LSTM (CHB-MIT)

Projecte de l’assignatura **Processament de Senyal, Imatge i Vídeo (PSIV)** del **Grau d’Enginyeria de Dades (UAB)**.  
Implementem i avaluem models de Deep Learning per detectar **crisis epilèptiques (seizures)** a partir de senyals **EEG**, comparant una arquitectura per finestra (CNN) i una arquitectura amb context temporal (CNN+LSTM).

**Autors:** Álvaro Bello · Adrián Fuster · Marc Cases · Namanmahi Kumar  
**Universitat:** Universitat Autònoma de Barcelona (UAB)  
**Repositori:** https://github.com/adrianfuuster/Deteccio-crisis-epileptica

---

## Objectiu

Construir un pipeline de classificació binària (ictal vs normal) sobre finestres EEG i analitzar com varia el rendiment segons:

- **Arquitectura:** CNN vs CNN+LSTM  
- **Estratègia de partició:** poblacional (inter-pacient) vs personalitzada (intra-pacient)

L’èmfasi és en la **generalització** (què passa quan canvia el pacient o quan canviem l’enregistrament / la crisi).

---

## Dataset (CHB-MIT)

- **Senyals:** EEG scalp (CHB-MIT)
- **Entrada al model:** finestres de forma **[C, T] = [21 canals, 128 mostres]**
- **Etiqueta:** `0 = normal`, `1 = ictal (seizure)`
- **Dades al pipeline:** `.npz` (finestres) + `.parquet` (etiquetes i metadades com `patient_id`, `filename`, `global_interval`)

---

## Models implementats

### Sistema 1 (S1): CNN per finestra
Classificació independent de cada finestra EEG mitjançant convolucions 1D.

- Punt fort: eficiència i bon rendiment en personalització intra-pacient.
- Limitació: manca de context temporal (més sensible a finestres sorolloses).

### Sistema 2 (S2): CNN + LSTM amb context temporal
Extracció d’embeddings per finestra amb CNN i modelatge temporal amb LSTM sobre seqüències de longitud `K`.

- Punt fort: integra dependències temporals; tendeix a millorar la sensibilitat (recall positiu).
- Limitació: major cost i necessitat de particions robustes per evitar folds degenerats.

---

## Estratègies d’entrenament i validació (3)

Per cada sistema s’avaluen tres règims:

1) **Poblacional (LOPO)**  
   *Leave-One-Patient-Out* → generalització inter-pacient.

2) **Personalitzat per fitxer (`filename`)**  
   Intra-pacient: es deixa fora un enregistrament per test.

3) **Personalitzat per crisi (`global_interval`)**  
   Intra-pacient: es deixa fora un interval associat a crisi/segment.

📌 Total: **6 configuracions**
- **S1-Pop**, **S1-Fitxer**, **S1-Crisi**
- **S2-Pop**, **S2-Fitxer**, **S2-Crisi**

---

## Mètriques

Donat el desbalanceig, prioritzem:

- **Recall(+)**: sensibilitat (detecció de crisis; minimitzar FN)
- **Recall(-)**: especificitat (control de FP)
- **F1(+)**
- **Balanced Accuracy**

---

## Fitxers del repositori

- `system1_analysis.py` — Entrenament i avaluació del **Sistema 1 (CNN)**
- `system2_analysis.py` — Entrenament i avaluació del **Sistema 2 (CNN+LSTM)**
- `Presentation_EEG.pdf` — Presentació amb resultats, figures i anàlisi

---

## Execució

La forma d’execució depèn dels arguments definits als scripts. Recomanat:

```bash
python system1_analysis.py --help
python system2_analysis.py --help
````

---

## Notes metodològiques rellevants

En l’escenari **personalitzat per crisi**, és possible que alguns folds de test continguin **0 positius** (cap finestra ictal). Això pot fer que **recall(+)** sigui degenerat o no definit (`NaN`) i incrementar molt la variabilitat dels resultats. En aquests casos, convé utilitzar estratificació amb restricció de grup o redefinir la unitat de partició per garantir presència mínima de positius per fold.

---

## Resultats

Els resultats i figures es mostren a `Presentation_EEG.pdf` (gràfiques W&B, comparatives i taules agregades).

