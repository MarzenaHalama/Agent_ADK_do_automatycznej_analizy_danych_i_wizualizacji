
# Agent ADK do automatycznej analizy danych i wizualizacji

Repozytorium zbudowane w oparciu o **Google Agent Development Kit (ADK)** — framework do tworzenia agentów LLM zdolnych do autonomicznego wykonywania zadań analitycznych.

Agent integruje pełen pipeline data science: automatyczne czyszczenie danych, trenowanie modeli ML (regresja / klasyfikacja), wykrywanie anomalii oraz wizualizację wyników w spójnym, estetycznym stylu. Każdy etap jest dostępny jako narzędzie (`FunctionTool`) wywoływane przez agenta na podstawie naturalnego języka.

Kod analityczny jest zorganizowany modułowo w pakiecie `src/` i pokryty testami jednostkowymi (`pytest`).

---

## Author
Marzena Halama

---

## Struktura repozytorium

```
analiza_danych_i_wizualizacja/
│
├── data/
│   ├── data_classification_iris.csv   # dane klasyfikacji (Iris)
│   ├── data_regression.csv            # dane regresji
│   └── data_timeseries_water.csv      # dane szeregów czasowych
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # globalna konfiguracja i styl wykresów
│   ├── data_loader.py                 # wczytywanie i walidacja danych
│   ├── analysis.py                    # regresja, klasyfikacja (ML)
│   ├── anomaly_detection.py           # wykrywanie anomalii (IsolationForest, Z-score)
│   ├── visualization.py               # spójne wizualizacje matplotlib
│   └── pipeline.py                    # orkiestracja pełnego pipeline'u
│
├── tests/
│   ├── conftest.py                    # wspólne fixtures pytest
│   ├── test_data_loader.py
│   ├── test_analysis.py
│   ├── test_anomaly_detection.py
│   └── test_visualization.py
│
├── generated_plots/                   # wykresy generowane automatycznie
├── requirements.txt
└── README.md
```

## Utwórz i aktywuj środowisko wirtualne:

python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows


## Zainstaluj wymagane biblioteki:

pip install -r requirements.txt

Zawartość pliku requirements.txt:

pandas
scikit-learn
matplotlib
scipy
google-adk
google-genai
python-dotenv

## Użycie

### 1. Analiza danych

Moduł analysis.py umożliwia:

* automatyczne czyszczenie i konwersję danych,

* regresję liniową i sieć neuronową (MLP),

* klasyfikację,

* wykrywanie anomalii (IsolationForest + Z-score).

### 2. Wizualizacje

Moduł viz.py generuje wykresy w spójnym stylu (#555555, jasny tekst, turkusowo–zielona paleta).


Obsługiwane typy wykresów:

* histogram
* boxplot
* scatter
* reg_line
* confusion_matrix
* pred_vs_true
* pred_sequence
* residuals
* anomaly_scatter

### 3. Agenci LLM

Repozytorium zawiera pliki:

* demo_agent_chat

* LLM_agents_vs - koneicznie skonfiguruj go samodzielnie! 

Pokazują one integrację analizy danych z agentami LLM (np. Google GenAI) — w celu wspomagania interpretacji wyników i automatyzacji procesów analitycznych.

**Kluczowy wniosek projektu:**

Instrukcja (prompt) jest najważniejszym elementem pracy agenta LLM.
Od sposobu, w jaki formułujesz kontekst, polecenia i ograniczenia, zależy jakość, precyzja i użyteczność odpowiedzi.
Nawet przy tym samym modelu (np. Google GenAI lub OpenAI GPT) różnice między wynikami mogą być ogromne — wyłącznie z powodu innej instrukcji.

W projektach analitycznych dobrze zdefiniowany kontekst i rola agenta (np. “ekspert ds. wizualizacji danych” lub “model interpretujący wyniki regresji”) mają kluczowe znaczenie dla skuteczności całego systemu.


