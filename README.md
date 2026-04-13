# Student Performance Analysis - Streamlit Application

## English / Engleza

### Application Logic

**The core of this project:** We analyze **what factors influence student exam scores (Exam_Score)**.

- **For Regression Tasks**: We predict the exact **Exam_Score** value (0-100) based on student characteristics (attendance, hours studied, family income, etc.)
- **For Classification Tasks**: We classify students into **two performance levels** based on whether their score is above or below the median:
  - Performance = 1: Score > Median (High Performance)
  - Performance = 0: Score ≤ Median (Low Performance)

**Example:** If a student has high attendance, studies many hours, and has good family support, the model predicts they will have a high exam score.

### Project Overview

This is an interactive machine learning application built with **Streamlit** that analyzes factors influencing student performance using multiple algorithms:

- **Exploratory Data Analysis (EDA)** - Data visualization and statistical analysis
- **Clustering** - K-Means clustering to group students
- **Regression Models** - Linear Regression, Random Forest Regressor, XGBoost Regressor for predicting exam scores
- **Classification Models** - Logistic Regression, Random Forest Classification for predicting performance level

The application includes model comparison, feature importance analysis, individual predictions, and data export functionality.

### Technologies Used

- **Python 3.8+**
- **Streamlit** - Web interface
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting
- **Matplotlib & Seaborn** - Data visualization
- **Statsmodels** - Statistical analysis

### Setup Instructions

#### 1. Clone or Download the Project
```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd PROIECT-psw/Proiect
```

#### 2. Create Virtual Environment (if not already exists)
```bash
python -m venv venv
```

#### 3. Activate Virtual Environment
- **Windows:**
```bash
.\venv\Scripts\activate
```
- **Mac/Linux:**
```bash
source venv/bin/activate
```

#### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 5. Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Project Structure

```
PROIECT-psw/Proiect/
├── app.py                              # Main application entry point
├── preprocessing.py                    # Data loading and preprocessing
├── eda.py                             # Exploratory Data Analysis
├── clustering.py                       # K-Means clustering
├── regression.py                       # Linear Regression
├── random_forest.py                    # Random Forest Regressor
├── xgboost_regressor.py               # XGBoost Regressor
├── classification.py                   # Random Forest Classification
├── logistic_regression_classification.py  # Logistic Regression
├── StudentPerformanceFactors.csv      # Raw data
├── StudentPerformance_Clean.csv       # Cleaned data
├── StudentPerformance_Scaled.csv      # Scaled data
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
└── venv/                              # Virtual environment (not uploaded to GitHub)
```

### Features

✓ Data preview in multiple formats (raw, clean, scaled)  
✓ Download data as CSV or Excel  
✓ Model performance comparison  
✓ Feature importance visualization  
✓ Interactive predictions for single students  
✓ Confusion matrices and ROC curves  
✓ Model coefficient interpretation  

### Notes

- The application automatically handles data preprocessing
- Train/Test split is 80/20 with stratification for classification tasks
- Default random_state=42 for reproducible results

---

## Romana

### Logica Aplicatiei

**Nucleul acestui proiect:** Analizam **ce factori influentiaza scorurile la examen ale studentilor (Exam_Score)**.

- **Pentru Sarcinile de Regresie**: Prezicemi valoarea exacta a **Exam_Score** (0-100) bazata pe caracteristicile studentului (prezenta, ore de studiu, venit familial, etc.)
- **Pentru Sarcinile de Clasificare**: Clasificam studentii in **doua nivele de performanta** bazate pe daca scorul este deasupra sau sub mediana:
  - Performance = 1: Score > Mediana (Performanta Inalta)
  - Performance = 0: Score ≤ Mediana (Performanta Joasa)

**Exemplu:** Daca un student are prezenta inalta, studiaza multe ore si are suport familial bun, modelul prezice ca va avea un scor inalt la examen.

### Descrierea Proiectului

Aceasta este o aplicatie interactiva de machine learning construita cu **Streamlit** care analizeaza factorii care influentiaza performanta studentilor folosind mai multi algoritmi:

- **Exploratory Data Analysis (EDA)** - Vizualizare si analiza statistica a datelor
- **Clustering** - Grupare studentilor cu K-Means
- **Modele de Regresie** - Regresie Liniara, Random Forest Regressor, XGBoost Regressor pentru predictia scorurilor
- **Modele de Clasificare** - Regresie Logistica, Random Forest Classification pentru predictia nivelului de performanta

Aplicatia include comparatie intre modele, analiza importantei variabilelor, predictii individuale si export de date.

### Tehnologii Utilizate

- **Python 3.8+**
- **Streamlit** - Interfata web
- **Pandas** - Manipulare date
- **Scikit-learn** - Algoritmi machine learning
- **XGBoost** - Gradient boosting
- **Matplotlib & Seaborn** - Vizualizare date
- **Statsmodels** - Analiza statistica

### Instructiuni de Configurare

#### 1. Clone sau Download Proiect
```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd PROIECT-psw/Proiect
```

#### 2. Creeaza Mediu Virtual (daca nu exista deja)
```bash
python -m venv venv
```

#### 3. Activeaza Mediul Virtual
- **Windows:**
```bash
.\venv\Scripts\activate
```
- **Mac/Linux:**
```bash
source venv/bin/activate
```

#### 4. Instaleaza Dependentele
```bash
pip install -r requirements.txt
```

#### 5. Ruleaza Aplicatia
```bash
streamlit run app.py
```

Aplicatia se va deschide in browserul tau la `http://localhost:8501`

### Structura Proiectului

```
PROIECT-psw/Proiect/
├── app.py                              # Punctul de intrare principal
├── preprocessing.py                    # Incarcare si preprocesare date
├── eda.py                             # Analiza Exploratorie a Datelor
├── clustering.py                       # Clustering K-Means
├── regression.py                       # Regresie Liniara
├── random_forest.py                    # Random Forest Regressor
├── xgboost_regressor.py               # XGBoost Regressor
├── classification.py                   # Random Forest Classification
├── logistic_regression_classification.py  # Regresie Logistica
├── StudentPerformanceFactors.csv      # Date brute
├── StudentPerformance_Clean.csv       # Date curatate
├── StudentPerformance_Scaled.csv      # Date standardizate
├── requirements.txt                    # Dependente Python
├── .gitignore                         # Reguli git ignore
└── venv/                              # Mediu virtual (nu se incarca pe GitHub)
```

### Functionalitati

✓ Previzualizare date in mai multe formate (brute, curatate, scalate)  
✓ Download date ca CSV sau Excel  
✓ Comparatie performanta intre modele  
✓ Vizualizare importanta variabilelor  
✓ Predictii interactive pentru studenti individuali  
✓ Matrice de confuzie si curbe ROC  
✓ Interpretare coeficientii modelelor  

### Note Importante

- Aplicatia efectueaza preprocesarea datelor automat
- Impartirea Train/Test este 80/20 cu stratificare pentru sarcinile de clasificare
- Default random_state=42 pentru rezultate reproducibile

---

**Author / Autor:** Andreea  
**Date / Data:** April 2026 / Aprilie 2026
