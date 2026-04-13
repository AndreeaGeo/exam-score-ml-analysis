import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data():
    # 0. ÎNCĂRCARE DATE BRUTE (RAW) - Le păstrăm neatinse pentru EDA
    df_raw = pd.read_csv('StudentPerformanceFactors.csv')
    
    # Facem o copie pe care vom lucra (df devine varianta Clean)
    df = df_raw.copy()

    # 0. CURĂȚARE TEXT (FOARTE IMPORTANT) ---
    # Eliminăm spațiile goale de la începutul sau sfârșitul textelor (ex: 'High School ' -> 'High School')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    # 1. GESTIONARE VALORI LIPSĂ (Varianta B - Imputare)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Pentru numere, folosim mediana
            df[col] = df[col].fillna(df[col].median())
        else:
            # Pentru text, folosim cea mai frecventă valoare
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # 2. MAPARE MANUALĂ (Variabile Ordinale)
    mapping_dict = {
        'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
        'Access_to_Resources': {'Low': 0, 'Medium': 1, 'High': 2},
        'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
        'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2},
        'Teacher_Quality': {'Low': 0, 'Medium': 1, 'High': 2},
        'School_Type': {'Public': 0, 'Private': 1},
        'Peer_Influence': {'Negative': 0, 'Neutral': 1, 'Positive': 2},
        'Distance_from_Home': {'Near': 0, 'Moderate': 1, 'Far': 2},
        'Parental_Education_Level': {
            'High School': 0, 
            'College': 1, 
            'Postgraduate': 2
        }
    }
    
    for col, mapping in mapping_dict.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df_pre_outliers = df.copy() # Varianta înainte de eliminarea outlierilor, pentru comparație în EDA

    # Alegem doar coloanele numerice continue unde am văzut outlieri în EDA si eliminam outlierii
    cols_cu_outlieri = ['Hours_Studied', 'Attendance', 'Exam_Score', 'Previous_Scores', 'Sleep_Hours']
    df =remove_outliers_iqr(df, cols_cu_outlieri)
            
    # 3. ONE-HOT ENCODING (Variabile Nominale)
    # În loc de LabelEncoder care suprascrie, folosim get_dummies care creează coloane noi
    nominal_cols = ['Gender', 'Internet_Access', 
                    'Extracurricular_Activities', 'Learning_Disabilities']
    
    # Verificăm care din aceste coloane chiar există în DF înainte de a face encoding
    existing_nominal = [c for c in nominal_cols if c in df.columns]
    
    # Aceasta este linia magică ce creează coloanele noi si elimina una
    #  (Gender_Male, etc.)
    df = pd.get_dummies(df, columns=existing_nominal, drop_first=True, dtype=int)
    
        
    # 4. CREARE TARGET PENTRU CLASIFICARE (Persoana B)
    # În loc de pragul fix de 50, folosim mediana pentru a asigura variabilitatea
    mediana_scor = df['Exam_Score'].median()
    
    # Creăm coloana 'Performance': 
    # 1 dacă este peste mediană (performanță ridicată)
    # 0 dacă este sub sau egal cu mediana (performanță scăzută)
    df['Performance'] = (df['Exam_Score'] > mediana_scor).astype(int)
    
    # Eliminăm vechea coloană 'Status' dacă mai exista, pentru a nu aglomera datele
    if 'Status' in df.columns:
        df = df.drop(columns=['Status'])

# Aplicăm logaritmarea pe Tutoring_Sessions pentru a reduce asimetria pozitivă
    if 'Tutoring_Sessions' in df.columns:
        # log1p aplică log(1+x), deci transformă 0 în 0, 1 în 0.69, 8 în 2.19 etc.
        df['Tutoring_Sessions'] = np.log1p(df['Tutoring_Sessions'])
    
    # 5. SCALARE DATE (Pentru K-Means și Regresie)
    df_clean = df.copy()

    # Siguranță: Scaler-ul acceptă DOAR cifre. Eliminăm orice coloană care a rămas text din greșeală.
    df_only_numeric = df_clean.select_dtypes(exclude=['object'])
    
    scaler = StandardScaler()
    # Scalăm toate coloanele, deoarece acum toate sunt numerice
    df_scaled = pd.DataFrame(scaler.fit_transform(df_only_numeric), columns=df_only_numeric.columns)

    # 6. SALVARE ÎN FIȘIERE CSV (Opțional, pentru control)
    df_clean.to_csv('StudentPerformance_Clean.csv', index=False)
    df_scaled.to_csv('StudentPerformance_Scaled.csv', index=False)

    # Returnăm toate cele 3 versiuni
    return df_raw,df_pre_outliers,df, df_scaled


def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        limita_inf = Q1 - 1.5 * IQR
        limita_sup = Q3 + 1.5 * IQR
        
        # Păstrăm doar datele din interiorul limitelor
        df = df[(df[col] >= limita_inf) & (df[col] <= limita_sup)]
    return df