import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from scipy import stats  # Aceasta este biblioteca pentru teste statistice

# 1. Analiza Valorilor Lipsă (pe datele RAW)
def show_missing_analysis(df_raw):
    st.subheader("1. Analiza Valorilor Lipsă (Date Brute)")
    missing_vals = df_raw.isnull().sum()
    missing_percent = (missing_vals / len(df_raw)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Values': missing_vals,
        'Percentage': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False)

    if not missing_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        missing_df['Percentage'].plot(kind='barh', color='orange', ax=ax)
        ax.set_title('Procentul valorilor lipsă per coloană')
        ax.set_xlabel('Procent (%)')
        st.pyplot(fig)

        
        st.markdown("""
        > **Notă Metodologică:** Deoarece lipsesc **sub 2%** din date pe fiecare coloană prezentă iar restul coloanelor nu au valori lipsă, s-a optat pentru 
        > **imputare (completare)** în loc de ștergere. Ar fi fost o pierdere de informație să eliminăm 
        > rânduri întregi pentru goluri atât de mici. Prin folosirea **medianei și a modului**, 
        > acoperim aceste celule lipsă fără a distorsiona realitatea statistică a setului de date.
        """)
    else:
        st.success(" Setul de date nu conține valori lipsă.")

# 2. Distribuții (date numerice) și Frecvențe (date categorice)
def show_distributions_and_frequencies(df_raw, df_clean):
    st.subheader("2. Analiza Distribuțiilor și Frecvențelor")
    
    # --- SECȚIUNEA A: VARIABILE NUMERICE (Histograme) ---
    st.write("###  Distribuții - Variabile Numerice")
    numeric_vars = [
        'Exam_Score', 'Hours_Studied', 'Attendance', 
        'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity'
    ]
    
    cols_to_plot_num = [c for c in numeric_vars if c in df_clean.columns]
    n_cols = 3
    n_rows_num = math.ceil(len(cols_to_plot_num) / n_cols)
    
    fig1, axes1 = plt.subplots(n_rows_num, n_cols, figsize=(18, 4 * n_rows_num))
    axes1 = axes1.flatten()
    
    for i, col in enumerate(cols_to_plot_num):
        sns.histplot(df_clean[col], kde=True, ax=axes1[i], color='skyblue', edgecolor='black')
        axes1[i].set_title(f"Distribuție: {col}", fontsize=14)
        axes1[i].set_xlabel("")
        axes1[i].set_ylabel("Frecvență")

    for j in range(i + 1, len(axes1)):
        axes1[j].axis('off')
    plt.tight_layout()
    st.pyplot(fig1)

    st.divider() # O linie separatoare între cele două tipuri de date
    st.info("""
    **Transformare: Tutoring Sessions**
    * **Problema:** Datele brute prezentau o asimetrie ridicată (mulți studenți cu 0 meditații, foarte puțini cu 8).
    * **Soluția:** Am aplicat logaritmarea, conform recomandărilor de la seminar.
    * **Scop:** Prevenirea distorsionării modelelor de Clustering și Regresie de către valorile extreme.
    """)
    st.divider()

    # --- SECȚIUNEA B: VARIABILE CATEGORICE (Bar Charts) ---
    st.write("### Frecvențe - Variabile Categorice")
    
    # Folosim df_raw pentru categorice ca să vedem etichetele (Low, High) nu cifrele (0, 1)
    categorical_vars = [
        'Gender', 'Parental_Involvement', 'Motivation_Level', 
        'Internet_Access', 'Family_Income', 'Teacher_Quality', 
        'Peer_Influence', 'School_Type', 'Extracurricular_Activities'
    ]
    
    cols_to_plot_cat = [c for c in categorical_vars if c in df_raw.columns]
    n_rows_cat = math.ceil(len(cols_to_plot_cat) / n_cols)
    
    fig2, axes2 = plt.subplots(n_rows_cat, n_cols, figsize=(18, 4 * n_rows_cat))
    axes2 = axes2.flatten()
    
    for i, col in enumerate(cols_to_plot_cat):
        # Folosim countplot pentru frecvențe
        sns.countplot(data=df_raw, x=col, ax=axes2[i], palette='viridis', hue=col, legend=False)
        axes2[i].set_title(f"Frecvență: {col}", fontsize=14)
        axes2[i].set_xlabel("")
        axes2[i].set_ylabel("Număr Studenți")
        # Rotim etichetele dacă sunt prea lungi
        axes2[i].tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes2)):
        axes2[j].axis('off')
        
    plt.tight_layout()
    st.pyplot(fig2)
    
    st.info(" Histogramele ne arată cum se dispersează valorile măsurabile, în timp ce graficele de tip BarChart ne arată proporția categoriilor în eșantionul nostru.")


def show_outliers_analysis(df_pre, df_post):
    st.subheader("3. Analiza Detaliată a Outlierilor (Metoda IQR)")

    # Selector coloană
    potential_cols = ['Hours_Studied', 'Attendance', 'Exam_Score', 'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity']
    col = st.selectbox("Selectează variabila pentru inspecție:", potential_cols)

    # Funcție internă pentru calcul stats
    def get_stats(data, column):
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        mediana = data[column].median()
        return q1, q3, iqr, mediana

    c1, c2 = st.columns(2)

    # --- COLOANA STÂNGA: ÎNAINTE ---
    with c1:
        st.write("###  Înainte de eliminare")
        q1_a, q3_a, iqr_a, med_a = get_stats(df_pre, col)
        
        fig_a, ax_a = plt.subplots()
        sns.boxplot(x=df_pre[col], color='#ff9999', ax=ax_a)
        st.pyplot(fig_a)
        
        st.metric("Mediana", f"{med_a:.2f}")
        st.metric("Interval IQR", f"{iqr_a:.2f}")
        
        outliers_count = len(df_pre[(df_pre[col] < (q1_a - 1.5*iqr_a)) | (df_pre[col] > (q3_a + 1.5*iqr_a))])
        st.write(f" Outlieri detectați matematic: **{outliers_count}**")

    # --- COLOANA DREAPTĂ: DUPĂ ---
    with c2:
        st.write("###  După procesare")
        q1_b, q3_b, iqr_b, med_b = get_stats(df_post, col)
        
        fig_b, ax_b = plt.subplots()
        sns.boxplot(x=df_post[col], color='#99ff99', ax=ax_b)
        st.pyplot(fig_b)
        
        st.metric("Mediana Nouă", f"{med_b:.2f}")
        st.metric("IQR Nou", f"{iqr_b:.2f}")
        
        # Calculăm câți outlieri mai apar în noul grafic (paradoxul de care vorbeam)
        new_outliers = len(df_post[(df_post[col] < (q1_b - 1.5*iqr_b)) | (df_post[col] > (q3_b + 1.5*iqr_b))])
        st.write(f" Outlieri pe noul grafic: **{new_outliers}**")
    st.divider()
    

    # --- EXPLICAȚIE DINAMICĂ ---
    cols_targeted = ['Hours_Studied', 'Attendance', 'Exam_Score', 'Previous_Scores', 'Sleep_Hours',"Physical_Activity"]
    
    if col == 'Tutoring_Sessions':
        st.info(f"""
        **Strategia pentru {col}:**
        Pentru această variabilă, am decis să **NU eliminăm** outlierii prin metoda IQR. 
        Motivul? Logaritmarea (vizibilă în valorile medianei de {med_b:.2f}) a fost suficientă pentru a aduce valorile extreme mai aproape de centru fără a pierde datele studenților care nu fac meditații.
        """)
    elif col in cols_targeted:
        if outliers_count > 0:
          st.success(f"""
            **Strategia pentru {col}:**
            * **Acțiune:** Am eliminat cele **{outliers_count}** valori care se abăteau prea mult de la restul grupului.
            * **Rezultat:** Deși intervalul IQR a rămas la valoarea de {iqr_b:.2f}, am reușit să eliminăm punctele extreme de la marginile graficului.
            * **Scop:** Am făcut această curățare pentru ca algoritmul de clustering să poată identifica grupurile principale de studenți fără a fi „păcălit” de câteva cazuri izolate sau atipice.
            """)
        else:
            st.write(f"**Notă:** Deși {col} a fost pe lista de monitorizare, nu au fost detectați outlieri care să necesite eliminare la acest pas.")
    else:
        st.write(f"Variabila {col} a fost păstrată în forma sa originală post-imputare.")

    procent_pastrat = (len(df_post) / len(df_pre)) * 100
    st.write(f" În urma curățării tuturor variabilelor, am păstrat **{len(df_post)}** studenți dintr-un total de **{len(df_pre)}** ({procent_pastrat:.1f}% din date).")
    st.info(f"""
    ### Alegerea metodei IQR pentru detecția outlierilor:
    
    Am ales **Interquartile Range (IQR)** ca metodă de detecție a outlierilor pentru a asigura o curățare adaptată profilului mixt al datelor noastre:
    
    1. **Versatilitate (Distribuții Mixte):** Setul nostru de date conține atât variabile cu distribuție normală (*Exam Score*), cât și variabile cu distribuție uniformă (*Attendance*). Spre deosebire de **Z-Score** (care presupune că datele sunt neapărat sub formă de clopot), IQR este o metodă neparametrică. Aceasta funcționează corect indiferent de forma distribuției.
    
    2. **Stabilitate Matematică:**
       Calculul se bazează pe **quartile** și **mediană**, indicatori care nu sunt distorsionați de valorile extreme (spre deosebire de medie, care este „trasă” de outlieri).
       * $IQR = Q_3 - Q_1$
       * Limite: Am adăugat o marjă de rezervă de **1.5 ori** dimensiunea IQR, atât în stânga, cât și în dreapta.
    
    3. **Impactul asupra Algoritmului K-Means:**
       Am eliminat aceste puncte extreme deoarece **K-Means** este extrem de sensibil la distanțe. Un singur outlier poate deplasa centrul unui întreg cluster (centroidul), alterând profilul grupurilor de studenți pe care încercăm să îi identificăm.
    """)
    



# 4. Matricea de Corelație
def show_correlation_analysis(df_clean):
    st.subheader("4. Matricea de Corelație (Heatmap)")
    # Corelația are sens doar pe valorile numerice (df_clean e deja procesat)
    corr_matrix = df_clean.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.divider()

    # --- NOTĂ EXPLICATIVĂ ---
    st.info("""
    ### Interpretarea Matricei de Corelație:
    
    Un **Heatmap** este o reprezentare vizuală a modului în care variabilele din setul nostru de date se influențează reciproc. Numerele (coeficienții) variază între **-1 și 1**:
    * **Valori pozitive (Roșu):** Variabilele cresc împreună.
    * **Valori negative (Albastru):** Când o variabilă crește, cealaltă tinde să scadă.
    * **Valori apropiate de 0 (Alb):** Nu există o legătură directă între variabile.

    ###  Observații pe setul de date:
    
    1. **Principalii factori de succes:** Cele mai puternice legături cu nota finală (**Exam_Score**) sunt **Attendance (0.58)** și **Hours_Studied (0.45)**. Aceasta confirmă faptul că prezența și timpul de studiu sunt cei mai buni predictori ai performanței.
    
    2. **Impact redus:** Factori precum **Sleep_Hours (-0.02)** și **Physical_Activity (0.01)** au corelații aproape de zero. Acest lucru sugerează că, în acest set de date, stilul de viață (somnul și sportul) nu influențează direct nota la examen la fel de mult ca efortul academic.
    
    3. **Independența variabilelor:** Majoritatea factorilor independenți (cum ar fi orele de studiu vs. orele de somn) au corelații foarte mici între ei. Acest lucru este benefic pentru modelele de regresie, deoarece variabilele nu se "suprapun" informațional.
    """)



# --- FUNCȚIA APELATĂ DIN APP.PY ---
def show_eda(df_raw, df_clean):
    st.title(" Exploratory Data Analysis (EDA)")
    
    main_tabs = st.tabs(["Valori Lipsă", "Distribuții & Frecvențe", "Outlieri", "Corelații"])
    
    with main_tabs[0]:
        show_missing_analysis(df_raw)
    with main_tabs[1]:
        show_distributions_and_frequencies(df_raw, df_clean)
    with main_tabs[2]:
        show_outliers_analysis(df_raw,df_clean)
    with main_tabs[3]:
        show_correlation_analysis(df_clean)
    