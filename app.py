import io

import streamlit as st
import pandas as pd

from preprocessing import load_and_preprocess_data
from eda import show_eda
from clustering import show_clustering
from regression import show_regression
from random_forest import show_random_forest
from xgboost_regressor import show_xgboost_regressor
from classification import show_classification
from logistic_regression_classification import show_logistic_regression_classification

# Configurare pagină
st.set_page_config(page_title="Proiect Performanță Studenți", layout="wide")

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def show_home(df_clean, df_scaled,df_raw):
    st.title(" Pagina Principală")
    st.subheader("Analiza factorilor care influențează performanța studenților.")
    st.write("Bine ați venit! Această secțiune oferă o privire de ansamblu asupra setului de date în diferite stadii de prelucrare și permite exportul acestora.")

    # --- SECȚIUNEA 1: PREVIEW DATE ---
    tab1, tab2, tab3 = st.tabs([" Date Brute ", " Date Curățate ", 
    " Date Scalate "])
    
    with tab1:
        st.write("### Preview: Date Brute")
        st.dataframe(df_raw.head(10))
        st.caption(f"Dimensiune inițială: {df_raw.shape[0]} rânduri și {df_raw.shape[1]} coloane.")

    with tab2:
        st.write("### Preview: Date după Preprocesare (Fără Outlieri, Mapate)")
        st.dataframe(df_clean.head(10))
        st.caption(f"Dimensiune actuală: {df_clean.shape[0]} rânduri (după eliminarea outlierilor).")

    with tab3:
        st.write("### Preview: Date Standardizate (StandardScaler)")
        st.dataframe(df_scaled.head(10))
        st.caption("Toate variabilele au media 0 și deviația standard 1.")

    st.divider()

    # --- SECȚIUNEA 2: BUTOANE DESCARCARE ---
    st.subheader(" Exportă Datele")
    st.write("Alege formatul și varianta de date pe care dorești să o descarci:")

    # Organizăm butoanele în coloane pentru un aspect curat
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("**Varianta RAW**")
        st.download_button(
            label="CSV",
            data=df_raw.to_csv(index=False).encode('utf-8'),
            file_name='date_brute.csv',
            mime='text/csv'
        )
        st.download_button(
            label="Excel",
            data=to_excel(df_raw),
            file_name='date_brute.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    with col2:
        st.success("**Varianta CLEAN**")
        st.download_button(
            label="CSV",
            data=df_clean.to_csv(index=False).encode('utf-8'),
            file_name='date_curatate.csv',
            mime='text/csv'
        )
        st.download_button(
            label="Excel",
            data=to_excel(df_clean),
            file_name='date_curatate.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    with col3:
        st.warning("**Varianta SCALED**")
        st.download_button(
            label="CSV",
            data=df_scaled.to_csv(index=False).encode('utf-8'),
            file_name='date_scalate.csv',
            mime='text/csv'
        )
        st.download_button(
            label="Excel",
            data=to_excel(df_scaled),
            file_name='date_scalate.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )


def main():
    # Încărcăm ambele variante ale datelor
    df_raw,df_pre_outliers,df_clean, df_scaled = load_and_preprocess_data()
    
    st.sidebar.title("Meniu Proiect")
    pagini = {
        "Pagina Principală": lambda: show_home(df_clean,df_scaled,df_raw),
        "Explorare Date (EDA)": lambda: show_eda(df_raw, df_clean),
        "Clustering (K-Means)": lambda: show_clustering(df_scaled,df_clean),
        "Regresie liniară": lambda: show_regression(df_clean),
        "Random Forest Regressor": lambda: show_random_forest(df_clean),
        "XGBoost Regressor": lambda: show_xgboost_regressor(df_clean),
        "Regresie logistică": lambda: show_logistic_regression_classification(df_clean),
        "Random Forest Classification": lambda: show_classification(df_clean),
    }
    
    selectie = st.sidebar.radio("Navigați la:", list(pagini.keys()))
    pagini[selectie]()

if __name__ == "__main__":
    main()

    
