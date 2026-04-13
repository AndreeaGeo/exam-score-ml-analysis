import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def show_clustering(df_scaled, df_clean):
    st.title(" Segmentarea Studenților (K-Means Clustering)")
    
    st.markdown("""
    Clustering-ul ne ajută să grupăm studenții în funcție de caracteristicile lor comune, fără a folosi etichete predefinite. 
    Folosim datele **scalate** pentru ca toate variabilele (ore de studiu, note, prezență) să aibă aceeași pondere în calculul distanțelor.
    """)

    # --- 1. METODA ELBOW ---
    st.subheader("1. Determinarea numărului optim de grupuri (Metoda Elbow)")
    
    wcss = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)
    
    fig_elbow, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_range, wcss, marker='o', linestyle='--', color='b')
    ax.set_title('Metoda Elbow (Cotul)')
    ax.set_xlabel('Număr de Clustere (k)')
    ax.set_ylabel('WCSS (Suma pătratelor distanțelor)')
    # Marcăm "cotul" sugerat (de obicei k=3 sau 4)
    st.pyplot(fig_elbow)
    
    st.info("""
    **Interpretare:** Căutăm punctul unde panta graficului devine brusc mai lină (arată ca un „cot”). 
    Acesta este numărul optim de grupuri unde adăugarea unui nou cluster nu mai aduce o îmbunătățire semnificativă.
    """)

    # --- 2. RULARE K-MEANS ---
    st.divider()
    st.subheader("2. Aplicarea Algoritmului K-Means")
    
    k_select = st.slider("Alege numărul de clustere (k):", min_value=2, max_value=6, value=3)
    
    kmeans = KMeans(n_clusters=k_select, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    
    # Adăugăm rezultatul în DF-ul curățat pentru analiză
    df_with_clusters = df_clean.copy()
    df_with_clusters['Cluster'] = clusters

    # --- 3. VIZUALIZARE (PCA) ---
    st.write(f"### Vizualizarea celor {k_select} grupuri identificat")
    
    # Reducem dimensiunile la 2D pentru a putea face graficul
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters
    
    fig_pca, ax_pca = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', ax=ax_pca, s=60)
    ax_pca.set_title("Reprezentarea 2D a clusterelor (prin PCA)")
    st.pyplot(fig_pca)

    # --- 4. PROFILUL CLUSTERELOR ---
    st.divider()
    st.subheader("3. Profilul și Caracteristicile Grupurilor")
    
    # Calculăm media fiecărei variabile pe cluster
    cluster_profile = df_with_clusters.groupby('Cluster').mean()
    
    st.write("Media principalilor indicatori per grup:")
    important_cols = ['Exam_Score', 'Hours_Studied', 'Attendance', 'Previous_Scores']
    st.dataframe(cluster_profile[important_cols].style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='#ff9999'))

    st.markdown("""
    > **Notă interpretativă:** 
    > * **Culorile verzi** indică grupul cu cele mai bune rezultate.
    > * **Culorile roșii** indică grupul care are nevoie de cea mai mare atenție (risc de abandon sau note mici).
    """)