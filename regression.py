import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


MODEL_FEATURES = [
    "Attendance",
    "Hours_Studied",
    "Access_to_Resources",
    "Parental_Involvement",
    "Tutoring_Sessions",
    "Parental_Education_Level",
    "Peer_Influence",
]

INITIAL_CANDIDATES = [
    "Attendance",
    "Hours_Studied",
    "Previous_Scores",
    "Access_to_Resources",
    "Parental_Involvement",
    "Tutoring_Sessions",
    "Parental_Education_Level",
    "Peer_Influence",
]

VIF_FEATURES = [
    "Attendance",
    "Hours_Studied",
    "Previous_Scores",
    "Access_to_Resources",
    "Parental_Involvement",
    "Tutoring_Sessions",
    "Parental_Education_Level",
    "Peer_Influence",
]

ORDINAL_FEATURE_OPTIONS = {
    "Access_to_Resources": {
        "Redus": 0.0,
        "Mediu": 1.0,
        "Ridicat": 2.0,
    },
    "Parental_Involvement": {
        "Redus": 0.0,
        "Mediu": 1.0,
        "Ridicat": 2.0,
    },
    "Peer_Influence": {
        "Negativă": 0.0,
        "Neutră": 1.0,
        "Pozitivă": 2.0,
    },
    "Parental_Education_Level": {
        "High School": 0.0,
        "College": 1.0,
        "Postgraduate": 2.0,
    },
}


def show_regression(df_clean):
    st.title("Regresie liniară")

    st.markdown(
        """
        ### Contextul analizei

        Scopul acestei analize este dublu: pe de o parte, urmărim să identificăm factorii care influențează
        performanța academică a studenților, măsurată prin variabila țintă `Exam_Score`, iar pe de altă parte
        dorim să construim un model capabil să realizeze predicții asupra acestei variabile.

        Pentru atingerea acestui obiectiv, utilizăm regresia liniară, o metodă care permite modelarea relației
        dintre scorul la examen și variabilele explicative selectate din setul de date. Analiza este realizată
        pe setul de date obținut după etapa de preprocesare, astfel încât informațiile folosite în modelare
        să fie deja curate și pregătite pentru interpretare statistică.

        Demersul este organizat în trei etape principale. Mai întâi, stabilim variabilele candidate, apoi
        construim modelul de regresie liniară și împărțim datele în set de antrenare și set de testare.
        În final, evaluăm performanța modelului cu ajutorul indicatorilor statistici și al graficelor de diagnostic.
        """
    )

    available_features = [feature for feature in MODEL_FEATURES if feature in df_clean.columns]
    results = prepare_regression_results(df_clean, available_features)

    tab1, tab2, tab3 = st.tabs(
        [
            "1. Selectarea variabilelor explicative",
            "2. Modelul de regresie",
            "3. Evaluarea modelului",
        ]
    )

    with tab1:
        show_variable_selection(df_clean, available_features)

    with tab2:
        show_model_building(results)

    with tab3:
        show_model_evaluation(results)


def show_variable_selection(df, available_features):
    st.markdown(
        """
        În această etapă toate variabilele explicative pornesc ca variabile candidate pentru modelul de regresie.
        Ulterior, acestea sunt evaluate succesiv prin analiză vizuală, analiză a corelației și verificarea
        multicoliniarității prin VIF, pentru a decide care dintre ele rămân în modelul final.

        În urma acestor etape succesive de selecție, modelul final de regresie liniară utilizează
        următoarele variabile explicative:

        - `Attendance`
        - `Hours_Studied`
        - `Access_to_Resources`
        - `Parental_Involvement`
        - `Tutoring_Sessions`
        - `Parental_Education_Level`
        - `Peer_Influence`

        Modelul rezultat permite explicarea variației scorului la examen (`Exam_Score`) pe baza
        unor factori relevanți legați de comportamentul academic și contextul educațional al studenților.
        """
    )

    subtabs = st.tabs(["Relații cu target", "Corelație", "Multicoliniaritate"])

    with subtabs[0]:
        show_relationships(df)

    with subtabs[1]:
        show_target_correlation(df)

    with subtabs[2]:
        show_vif(df)


def show_relationships(df):
    st.subheader("Relația variabilelor cu Exam_Score")
    st.write(
        "Prin inspecția grafică observăm dacă variabilele numerice au o tendință aproximativ liniară "
        "cu `Exam_Score` și dacă variabilele categorice produc diferențe vizibile între distribuțiile scorurilor."
    )
    st.markdown(
        """
        Pe baza analizei vizuale, au fost păstrate inițial variabilele care prezintă tendințe clare
        sau diferențe vizibile între distribuțiile scorului.

        Astfel, variabilele care explică cel mai bine `Exam_Score` în această etapă inițială sunt:
        - `Attendance`
        - `Hours_Studied`
        - `Previous_Scores`
        - `Access_to_Resources`
        - `Parental_Involvement`
        - `Tutoring_Sessions`

        Variabilele care prezintă o influență mai redusă, dar totuși vizibilă, sunt:
        - `Parental_Education_Level`
        - `Peer_Influence`
        - `Family_Income`

        În schimb, variabilele care nu au fost reținute după analiza vizuală, deoarece nu prezintă relații
        vizibile cu scorul la examen, sunt:
        - `Sleep_Hours`
        - `Physical_Activity`
        - `Motivation_Level`
        - `Teacher_Quality`
        - `Gender_Male`
        - `Internet_Access_Yes`
        - `Extracurricular_Activities_Yes`
        - `School_Type`
        - `Distance_from_Home`
        - `Learning_Disabilities_Yes`
        """
    )

    st.write("### Variabile numerice")
    continuous_cols = [
        "Hours_Studied",
        "Attendance",
        "Sleep_Hours",
        "Previous_Scores",
        "Tutoring_Sessions",
        "Physical_Activity",
    ]

    available_continuous = [col for col in continuous_cols if col in df.columns]
    cont_cols_layout = st.columns(2)

    for i, col in enumerate(available_continuous):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=df[col], y=df["Exam_Score"], ax=ax, color="#1f77b4", alpha=0.7)
        ax.set_title(f"{col} vs Exam_Score")
        ax.set_xlabel(col)
        ax.set_ylabel("Exam_Score")
        cont_cols_layout[i % 2].pyplot(fig)
        plt.close(fig)

    st.divider()

    st.write("### Variabile categorice")
    categorical_cols = [
        "Parental_Involvement",
        "Access_to_Resources",
        "Motivation_Level",
        "Family_Income",
        "Teacher_Quality",
        "Peer_Influence",
        "Parental_Education_Level",
        "Distance_from_Home",
        "Gender_Male",
        "Internet_Access_Yes",
        "Extracurricular_Activities_Yes",
        "Learning_Disabilities_Yes",
    ]

    available_categorical = [col for col in categorical_cols if col in df.columns]
    cat_cols_layout = st.columns(2)

    for i, col in enumerate(available_categorical):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df[col], y=df["Exam_Score"], ax=ax, color="#8ecae6")
        ax.set_title(f"{col} vs Exam_Score")
        ax.set_xlabel(col)
        ax.set_ylabel("Exam_Score")
        cat_cols_layout[i % 2].pyplot(fig)
        plt.close(fig)


def show_target_correlation(df):
    st.subheader("Corelația cu Exam_Score")
    st.markdown(
        """
        Analiza corelației cu `Exam_Score` este realizată pe toate variabilele numerice disponibile în setul de date.
        Ea ne ajută să observăm cât de puternică este legătura liniară dintre fiecare variabilă și scorul final,
        oferind un criteriu util în selecția predictorilor pentru modelul de regresie.
        """
    )

    numeric_columns = [
        column
        for column in df.select_dtypes(include=["number"]).columns.tolist()
        if column != "Performance"
    ]
    corr = df[numeric_columns].corr(numeric_only=True)["Exam_Score"].sort_values(ascending=False)
    corr_without_target = corr.drop("Exam_Score")

    fig, ax = plt.subplots(figsize=(8, 6))
    corr_without_target.plot(kind="barh", ax=ax, color="skyblue")
    ax.set_title("Corelația variabilelor numerice cu Exam_Score")
    ax.set_xlabel("Coeficient de corelație")
    ax.set_ylabel("Variabile")
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Vezi valorile corelațiilor"):
        st.dataframe(corr.to_frame("Corelație"))

    st.info(
        "Valorile apropiate de 1 indică o legătură liniară pozitivă puternică, valorile apropiate de -1 "
        "indică o legătură liniară negativă puternică, iar valorile apropiate de 0 sugerează o asociere "
        "liniară slabă. Corelația descrie asocierea dintre variabile, nu demonstrează cauzalitate."
    )

    st.markdown(
        """
        Pe baza analizei corelației cu variabila țintă `Exam_Score`, au fost selectate inițial ca variabile candidate
        acelea care prezintă o legătură liniară relevantă (coeficient de corelație mai mare de aproximativ 0.1),
        deoarece acestea contribuie la explicarea variației scorului.

        Astfel, au fost propuse inițial spre păstrare următoarele variabile:
        - `Attendance`
        - `Hours_Studied`
        - `Previous_Scores`
        - `Access_to_Resources`
        - `Parental_Involvement`
        - `Tutoring_Sessions`
        - `Parental_Education_Level`
        - `Peer_Influence`

        În schimb, variabilele care au prezentat corelații foarte scăzute sau apropiate de zero au fost propuse
        spre eliminare din analiză, deoarece nu oferă informații relevante pentru explicarea scorului la examen.
        """
    )

    selected_feature = st.selectbox(
        "Selectează o variabilă pentru interpretare:",
        corr_without_target.index.tolist(),
    )
    selected_value = corr_without_target[selected_feature]

    if abs(selected_value) >= 0.7:
        intensity = "foarte puternică"
    elif abs(selected_value) >= 0.5:
        intensity = "puternică"
    elif abs(selected_value) >= 0.3:
        intensity = "moderată"
    elif abs(selected_value) >= 0.1:
        intensity = "slabă"
    else:
        intensity = "foarte slabă"

    if selected_value > 0:
        direction = "pozitivă"
        effect_text = "creșterea"
    elif selected_value < 0:
        direction = "negativă"
        effect_text = "scăderea"
    else:
        direction = "nulă"
        effect_text = "o variație clară a"

    st.markdown(
        f"""
        **Interpretare pentru `{selected_feature}`**

        Corelația dintre `{selected_feature}` și `Exam_Score` este **{intensity}** și **{direction}**
        (`r = {selected_value:.3f}`).

        Aceasta sugerează că variația acestei variabile este asociată cu **{effect_text}** scorului la examen,
        însă relația observată trebuie interpretată ca asociere statistică, nu ca dovadă de cauzalitate.
        """
    )


def show_vif(df):
    st.subheader("Multicoliniaritate (VIF)")
    st.write(
        "VIF este o verificare preliminară aplicată pe variabilele explicative numerice, "
        "pentru a observa dacă unele dintre ele transmit informație foarte similară."
    )
    st.markdown(
        """
        Analiza VIF este utilizată pentru a verifica prezența multicoliniarității, adică situația în care
        două sau mai multe variabile explicative sunt puternic corelate între ele. Această verificare este
        importantă deoarece, într-un model de regresie liniară, predictori foarte asemănători pot face
        coeficienții instabili și mai greu de interpretat.
        """
    )

    available_vif_features = [feature for feature in VIF_FEATURES if feature in df.columns]
    X = df[available_vif_features].copy()

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(len(X.columns))
    ]

    vif_sorted = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)
    st.caption("Tabelul include variabilele candidate rămase după analiza vizuală și analiza corelației.")
    st.dataframe(vif_sorted, use_container_width=True)

    st.warning(
        "Regula practică: valori VIF sub 5 sunt de obicei acceptabile, între 5 și 10 cer atenție, "
        "iar peste 10 pot indica o problemă serioasă de multicoliniaritate."
    )
    st.info(
        """
        Analiza multicoliniarității a evidențiat valori ridicate ale indicatorului VIF pentru variabilele
        `Attendance`, `Hours_Studied` și `Previous_Scores`, ceea ce indică existența unor relații de dependență
        între acestea. Deși corelațiile perechi dintre variabile nu sunt foarte ridicate, valorile mari ale VIF
        sugerează că ele transmit informație similară în combinație, fiind astfel redundante în model.

        Variabila `Previous_Scores` a fost eliminată din model din cauza valorilor ridicate ale VIF, care indică
        multicoliniaritate în raport cu alte variabile, în special `Attendance` și `Hours_Studied`. Eliminarea
        acesteia contribuie la reducerea redundanței și la îmbunătățirea stabilității modelului.
        """
    )


def prepare_regression_results(df, available_features):
    X = df[available_features].copy()
    y = df["Exam_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    coef_df = pd.DataFrame(
        {"Variabila": available_features, "Coeficient": model.coef_}
    ).sort_values(by="Coeficient", ascending=False)

    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)

    return {
        "features": available_features,
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "coef_df": coef_df,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def render_prediction_input(feature, values_df, key_prefix=""):
    if feature in ORDINAL_FEATURE_OPTIONS:
        options = ORDINAL_FEATURE_OPTIONS[feature]
        numeric_values = list(options.values())
        feature_mean = float(values_df[feature].mean())
        default_value = min(numeric_values, key=lambda value: abs(value - feature_mean))
        option_labels = list(options.keys())
        default_index = numeric_values.index(default_value)
        selected_label = st.selectbox(
            feature,
            option_labels,
            index=default_index,
            key=f"{key_prefix}{feature}",
        )
        return float(options[selected_label])

    feature_min = float(values_df[feature].min())
    feature_max = float(values_df[feature].max())
    feature_mean = float(values_df[feature].mean())
    return st.number_input(
        feature,
        min_value=feature_min,
        max_value=feature_max,
        value=feature_mean,
        step=0.1,
        key=f"{key_prefix}{feature}",
    )


def show_model_building(results):
    st.subheader("Construirea modelului de regresie")
    st.markdown(
        """
        În această etapă construim modelul de regresie liniară folosind variabilele candidate selectate anterior.
        Datele sunt împărțite în două subseturi:

        - `set de antrenare (train)` folosit pentru estimarea coeficienților modelului
        - `set de testare (test)` folosit pentru verificarea capacității de generalizare pe date nevăzute
        """
    )

    train_size = len(results["X_train"])
    test_size = len(results["X_test"])
    total_size = train_size + test_size

    col1, col2, col3 = st.columns(3)
    col1.metric("Observații train", train_size)
    col2.metric("Observații test", test_size)
    col3.metric("Raport split", f"{train_size / total_size:.0%} / {test_size / total_size:.0%}")

    st.info(
        "Modelul este antrenat pe setul de train și apoi aplicat pe setul de test. "
        "Dacă performanța rămâne bună și pe test, modelul este mai credibil."
    )

    with st.expander("Variabile folosite în model", expanded=False):
        st.write(", ".join(results["features"]))

    st.write("### Grafic train: valori reale vs valori prezise")
    plot_real_vs_predicted(results["y_train"], results["y_pred_train"], "Train")

    st.write("### Grafic test: valori reale vs valori prezise")
    plot_real_vs_predicted(results["y_test"], results["y_pred_test"], "Test")

    st.write("### Predicție pentru un student")
    st.markdown(
        """
        Completează valorile variabilelor explicative pentru un student, apoi apasă butonul de mai jos
        pentru a estima nota așteptată la examen (`Exam_Score`).
        """
    )

    input_values = {}
    input_columns = st.columns(2)

    for index, feature in enumerate(results["features"]):
        col = input_columns[index % 2]
        with col:
            input_values[feature] = render_prediction_input(
                feature,
                results["X_train"],
                key_prefix="reg_",
            )

    if st.button("Calculează predicția"):
        student_df = pd.DataFrame([input_values])
        predicted_score = float(results["model"].predict(student_df)[0])
        st.success(f"Nota estimată la examen pentru acest student este **{predicted_score:.2f}**.")


def show_model_evaluation(results):
    st.subheader("Evaluarea modelului")
    st.markdown(
        """
        În această etapă evaluăm performanța modelului prin indicatori statistici și grafice de diagnostic.
        Comparația dintre rezultatele obținute pe train și pe test ne ajută să observăm dacă modelul generalizează bine
        sau dacă există semne de supraantrenare.
        """
    )

    st.write("### Indicatori de performanță")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Set de antrenare**")
        st.metric("R² train", f"{results['train_metrics']['R2']:.3f}")
        st.metric("RMSE train", f"{results['train_metrics']['RMSE']:.2f}")
        st.metric("MAE train", f"{results['train_metrics']['MAE']:.2f}")

    with col2:
        st.markdown("**Set de testare**")
        st.metric("R² test", f"{results['test_metrics']['R2']:.3f}")
        st.metric("RMSE test", f"{results['test_metrics']['RMSE']:.2f}")
        st.metric("MAE test", f"{results['test_metrics']['MAE']:.2f}")

    st.info(
        "Un model bine calibrat ar trebui să aibă performanțe apropiate pe train și test. "
        "Diferențele foarte mari pot indica overfitting."
    )

    if abs(results["train_metrics"]["R2"] - results["test_metrics"]["R2"]) < 0.1:
        st.success("Modelul generalizează bine (nu există overfitting semnificativ).")
    else:
        st.warning("Diferența dintre train și test sugerează posibil overfitting.")

    st.write("### Ecuația modelului de regresie liniară")
    
    # Construim ecuația
    intercept = results["model"].intercept_
    equation_parts = [f"{intercept:.4f}"]
    
    for feature, coef in zip(results["features"], results["model"].coef_):
        sign = "+" if coef >= 0 else "-"
        equation_parts.append(f"{sign} {abs(coef):.4f} × {feature}")
    
    equation_text = "Exam_Score = " + " ".join(equation_parts)
    st.code(equation_text, language="python")

    st.write("### Coeficienții modelului")
    st.dataframe(results["coef_df"], use_container_width=True)
    st.markdown(
        """
        Coeficienții arată sensul și intensitatea influenței fiecărei variabile asupra lui `Exam_Score`.
        Un coeficient pozitiv sugerează creșterea scorului, iar un coeficient negativ sugerează o scădere,
        menținând constante celelalte variabile.
        """
    )

    selected_coef_feature = st.selectbox(
        "Selectează un coeficient pentru interpretare:",
        results["coef_df"]["Variabila"].tolist(),
    )
    selected_coef_value = float(
        results["coef_df"].loc[
            results["coef_df"]["Variabila"] == selected_coef_feature, "Coeficient"
        ].iloc[0]
    )

    if selected_coef_value > 0:
        interpretation = (
            f"La o creștere cu o unitate a variabilei `{selected_coef_feature}`, "
            f"scorul estimat la examen crește, în medie, cu **{selected_coef_value:.2f}** unități, "
            "menținând constante celelalte variabile din model."
        )
    elif selected_coef_value < 0:
        interpretation = (
            f"La o creștere cu o unitate a variabilei `{selected_coef_feature}`, "
            f"scorul estimat la examen scade, în medie, cu **{abs(selected_coef_value):.2f}** unități, "
            "menținând constante celelalte variabile din model."
        )
    else:
        interpretation = (
            f"Coeficientul asociat variabilei `{selected_coef_feature}` este foarte apropiat de zero, "
            "ceea ce sugerează un efect liniar foarte redus asupra scorului estimat la examen."
        )

    st.info(interpretation)

    st.write("### Distribuția reziduurilor pe setul de test")
    residuals_test = results["y_test"] - results["y_pred_test"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(residuals_test, kde=True, ax=ax, color="#90be6d")
    ax.set_title("Distribuția reziduurilor")
    ax.set_xlabel("Eroare (Real - Prezis)")
    ax.set_ylabel("Frecvență")
    st.pyplot(fig)
    plt.close(fig)
    st.info(
        "Distribuția reziduurilor sugerează o formă apropiată de normalitate, ceea ce indică faptul că "
        "erorile modelului sunt, în ansamblu, echilibrate și nu prezintă abateri extreme sistematice."
    )

    st.write("### Reziduuri vs valori prezise")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x=results["y_pred_test"], y=residuals_test, ax=ax, color="#577590")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Valori prezise")
    ax.set_ylabel("Reziduuri")
    ax.set_title("Residual Plot")
    st.pyplot(fig)
    plt.close(fig)
    st.info(
        "Graficul reziduurilor în raport cu valorile prezise arată o dispersie relativ aleatoare în jurul "
        "liniei 0,ceea ce sugerează că modelul surprinde rezonabil relația "
        "liniară din date și că ipoteza de omoscedasticitate este, în linii mari, acceptabilă. Se observă totuși "
        "și un tipar sub forma unor benzi paralele, explicabil prin natura discretă a unor variabile din setul de "
        "date; acest aspect nu indică neapărat o problemă de modelare, ci reflectă faptul că valorile posibile ale "
        "predictorilor sunt limitate."
    )

    top_features = results["coef_df"].head(3)
    st.success(
        "Cele mai influente variabile în model sunt: "
        + ", ".join(
            f"{row['Variabila']} ({row['Coeficient']:.2f})"
            for _, row in top_features.iterrows()
        )
        + "."
    )

    st.write("### Concluzie pentru regresie")
    show_regression_conclusion(results)


def show_regression_conclusion(linear_results):
    st.write("**Metrici Regresie Liniară pe setul de test:**")
    
    metrics_data = {
        "Metrica": ["R² (Coeficient de determinare)", "RMSE (Root Mean Squared Error)", "MAE (Mean Absolute Error)"],
        "Valoare": [
            f"{linear_results['test_metrics']['R2']:.3f}",
            f"{linear_results['test_metrics']['RMSE']:.3f}",
            f"{linear_results['test_metrics']['MAE']:.3f}"
        ]
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)

    st.info(
        "Regresia liniară oferă o interpretare clară a coeficienților. Valorile metricilor de mai sus "
        "indică performanța modelului pe datele de test (date nevăzute în antrenament)."
    )


def plot_real_vs_predicted(y_true, y_pred, dataset_label):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x=y_true, y=y_pred, ax=ax, color="#277da1", alpha=0.75)
    ax.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        color="red",
        linestyle="--",
    )
    ax.set_xlabel("Valori reale")
    ax.set_ylabel("Valori prezise")
    ax.set_title(f"Real vs Predicted - {dataset_label}")
    st.pyplot(fig)
    plt.close(fig)
