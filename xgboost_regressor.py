import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


XGB_FEATURES = [
    "Attendance",
    "Hours_Studied",
    "Access_to_Resources",
    "Parental_Involvement",
    "Tutoring_Sessions",
    "Parental_Education_Level",
    "Peer_Influence",
]

EXCLUDED_XGB_FEATURES = {"Exam_Score", "Performance"}

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


def show_xgboost_regressor(df_clean):
    st.title("XGBoost Regressor")

    xgb_regressor_cls = get_xgb_regressor_class()
    if xgb_regressor_cls is None:
        st.error(
            "Pachetul `xgboost` nu este instalat în mediul curent. "
            "Pentru a folosi această pagină, instalează-l cu `pip install xgboost` în venv."
        )
        return

    st.markdown(
        """
        ### Contextul analizei

        În această secțiune folosim `XGBoost Regressor` pentru a estima variabila țintă `Exam_Score`.
        Scopul este să evaluăm dacă un model de tip boosting, care construiește arborii secvențial și învață
        din erorile modelului anterior, poate oferi predicții mai bune decât un model bazat pe bagging.

        Pentru o comparație clară, construim două variante:

        - `XGBoost restrâns`, bazat pe aceleași variabile finale folosite în regresia liniară
        - `XGBoost extins`, construit pe toate variabilele numerice disponibile după preprocesare

        Astfel putem observa dacă modelul beneficiază de un set mai larg de predictori și dacă abordarea de tip
        boosting aduce un câștig de performanță în raport cu modelele anterioare.
        """
    )

    restricted_features = [feature for feature in XGB_FEATURES if feature in df_clean.columns]
    extended_features = get_extended_xgb_features(df_clean)

    restricted_results = prepare_xgb_results(
        df_clean,
        restricted_features,
        model_name="XGBoost restrâns",
        regressor_cls=xgb_regressor_cls,
    )
    extended_results = prepare_xgb_results(
        df_clean,
        extended_features,
        model_name="XGBoost extins",
        regressor_cls=xgb_regressor_cls,
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "1. Ce face algoritmul",
            "2. Modelele XGBoost",
            "3. Evaluarea și comparația",
        ]
    )

    with tab1:
        show_xgb_concepts(restricted_features, extended_features)

    with tab2:
        show_xgb_model_building(restricted_results, extended_results)

    with tab3:
        show_xgb_evaluation(restricted_results, extended_results)


def get_xgb_regressor_class():
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None
    return XGBRegressor


def get_extended_xgb_features(df):
    return [
        column
        for column in df.columns
        if column not in EXCLUDED_XGB_FEATURES and pd.api.types.is_numeric_dtype(df[column])
    ]


def show_xgb_concepts(restricted_features, extended_features):
    st.subheader("Cum funcționează XGBoost")
    st.info(
        "XGBoost este un algoritm de tip boosting. Spre deosebire de Random Forest, arborii nu sunt construiți "
        "independent, ci secvențial, fiecare nou arbore încercând să corecteze erorile celui anterior."
    )

    st.markdown(
        """
        Principiul de bază este următorul:

        - modelul pornește de la o predicție inițială
        - se calculează erorile sau reziduurile modelului curent
        - un nou arbore este antrenat pentru a învăța aceste erori
        - predicția finală se actualizează treptat prin adăugarea contribuției fiecărui arbore

        Prin această construcție secvențială, XGBoost poate reduce atât eroarea de aproximare, cât și eroarea de predicție.
        """
    )

    with st.expander("Detalii teoretice suplimentare", expanded=False):
        st.markdown(
            """
            În seminar, ideea centrală a boosting-ului a fost că fiecare arbore nou învață din greșelile modelului precedent.
            Dacă în Random Forest arborii votează independent, în XGBoost arborii cooperează secvențial.

            La fiecare pas, modelul calculează cât de mult mai are de corectat și antrenează un arbore mic pe aceste erori.
            Contribuția noului arbore este controlată de `learning_rate`, iar complexitatea modelului este influențată
            și de parametri precum `max_depth`, `n_estimators` și `subsample`.

            Avantajul major este că XGBoost poate obține performanțe foarte bune pe date tabelare, dar necesită
            mai multă atenție la hiperparametri decât Random Forest, deoarece este mai sensibil la overfitting.
            """
        )

    st.write("### De ce este util în proiectul nostru")
    st.markdown(
        """
        XGBoost ne ajută deoarece:

        - poate surprinde relații complexe dintre variabile și scorul final
        - învață secvențial din erori și poate rafina progresiv predicțiile
        - oferă performanțe foarte bune pe date tabelare
        - permite controlul fin al modelului prin hiperparametri
        - oferă și el un clasament al importanței variabilelor
        """
    )

    st.write("### Modelele comparate")
    st.markdown(
        """
        - `XGBoost restrâns`: folosește aceleași variabile finale ca regresia liniară, pentru o comparație directă
        - `XGBoost extins`: folosește toate variabilele numerice disponibile după preprocesare
        """
    )

    with st.expander("Variabile în modelul restrâns", expanded=False):
        st.write(", ".join(f"`{feature}`" for feature in restricted_features))

    with st.expander("Variabile în modelul extins", expanded=False):
        st.write(", ".join(f"`{feature}`" for feature in extended_features))


def prepare_xgb_results(df, available_features, model_name, regressor_cls):
    X = df[available_features].copy()
    y = df["Exam_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = regressor_cls(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    importance_df = pd.DataFrame(
        {"Variabila": available_features, "Importanță": model.feature_importances_}
    ).sort_values(by="Importanță", ascending=False)

    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)

    return {
        "model_name": model_name,
        "features": available_features,
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "importance_df": importance_df,
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


def show_xgb_model_building(restricted_results, extended_results):
    st.subheader("Construirea modelelor XGBoost")
    st.markdown(
        """
        Ambele modele sunt antrenate pe aceeași împărțire a datelor în:

        - `set de antrenare (train)` pentru învățarea arborilor
        - `set de testare (test)` pentru verificarea generalizării

        Parametrii utilizați sunt aceiași pentru ambele modele, astfel încât comparația să fie corectă,
        iar diferențele de performanță să provină în principal din setul de variabile folosit.
        """
    )

    comparison_df = pd.DataFrame(
        [
            {
                "Model": restricted_results["model_name"],
                "Număr variabile": len(restricted_results["features"]),
                "Observații train": len(restricted_results["X_train"]),
                "Observații test": len(restricted_results["X_test"]),
            },
            {
                "Model": extended_results["model_name"],
                "Număr variabile": len(extended_results["features"]),
                "Observații train": len(extended_results["X_train"]),
                "Observații test": len(extended_results["X_test"]),
            },
        ]
    )
    st.dataframe(comparison_df, use_container_width=True)

    with st.expander("Parametrii modelului", expanded=False):
        st.markdown(
            """
            - `n_estimators = 200`
            - `max_depth = 4`
            - `learning_rate = 0.05`
            - `subsample = 0.8`
            - `colsample_bytree = 0.8`
            - `random_state = 42`
            """
        )

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Grafic test: XGBoost restrâns")
        st.caption(
            "Acest grafic compară valorile reale ale lui `Exam_Score` cu valorile estimate de model. "
            "Rolul lui este de a arăta cât de apropiate sunt predicțiile de situația ideală, reprezentată de diagonala roșie."
        )
        plot_real_vs_predicted(
            restricted_results["y_test"],
            restricted_results["y_pred_test"],
            "Test - Restrâns",
        )

    with col2:
        st.write("### Grafic test: XGBoost extins")
        st.caption(
            "Graficul are același rol pentru modelul extins și permite o comparație vizuală între cele două variante. "
            "Un model mai bun va avea, în general, punctele mai apropiate de diagonala ideală."
        )
        plot_real_vs_predicted(
            extended_results["y_test"],
            extended_results["y_pred_test"],
            "Test - Extins",
        )

    st.write("### Predicție pentru un student")
    st.markdown(
        """
        Formularul de mai jos folosește modelul `XGBoost restrâns`, deoarece este mai ușor de completat
        și rămâne direct comparabil cu modelele anterioare.
        """
    )

    input_values = {}
    input_columns = st.columns(2)

    for index, feature in enumerate(restricted_results["features"]):
        col = input_columns[index % 2]
        with col:
            input_values[feature] = render_prediction_input(
                feature,
                restricted_results["X_train"],
                key_prefix="xgb_",
            )

    if st.button("Calculează predicția XGBoost"):
        student_df = pd.DataFrame([input_values])
        predicted_score = float(restricted_results["model"].predict(student_df)[0])
        st.success(f"Nota estimată la examen pentru acest student este **{predicted_score:.2f}**.")


def show_xgb_evaluation(restricted_results, extended_results):
    st.subheader("Evaluarea și comparația modelelor")
    st.markdown(
        """
        În această etapă comparăm performanța celor două modele XGBoost și analizăm dacă modelul extins,
        construit pe mai multe variabile, oferă un avantaj real față de modelul restrâns.
        """
    )

    metrics_df = pd.DataFrame(
        [
            {
                "Model": restricted_results["model_name"],
                "R² train": restricted_results["train_metrics"]["R2"],
                "R² test": restricted_results["test_metrics"]["R2"],
                "RMSE train": restricted_results["train_metrics"]["RMSE"],
                "RMSE test": restricted_results["test_metrics"]["RMSE"],
                "MAE train": restricted_results["train_metrics"]["MAE"],
                "MAE test": restricted_results["test_metrics"]["MAE"],
            },
            {
                "Model": extended_results["model_name"],
                "R² train": extended_results["train_metrics"]["R2"],
                "R² test": extended_results["test_metrics"]["R2"],
                "RMSE train": extended_results["train_metrics"]["RMSE"],
                "RMSE test": extended_results["test_metrics"]["RMSE"],
                "MAE train": extended_results["train_metrics"]["MAE"],
                "MAE test": extended_results["test_metrics"]["MAE"],
            },
        ]
    )

    st.write("### Comparația performanței")
    st.dataframe(metrics_df, use_container_width=True)

    best_results = max(
        [restricted_results, extended_results],
        key=lambda result: result["test_metrics"]["R2"],
    )

    if extended_results["test_metrics"]["R2"] > restricted_results["test_metrics"]["R2"]:
        st.info(
            "Modelul extins obține un `R²` mai bun pe setul de test, ceea ce sugerează că variabilele suplimentare "
            "aduc informație utilă și pentru modelul XGBoost."
        )
    else:
        st.info(
            "Modelul restrâns obține un `R²` cel puțin la fel de bun pe setul de test, ceea ce sugerează că "
            "variabilele suplimentare nu aduc un câștig clar de performanță."
        )

    if abs(best_results["train_metrics"]["R2"] - best_results["test_metrics"]["R2"]) < 0.1:
        st.success(
            f"Concluzie: modelul `{best_results['model_name']}` oferă cea mai bună performanță pe test și "
            "sugerează o capacitate bună de generalizare."
        )
    else:
        st.warning(
            f"Concluzie provizorie: modelul `{best_results['model_name']}` are cea mai bună performanță pe test, "
            "dar diferența dintre train și test sugerează că rezultatele trebuie interpretate cu prudență."
        )

    selected_model_name = st.selectbox(
        "Selectează modelul pentru analiza detaliată:",
        [restricted_results["model_name"], extended_results["model_name"]],
        key="xgb_model_select",
    )
    selected_results = (
        restricted_results
        if selected_model_name == restricted_results["model_name"]
        else extended_results
    )

    st.write("### Importanța variabilelor")
    st.caption(
        "Acest tabel și graficul de mai jos arată contribuția relativă a fiecărei variabile în modelul selectat. "
        "Și în XGBoost, interpretarea predictorilor se face prin importanță, nu prin coeficienți."
    )
    st.dataframe(selected_results["importance_df"], use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=selected_results["importance_df"].head(10),
        x="Importanță",
        y="Variabila",
        ax=ax,
        color="#d97706",
    )
    ax.set_title(f"Feature Importance - {selected_results['model_name']}")
    ax.set_xlabel("Importanță")
    ax.set_ylabel("Variabile")
    st.pyplot(fig)
    plt.close(fig)

    selected_feature = st.selectbox(
        "Selectează o variabilă pentru interpretarea importanței:",
        selected_results["importance_df"]["Variabila"].tolist(),
        key="xgb_feature_select",
    )
    selected_importance = float(
        selected_results["importance_df"].loc[
            selected_results["importance_df"]["Variabila"] == selected_feature, "Importanță"
        ].iloc[0]
    )

    st.info(
        f"Variabila `{selected_feature}` are o importanță de **{selected_importance:.3f}** în modelul "
        f"`{selected_results['model_name']}`. Cu cât valoarea este mai mare, cu atât variabila contribuie "
        "mai mult la îmbunătățirea predicțiilor."
    )

    st.write("### Distribuția reziduurilor pe setul de test")
    st.caption(
        "Graficul reziduurilor are rolul de a arăta cum sunt distribuite erorile de predicție ale modelului. "
        "El oferă o imagine suplimentară asupra stabilității modelului și asupra prezenței unor abateri mari."
    )
    residuals_test = selected_results["y_test"] - selected_results["y_pred_test"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(residuals_test, kde=True, ax=ax, color="#f4a261")
    ax.set_title(f"Distribuția reziduurilor - {selected_results['model_name']}")
    ax.set_xlabel("Eroare (Real - Prezis)")
    ax.set_ylabel("Frecvență")
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Dacă reziduurile sunt concentrate în jurul valorii 0 și nu apar abateri foarte mari, modelul produce "
        "predicții rezonabil de stabile pentru scorul la examen."
    )


def plot_real_vs_predicted(y_true, y_pred, dataset_label):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x=y_true, y=y_pred, ax=ax, color="#bc6c25", alpha=0.75)
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
