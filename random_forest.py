import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


RF_FEATURES = [
    "Attendance",
    "Hours_Studied",
    "Access_to_Resources",
    "Parental_Involvement",
    "Tutoring_Sessions",
    "Parental_Education_Level",
    "Peer_Influence",
]

EXCLUDED_RF_FEATURES = {"Exam_Score", "Performance"}

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


def show_random_forest(df_clean):
    st.title("Random Forest Regressor")

    st.markdown(
        """
        ### Contextul analizei

        În această secțiune folosim `Random Forest Regressor` pentru a estima variabila țintă `Exam_Score`.
        Scopul este să verificăm dacă un model mai flexibil decât regresia liniară poate surprinde mai bine
        relațiile dintre predictori și scorul final.

        Pentru o comparație clară, construim două variante:

        - `Random Forest restrâns`, bazat pe aceleași variabile finale folosite în regresia liniară
        - `Random Forest extins`, construit pe toate variabilele numerice disponibile după preprocesare

        În acest mod putem compara un model mai simplu și mai ușor de urmărit cu unul mai bogat în informație
        și putem observa dacă variabilele suplimentare aduc un câștig real de performanță.
        """
    )

    restricted_features = [feature for feature in RF_FEATURES if feature in df_clean.columns]
    extended_features = get_extended_rf_features(df_clean)

    restricted_results = prepare_random_forest_results(
        df_clean,
        restricted_features,
        model_name="Random Forest restrâns",
    )
    extended_results = prepare_random_forest_results(
        df_clean,
        extended_features,
        model_name="Random Forest extins",
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "1. Ce face algoritmul",
            "2. Modelele Random Forest",
            "3. Evaluarea și comparația",
        ]
    )

    with tab1:
        show_rf_concepts(restricted_features, extended_features)

    with tab2:
        show_rf_model_building(restricted_results, extended_results)

    with tab3:
        show_rf_evaluation(restricted_results, extended_results)


def get_extended_rf_features(df):
    return [
        column
        for column in df.columns
        if column not in EXCLUDED_RF_FEATURES and pd.api.types.is_numeric_dtype(df[column])
    ]


def show_rf_concepts(restricted_features, extended_features):
    st.subheader("Cum funcționează Random Forest")
    st.info(
        "Random Forest este un algoritm de tip ensemble care combină mai mulți arbori de decizie. "
        "Fiecare arbore este antrenat pe un eșantion bootstrap și folosește un subset aleator de variabile "
        "la fiecare split."
    )

    st.markdown(
        """
        Principiul de bază este următorul:

        - se generează mai multe eșantioane bootstrap din datele de antrenare
        - pentru fiecare eșantion se antrenează un arbore de decizie
        - la fiecare nod, arborele caută split-ul optim doar într-un subset aleator de variabile
        - predicția finală este media predicțiilor tuturor arborilor

        Prin această strategie, modelul devine mai stabil decât un singur arbore și generalizează mai bine.
        """
    )

    with st.expander("Detalii teoretice suplimentare", expanded=False):
        st.markdown(
            """
            Din punct de vedere teoretic, Random Forest pornește de la ideea de `bagging`
            (`bootstrap aggregating`). În loc să construim un singur arbore de decizie, construim mulți
            arbori diferiți pe eșantioane bootstrap, adică pe seturi de date obținute prin extrageri repetate,
            cu revenire, din setul de antrenare.

            Fiecare arbore vede astfel o versiune ușor diferită a datelor. În plus, la fiecare nod, arborele
            nu caută cel mai bun split dintre toate variabilele disponibile, ci doar dintre un subset aleator
            de variabile. Această restricție introduce diversitate între arbori și reduce dependența modelului
            de câțiva predictori dominanți.

            În regresie, predicția finală este media predicțiilor tuturor arborilor. Prin această agregare,
            variația arborilor individuali scade, iar modelul final devine mai robust și mai puțin sensibil
            la fluctuațiile din date.
            """
        )

    st.write("### De ce este util în proiectul nostru")
    st.markdown(
        """
        Random Forest ne ajută deoarece:

        - poate surprinde relații neliniare pe care regresia liniară nu le surprinde bine
        - poate valorifica interacțiuni între variabile fără să le definim explicit
        - este mai puțin sensibil la multicoliniaritate decât regresia liniară
        - poate lucra bine și cu un număr mai mare de predictori
        - oferă un clasament al importanței variabilelor prin `feature importance`
        """
    )

    st.write("### Modelele comparate")
    st.markdown(
        """
        - `Random Forest restrâns`: folosește aceleași variabile finale ca regresia liniară, pentru o comparație directă
        - `Random Forest extins`: folosește toate variabilele numerice disponibile după preprocesare
        """
    )

    with st.expander("Variabile în modelul restrâns", expanded=False):
        st.write(", ".join(f"`{feature}`" for feature in restricted_features))

    with st.expander("Variabile în modelul extins", expanded=False):
        st.write(", ".join(f"`{feature}`" for feature in extended_features))


def prepare_random_forest_results(df, available_features, model_name):
    X = df[available_features].copy()
    y = df["Exam_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    importance_df = pd.DataFrame(
        {"Variabila": X.columns, "Importanță": model.feature_importances_}
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


def show_rf_model_building(restricted_results, extended_results):
    st.subheader("Construirea modelelor Random Forest")
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
            - `max_depth = 8`
            - `min_samples_split = 5`
            - `min_samples_leaf = 2`
            - `random_state = 42`
            """
        )

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Grafic test: Random Forest restrâns")
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
        st.write("### Grafic test: Random Forest extins")
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
        Formularul de mai jos folosește modelul `Random Forest restrâns`, deoarece este mai ușor de completat
        și rămâne direct comparabil cu modelul de regresie liniară.
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
                key_prefix="rf_",
            )

    if st.button("Calculează predicția Random Forest"):
        student_df = pd.DataFrame([input_values])
        predicted_score = float(restricted_results["model"].predict(student_df)[0])
        st.success(f"Nota estimată la examen pentru acest student este **{predicted_score:.2f}**.")


def show_rf_evaluation(restricted_results, extended_results):
    st.subheader("Evaluarea și comparația modelelor")
    st.markdown(
        """
        În această etapă comparăm performanța celor două modele Random Forest și analizăm dacă modelul extins,
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
            "aduc informație utilă pentru predicție."
        )
    else:
        st.info(
            "Modelul restrâns obține un `R²` cel puțin la fel de bun pe setul de test, ceea ce sugerează că "
            "variabilele suplimentare nu aduc un câștig clar de performanță."
        )

    if abs(best_results["train_metrics"]["R2"] - best_results["test_metrics"]["R2"]) < 0.1:
        st.success(
            f"Concluzie: modelul `{best_results['model_name']}` este varianta recomandată, "
            "având cea mai bună performanță pe test și o generalizare bună."
        )
    else:
        st.warning(
            f"Concluzie provizorie: modelul `{best_results['model_name']}` are cea mai bună performanță pe test, "
            "dar diferența dintre train și test sugerează că rezultatele trebuie interpretate cu prudență."
        )

    selected_model_name = st.selectbox(
        "Selectează modelul pentru analiza detaliată:",
        [restricted_results["model_name"], extended_results["model_name"]],
    )
    selected_results = (
        restricted_results
        if selected_model_name == restricted_results["model_name"]
        else extended_results
    )

    st.write("### Importanța variabilelor")
    st.caption(
        "Acest tabel și graficul de mai jos arată contribuția relativă a fiecărei variabile în modelul selectat. "
        "În Random Forest, interpretarea predictorilor se face prin importanță, nu prin coeficienți ca în regresia liniară."
    )
    st.dataframe(selected_results["importance_df"], use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=selected_results["importance_df"].head(10),
        x="Importanță",
        y="Variabila",
        ax=ax,
        color="#4c78a8",
    )
    ax.set_title(f"Feature Importance - {selected_results['model_name']}")
    ax.set_xlabel("Importanță")
    ax.set_ylabel("Variabile")
    st.pyplot(fig)
    plt.close(fig)

    selected_feature = st.selectbox(
        "Selectează o variabilă pentru interpretarea importanței:",
        selected_results["importance_df"]["Variabila"].tolist(),
    )
    selected_importance = float(
        selected_results["importance_df"].loc[
            selected_results["importance_df"]["Variabila"] == selected_feature, "Importanță"
        ].iloc[0]
    )

    st.info(
        f"Variabila `{selected_feature}` are o importanță de **{selected_importance:.3f}** în modelul "
        f"`{selected_results['model_name']}`. Cu cât valoarea este mai mare, cu atât variabila contribuie "
        "mai mult la reducerea erorii în arborii din ansamblu."
    )

    st.write("### Distribuția reziduurilor pe setul de test")
    st.caption(
        "Graficul reziduurilor are rolul de a arăta cum sunt distribuite erorile de predicție ale modelului. "
        "El oferă o imagine suplimentară asupra stabilității modelului și asupra prezenței unor abateri mari."
    )
    residuals_test = selected_results["y_test"] - selected_results["y_pred_test"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(residuals_test, kde=True, ax=ax, color="#72b7b2")
    ax.set_title(f"Distribuția reziduurilor - {selected_results['model_name']}")
    ax.set_xlabel("Eroare (Real - Prezis)")
    ax.set_ylabel("Frecvență")
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Rezidurile sunt normal distribuite, observam ca sunt concentrate în jurul valorii 0 și nu apar abateri foarte mari, modelul produce "
        "predicții rezonabil de stabile pentru scorul la examen."
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
