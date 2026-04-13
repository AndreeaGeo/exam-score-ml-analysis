import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


EXCLUDED_CLASSIFICATION_FEATURES = {"Performance", "Exam_Score"}

ORDINAL_FEATURE_OPTIONS = {
    "Parental_Involvement": {
        "Redus": 0.0,
        "Mediu": 1.0,
        "Ridicat": 2.0,
    },
    "Access_to_Resources": {
        "Redus": 0.0,
        "Mediu": 1.0,
        "Ridicat": 2.0,
    },
    "Motivation_Level": {
        "Redus": 0.0,
        "Mediu": 1.0,
        "Ridicat": 2.0,
    },
    "Family_Income": {
        "Redus": 0.0,
        "Mediu": 1.0,
        "Ridicat": 2.0,
    },
    "Teacher_Quality": {
        "Redus": 0.0,
        "Mediu": 1.0,
        "Ridicat": 2.0,
    },
    "Peer_Influence": {
        "Negativă": 0.0,
        "Neutră": 1.0,
        "Pozitivă": 2.0,
    },
    "Distance_from_Home": {
        "Aproape": 0.0,
        "Moderat": 1.0,
        "Departe": 2.0,
    },
    "Parental_Education_Level": {
        "High School": 0.0,
        "College": 1.0,
        "Postgraduate": 2.0,
    },
    "School_Type": {
        "Public": 0.0,
        "Privat": 1.0,
    },
}

BINARY_FEATURE_OPTIONS = {
    "Gender_Male": {"Feminin": 0.0, "Masculin": 1.0},
    "Internet_Access_Yes": {"Nu": 0.0, "Da": 1.0},
    "Extracurricular_Activities_Yes": {"Nu": 0.0, "Da": 1.0},
    "Learning_Disabilities_Yes": {"Nu": 0.0, "Da": 1.0},
}


def show_classification(df_clean):
    st.title("Random Forest Classification")

    st.markdown(
        """
        ### Contextul analizei

        În această secțiune folosim `RandomForestClassifier` pentru a clasifica studenții în funcție de variabila
        țintă `Performance`, unde:

        - `0` indică performanță scăzută
        - `1` indică performanță ridicată

        Modelul este construit pe setul de date preprocesat, folosind aceeași variabilă țintă `Performance`
        explicată și în secțiunea de regresie logistică. Deoarece această variabilă este obținută pe baza lui
        `Exam_Score`, coloana `Exam_Score` este exclusă din predictori pentru a evita scurgerea de informație
        (`data leakage`).
        """
    )

    feature_columns = get_classification_features(df_clean)
    results = prepare_classification_results(df_clean, feature_columns)

    tab1, tab2, tab3 = st.tabs(
        [
            "1. Ce face algoritmul",
            "2. Modelul de clasificare",
            "3. Evaluarea modelului",
        ]
    )

    with tab1:
        show_classification_concepts(feature_columns)

    with tab2:
        show_classification_model_building(results)

    with tab3:
        show_classification_evaluation(results)


def get_classification_features(df):
    return [
        column
        for column in df.columns
        if column not in EXCLUDED_CLASSIFICATION_FEATURES and pd.api.types.is_numeric_dtype(df[column])
    ]


def show_classification_concepts(feature_columns):
    st.subheader("Cum funcționează Random Forest Classification")
    st.info(
        "Random Forest Classification construiește mai mulți arbori de decizie independenți, fiecare antrenat "
        "pe date ușor diferite. La final, fiecare arbore votează, iar clasa prezisă este cea care primește cele mai multe voturi."
    )

    st.markdown(
        """
        Principiul de bază este următorul:

        - se generează mai multe eșantioane bootstrap din datele de antrenare
        - pentru fiecare eșantion se antrenează un arbore de decizie
        - la fiecare nod, arborele caută split-ul optim doar într-un subset aleator de variabile
        - predicția finală este obținută prin votul majoritar al arborilor

        Această strategie reduce instabilitatea unui singur arbore și duce, în general, la o clasificare mai robustă.
        """
    )

    with st.expander("Detalii teoretice suplimentare", expanded=False):
        st.markdown(
            """
            În seminar, Random Forest a fost prezentat ca un model de tip `bagging` (`bootstrap aggregating`).
            Ideea centrală este că un singur arbore de decizie poate fi instabil, însă o pădure de arbori
            independenți, construiți pe mostre ușor diferite, produce decizii mai stabile.

            În clasificare, predicția finală nu se obține prin medie, ci prin vot majoritar. Astfel, dacă
            majoritatea arborilor clasifică un student în categoria `Performance = 1`, aceasta devine clasa finală.

            Un avantaj important este că modelul poate surprinde relații neliniare și interacțiuni între predictori,
            fără să impună o formulă explicită, ceea ce îl face potrivit pentru probleme de clasificare pe date tabelare.
            """
        )

    st.write("### De ce este util în proiectul nostru")
    st.markdown(
        """
        Random Forest Classification ne ajută deoarece:

        - poate clasifica studenții în două categorii de performanță
        - poate surprinde relații complexe între variabile
        - este robust la zgomot și mai puțin sensibil la overfitting decât un singur arbore
        - oferă un clasament al importanței variabilelor
        - permite interpretarea rezultatelor prin metrici și grafice specifice clasificării
        """
    )

    with st.expander("Variabile folosite în model", expanded=False):
        st.write(", ".join(f"`{feature}`" for feature in feature_columns))


def prepare_classification_results(df, feature_columns):
    X = df[feature_columns].copy()
    y = df["Performance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        oob_score=True,
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    importance_df = pd.DataFrame(
        {"Variabila": X.columns, "Importanță": model.feature_importances_}
    ).sort_values(by="Importanță", ascending=False)

    train_metrics = calculate_classification_metrics(y_train, y_pred_train)
    test_metrics = calculate_classification_metrics(y_test, y_pred_test, y_prob_test)

    return {
        "features": feature_columns,
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_prob_test": y_prob_test,
        "importance_df": importance_df,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "classification_report": classification_report(y_test, y_pred_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test),
        "oob_score": model.oob_score_,
    }


def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average="weighted"),
    }
    if y_prob is not None:
        metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob)
    return metrics


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

    if feature in BINARY_FEATURE_OPTIONS:
        options = BINARY_FEATURE_OPTIONS[feature]
        feature_mean = float(values_df[feature].mean())
        default_value = 1.0 if feature_mean >= 0.5 else 0.0
        labels = list(options.keys())
        values = list(options.values())
        default_index = values.index(default_value)
        selected_label = st.selectbox(
            feature,
            labels,
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


def show_classification_model_building(results):
    st.subheader("Construirea modelului de clasificare")
    st.markdown(
        """
        Modelul este antrenat pe o împărțire stratificată a datelor în:

        - `set de antrenare (train)` pentru învățarea arborilor
        - `set de testare (test)` pentru verificarea performanței pe date nevăzute

        Împărțirea stratificată păstrează proporția claselor în ambele subseturi, ceea ce este important
        pentru o evaluare corectă a modelului de clasificare.
        """
    )

    total_size = len(results["X_train"]) + len(results["X_test"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Observații train", len(results["X_train"]))
    col2.metric("Observații test", len(results["X_test"]))
    col3.metric("Raport split", f"{len(results['X_train']) / total_size:.0%} / {len(results['X_test']) / total_size:.0%}")

    with st.expander("Parametrii modelului", expanded=False):
        st.markdown(
            """
            - `n_estimators = 300`
            - `max_depth = 8`
            - `min_samples_split = 5`
            - `min_samples_leaf = 2`
            - `class_weight = "balanced"`
            - `oob_score = True`
            - `random_state = 42`
            """
        )

    st.info(
        "În plus față de setul de test, modelul calculează și `OOB Score` (Out-of-Bag Score), o estimare internă "
        "a performanței, obținută din exemplele care nu au fost folosite la antrenarea fiecărui arbore."
    )
    st.metric("OOB Score", f"{results['oob_score']:.3f}")

    st.write("### Predicție pentru un student")
    st.markdown(
        """
        Formularul de mai jos estimează probabilitatea ca un student să fie încadrat în clasa de performanță ridicată (`Performance = 1`).
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
                key_prefix="clf_",
            )

    if st.button("Calculează clasificarea"):
        student_df = pd.DataFrame([input_values])
        predicted_class = int(results["model"].predict(student_df)[0])
        predicted_prob = float(results["model"].predict_proba(student_df)[0, 1])
        class_label = "Performanță ridicată" if predicted_class == 1 else "Performanță scăzută"
        st.success(
            f"Clasa estimată este **{class_label}**, cu o probabilitate de **{predicted_prob:.2%}** pentru `Performance = 1`."
        )


def show_classification_evaluation(results):
    st.subheader("Evaluarea modelului")
    st.markdown(
        """
        În această etapă evaluăm performanța modelului prin metrici și grafice specifice clasificării.
        Acestea ne ajută să înțelegem atât acuratețea modelului, cât și modul în care acesta separă cele două clase.
        """
    )

    st.write("### Indicatori de performanță")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy test", f"{results['test_metrics']['Accuracy']:.3f}")
    col2.metric("F1-weighted test", f"{results['test_metrics']['F1']:.3f}")
    col3.metric("ROC-AUC test", f"{results['test_metrics']['ROC_AUC']:.3f}")

    st.info(
        "Accuracy arată proporția de clasificări corecte, F1-weighted combină precizia și recall-ul ținând cont "
        "de distribuția claselor, iar ROC-AUC măsoară cât de bine separă modelul cele două clase."
    )

    st.write("### Matricea de confuzie")
    st.caption(
        "Matricea de confuzie arată câte observații au fost clasificate corect și unde apar erorile. "
        "Ea ajută la înțelegerea tipurilor de greșeli făcute de model."
    )
    plot_confusion_matrix(results["confusion_matrix"])

    st.info(
        "Valorile de pe diagonală reprezintă clasificările corecte, iar valorile din afara diagonalei reprezintă "
        "confuziile dintre cele două clase."
    )

    st.write("### Curba ROC")
    st.caption(
        "Curba ROC arată compromisurile dintre rata de adevărate pozitive și rata de false pozitive pentru praguri diferite. "
        "Cu cât curba este mai aproape de colțul stânga-sus, cu atât modelul separă mai bine clasele."
    )
    plot_roc_curve(results["y_test"], results["y_prob_test"])

    st.write("### Importanța variabilelor")
    st.caption(
        "Graficul și tabelul de mai jos arată ce variabile contribuie cel mai mult la deciziile modelului de clasificare."
    )
    st.dataframe(results["importance_df"], use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=results["importance_df"].head(10),
        x="Importanță",
        y="Variabila",
        ax=ax,
        color="#2563eb",
    )
    ax.set_title("Feature Importance - Random Forest Classification")
    ax.set_xlabel("Importanță")
    ax.set_ylabel("Variabile")
    st.pyplot(fig)
    plt.close(fig)

    selected_feature = st.selectbox(
        "Selectează o variabilă pentru interpretarea importanței:",
        results["importance_df"]["Variabila"].tolist(),
        key="classification_feature_select",
    )
    selected_importance = float(
        results["importance_df"].loc[
            results["importance_df"]["Variabila"] == selected_feature, "Importanță"
        ].iloc[0]
    )

    st.info(
        f"Variabila `{selected_feature}` are o importanță de **{selected_importance:.3f}** în model. "
        "Cu cât valoarea este mai mare, cu atât variabila influențează mai mult deciziile de clasificare."
    )

    with st.expander("Classification report", expanded=False):
        st.text(results["classification_report"])


def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Clasă prezisă")
    ax.set_ylabel("Clasă reală")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color="#1d4ed8", label="Random Forest")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="No Skill")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
