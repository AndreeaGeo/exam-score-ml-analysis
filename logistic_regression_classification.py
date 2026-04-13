import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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


def show_logistic_regression_classification(df_clean):
    st.title("Regresie logistică")

    st.markdown(
        """
        ### Contextul analizei

        În această secțiune folosim `LogisticRegression` pentru a clasifica studenții în funcție de variabila
        țintă `Performance`, unde:

        - `0` indică performanță scăzută
        - `1` indică performanță ridicată

        Modelul presupune mai multe etape. Mai întâi, datele sunt împărțite în `train` și `test`, astfel încât
        modelul să fie antrenat pe o parte din observații și evaluat separat pe date nevăzute. Această etapă este
        importantă deoarece permite verificarea capacității reale de generalizare a modelului.

        Predictorii sunt apoi scalați cu `MinMaxScaler`, care transformă valorile numerice într-un interval comparabil.
        În final, modelul este antrenat și evaluat prin metrici specifice clasificării.
        """
    )

    with st.expander("Despre variabila `Performance`", expanded=False):
        st.markdown(
            """
            În aplicație, coloana `Performance` este creată în etapa de preprocesare din [preprocessing.py](c:/Users/Andreea/Downloads/seminar%20psw/PROIECT-psw/Proiect/preprocessing.py),
            pe baza medianei variabilei `Exam_Score`.

            Fragmentul de cod folosit este:
            """
        )
        st.code(
            """mediana_scor = df['Exam_Score'].median()

# Creăm coloana 'Performance':
# 1 dacă este peste mediană (performanță ridicată)
# 0 dacă este sub sau egal cu mediana (performanță scăzută)
df['Performance'] = (df['Exam_Score'] > mediana_scor).astype(int)""",
            language="python",
        )
        st.markdown(
            """
            Astfel, fiecare student este încadrat într-una dintre cele două clase în funcție de poziționarea scorului său
            față de mediana scorurilor din setul de date preprocesat.

            Interpretarea este următoarea:

            - dacă `Exam_Score` este mai mare decât mediană, atunci `Performance = 1`
            - dacă `Exam_Score` este mai mic sau egal cu mediană, atunci `Performance = 0`

            Această transformare permite reformularea problemei dintr-una de regresie într-una de clasificare binară.
            Deoarece `Performance` este obținută direct din `Exam_Score`, coloana `Exam_Score` este exclusă din
            predictori pentru a evita `data leakage`, adică transmiterea directă de informație despre variabila țintă.
            """
        )

    feature_columns = get_classification_features(df_clean)
    results = prepare_logistic_results(df_clean, feature_columns)

    tab1, tab2, tab3 = st.tabs(
        [
            "1. Ce face algoritmul",
            "2. Modelul de clasificare",
            "3. Evaluarea modelului",
        ]
    )

    with tab1:
        show_logistic_concepts(feature_columns)

    with tab2:
        show_logistic_model_building(results)

    with tab3:
        show_logistic_evaluation(results)


def get_classification_features(df):
    return [
        column
        for column in df.columns
        if column not in EXCLUDED_CLASSIFICATION_FEATURES and pd.api.types.is_numeric_dtype(df[column])
    ]


def show_logistic_concepts(feature_columns):
    st.subheader("Cum funcționează regresia logistică")
    st.info(
        "Regresia logistică este un model de clasificare care estimează probabilitatea apartenenței unei observații "
        "la clasa pozitivă. În cazul nostru, aceasta estimează probabilitatea ca un student să aibă `Performance = 1`."
    )

    st.markdown(
        """
        Principiul de bază este următorul:

        - modelul combină liniar predictorii
        - rezultatul este transformat prin funcția logistică (`sigmoid`)
        - se obține o probabilitate între 0 și 1
        - pe baza unui prag de decizie, observația este încadrată într-una dintre cele două clase

        În acest fel, modelul nu prezice direct clasa, ci probabilitatea apartenenței la clasa pozitivă.
        """
    )

    with st.expander("Detalii teoretice suplimentare", expanded=False):
        st.markdown(
            """
            Regresia logistică este un model clasic pentru clasificare binară. Deși poartă denumirea de
            „regresie”, ea este utilizată pentru clasificare deoarece rezultatul final este exprimat sub forma
            unei probabilități, urmată de o decizie de tip clasă.

            Funcția logistică transformă combinația liniară a predictorilor într-o valoare cuprinsă între 0 și 1.
            Astfel, coeficienții modelului pot fi interpretați în sensul influenței predictorilor asupra
            probabilității de apartenență la clasa pozitivă.

            Pentru ca modelul să funcționeze mai stabil, predictorii sunt scalați cu `MinMaxScaler`, deoarece
            variabilele pot avea scale foarte diferite. Această uniformizare contribuie la o estimare mai stabilă
            a coeficienților și la o convergență mai bună a modelului.
            """
        )

    st.write("### De ce este utilă în proiectul nostru")
    st.markdown(
        """
        Regresia logistică ne ajută deoarece:

        - este un model de bază clar și ușor de interpretat
        - oferă probabilități, nu doar etichete de clasă
        - permite evaluarea clară prin confusion matrix, accuracy și ROC-AUC
        - este potrivită pentru comparație cu modele mai complexe, precum Random Forest Classification
        """
    )

    with st.expander("Variabile folosite în model", expanded=False):
        st.write(", ".join(f"`{feature}`" for feature in feature_columns))


def prepare_logistic_results(df, feature_columns):
    X = df[feature_columns].copy()
    y = df["Performance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=100, penalty="l2", solver="lbfgs")
    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]

    coef_df = pd.DataFrame(
        {"Variabila": feature_columns, "Coeficient": model.coef_[0]}
    ).sort_values(by="Coeficient", ascending=False)

    train_metrics = calculate_classification_metrics(y_train, y_pred_train)
    test_metrics = calculate_classification_metrics(y_test, y_pred_test, y_prob_test)

    return {
        "features": feature_columns,
        "model": model,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_prob_test": y_prob_test,
        "coef_df": coef_df,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "classification_report": classification_report(y_test, y_pred_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test),
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


def show_logistic_model_building(results):
    st.subheader("Construirea modelului de clasificare")
    st.markdown(
        """
        Modelul este antrenat pe o împărțire stratificată a datelor în:

        - `set de antrenare (train)` pentru estimarea parametrilor modelului
        - `set de testare (test)` pentru verificarea performanței pe date nevăzute

        Înainte de antrenare, predictorii sunt scalați cu `MinMaxScaler`, exact ca în exemplul din seminar.
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
            - `max_iter = 100`
            - `penalty = "l2"`
            - `solver = "lbfgs"`
            - `test_size = 0.2`
            - `random_state = 42`
            - `stratify = y` (split-ul stratificat asigură echilibru între clase)
            """
        )

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
                key_prefix="log_",
            )

    if st.button("Calculează clasificarea logistică"):
        student_df = pd.DataFrame([input_values])
        student_scaled = results["scaler"].transform(student_df)
        predicted_class = int(results["model"].predict(student_scaled)[0])
        predicted_prob = float(results["model"].predict_proba(student_scaled)[0, 1])
        class_label = "Performanță ridicată" if predicted_class == 1 else "Performanță scăzută"
        st.success(
            f"Clasa estimată este **{class_label}**, cu o probabilitate de **{predicted_prob:.2%}** pentru `Performance = 1`."
        )


def show_logistic_evaluation(results):
    st.subheader("Evaluarea modelului")
    st.markdown(
        """
        În această etapă evaluăm performanța modelului prin metrici și grafice specifice clasificării,
        exact în spiritul pașilor prezentați în seminar.
        """
    )

    st.write("### Indicatori de performanță")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy test", f"{results['test_metrics']['Accuracy']:.3f}")
    col2.metric("F1-weighted test", f"{results['test_metrics']['F1']:.3f}")
    col3.metric("ROC-AUC test", f"{results['test_metrics']['ROC_AUC']:.3f}")

    st.info(
        "Accuracy arată proporția de clasificări corecte, F1-weighted sintetizează precizia și recall-ul, "
        "iar ROC-AUC măsoară cât de bine separă modelul cele două clase."
    )

    st.write("### Matricea de confuzie")
    st.caption(
        "Matricea de confuzie arată câte observații au fost clasificate corect și unde apar greșelile."
    )
    plot_confusion_matrix(results["confusion_matrix"])

    st.write("### Curba ROC")
    st.caption(
        "Curba ROC arată relația dintre rata de adevărate pozitive și rata de false pozitive pentru praguri diferite. "
        "Cu cât curba este mai aproape de colțul stânga-sus, cu atât modelul separă mai bine clasele."
    )
    plot_roc_curve(results["y_test"], results["y_prob_test"])

    st.write("### Coeficienții modelului")
    st.caption(
        "În regresia logistică, coeficienții arată sensul influenței predictorilor asupra probabilității de apartenență la clasa pozitivă."
    )
    st.dataframe(results["coef_df"], use_container_width=True)

    selected_feature = st.selectbox(
        "Selectează o variabilă pentru interpretarea coeficientului:",
        results["coef_df"]["Variabila"].tolist(),
        key="logistic_feature_select",
    )
    selected_coef = float(
        results["coef_df"].loc[
            results["coef_df"]["Variabila"] == selected_feature, "Coeficient"
        ].iloc[0]
    )

    if selected_coef > 0:
        interpretation = (
            f"Coeficientul pozitiv pentru `{selected_feature}` sugerează o asociere pozitivă cu apartenența "
            "la clasa `Performance = 1`. În contextul regresiei logistice, acest lucru indică faptul că variabila "
            "contribuie la creșterea log-odds-ului pentru clasa pozitivă."
        )
    elif selected_coef < 0:
        interpretation = (
            f"Coeficientul negativ pentru `{selected_feature}` sugerează o asociere negativă cu apartenența "
            "la clasa `Performance = 1`, indicând o reducere a log-odds-ului pentru clasa pozitivă."
        )
    else:
        interpretation = (
            f"Coeficientul asociat variabilei `{selected_feature}` este foarte apropiat de zero, "
            "ceea ce indică o influență redusă asupra clasificării."
        )
    st.info(interpretation)

    with st.expander("Classification report", expanded=False):
        st.text(results["classification_report"])

    st.write("### Concluzie pentru clasificare")
    show_classification_conclusion(results)


def show_classification_conclusion(logistic_results):
    from classification import prepare_classification_results

    df_for_comparison = pd.concat(
        [
            logistic_results["X_train"],
            logistic_results["X_test"],
            pd.concat([logistic_results["y_train"], logistic_results["y_test"]]),
        ],
        axis=1,
    )

    rf_results = prepare_classification_results(df_for_comparison, logistic_results["features"])

    comparison_df = pd.DataFrame(
        [
            {
                "Model": "Regresie logistică",
                "Accuracy_test": logistic_results["test_metrics"]["Accuracy"],
                "F1_test": logistic_results["test_metrics"]["F1"],
                "ROC_AUC_test": logistic_results["test_metrics"]["ROC_AUC"],
            },
            {
                "Model": "Random Forest Classification",
                "Accuracy_test": rf_results["test_metrics"]["Accuracy"],
                "F1_test": rf_results["test_metrics"]["F1"],
                "ROC_AUC_test": rf_results["test_metrics"]["ROC_AUC"],
            },
        ]
    ).sort_values(
        by=["Accuracy_test", "F1_test", "ROC_AUC_test"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    st.dataframe(comparison_df, use_container_width=True)

    best_model = comparison_df.iloc[0]
    if best_model["Model"] == "Regresie logistică":
        reason = "oferă cele mai bune rezultate pe setul de test pentru `Accuracy`, `F1-weighted` și `ROC-AUC`"
    else:
        reason = "depășește regresia logistică pe setul de test și separă mai bine cele două clase"

    st.success(
        f"Modelul optim pentru partea de clasificare este **{best_model['Model']}**, deoarece {reason}. "
        f"Pe setul de test acesta obține `Accuracy = {best_model['Accuracy_test']:.3f}`, "
        f"`F1-weighted = {best_model['F1_test']:.3f}` și `ROC-AUC = {best_model['ROC_AUC_test']:.3f}`."
    )

    st.info(
        "Regresia logistică rămâne foarte utilă ca model de bază și pentru interpretare, dar modelul optim de "
        "clasificare este cel care separă cel mai bine clasele pe date nevăzute."
    )


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
    ax.plot(fpr, tpr, color="#1d4ed8", label="Regresie logistică")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="No Skill")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
