import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings

from models import SVMClassifier, RandomForestClassifier, NeuralNetworkClassifier

warnings.filterwarnings('ignore')

# ---------- Streamlit Config ----------
st.set_page_config(
    page_title="Iris Flower Classification ML App",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        color: #6c5ce7;
        text-align: center;
        margin-bottom: 0.2em;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #636e72;
        margin-bottom: 2rem;
    }
    .stRadio > div {
        flex-direction: column;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Load Data ----------
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    df["species_name"] = df["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
    return df, iris.target_names

# ---------- Session State ----------
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False
if "model_results" not in st.session_state:
    st.session_state.model_results = {}

# ---------- Main App ----------
def main():
    st.markdown('<div class="main-title">🌸 Iris Flower Classification ML Application</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">An Interactive Machine Learning Dashboard for Iris Species Classification</div>', unsafe_allow_html=True)

    df, target_names = load_data()

    st.sidebar.header("🔍 Navigation")
    tab = st.sidebar.radio("Go to", [
        "📊 Data Exploration",
        "🤖 Model Training",
        "📈 Model Comparison",
        "🔮 Prediction Interface",
        "🔍 Feature Analysis"
    ])

    # ---------- 📊 Data Exploration ----------
    if tab == "📊 Data Exploration":
        st.subheader("📊 Data Exploration")
        st.dataframe(df)
        st.write("### Summary Statistics")
        st.dataframe(df.describe())
        st.write("### Species Count")
        st.bar_chart(df["species_name"].value_counts())
        st.write("### Pairplot (Seaborn)")
        fig = sns.pairplot(df, hue="species_name")
        st.pyplot(fig)

    # ---------- 🤖 Model Training ----------
    elif tab == "🤖 Model Training":
        st.markdown("## 🤖 Model Training")
        st.write("Train a machine learning model and evaluate its performance on the Iris dataset.")

        model_choice = st.selectbox("Choose Model", ["SVM", "Random Forest", "Neural Network"])

        if st.button("Train Model"):
            # Prepare data
            X = df.drop(columns=["species", "species_name"])
            y = df["species"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Select and train model
            if model_choice == "SVM":
                model = SVMClassifier()
            elif model_choice == "Random Forest":
                model = RandomForestClassifier()
            else:
                model = NeuralNetworkClassifier()

            model.train(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Save results
            st.session_state.models_trained = True
            st.session_state.model_results = {
                "model": model,
                "X_test": X_test,
                "y_test": y_test,
                "y_pred": y_pred
            }

            # Evaluation Output
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()

            st.markdown("### 📈 Model Evaluation Metrics:")
            st.success(f"**Accuracy: {acc * 100:.0f}%**")

            st.markdown("#### 🔢 Confusion Matrix")
            cm_df = pd.DataFrame(cm, 
                index=["Actual Setosa", "Actual Versicolor", "Actual Virginica"],
                columns=["Pred Setosa", "Pred Versicolor", "Pred Virginica"]
            )
            st.dataframe(cm_df.style.set_properties(**{
                'background-color': '#f0f0f0',
                'color': 'black',
                'text-align': 'center'
            }))

            st.markdown("#### 📝 Classification Report")
            st.dataframe(report_df.round(2).style.set_properties(**{
                'background-color': '#f8f9fa',
                'color': 'black',
                'text-align': 'center'
            }))

    # ---------- Other Tabs (Placeholder) ----------
    elif tab == "📈 Model Comparison":
        st.subheader("📈 Model Comparison")
        st.info("🚧 Coming Soon")

    elif tab == "🔮 Prediction Interface":
        st.subheader("🔮 Prediction Interface")
        st.info("🚧 Coming Soon")

    elif tab == "🔍 Feature Analysis":
        st.subheader("🔍 Feature Analysis")
        st.info("🚧 Coming Soon")

# ---------- Run ----------
if __name__ == "__main__":
    main()
