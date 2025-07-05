import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Ensure utils and models are importable
from utils import create_confusion_matrix_plot, create_feature_importance_plot, create_correlation_heatmap
from models import SVMClassifier, RandomForestClassifier, NeuralNetworkClassifier

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classification ML App",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data

def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df, iris.target_names

if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

def main():
    st.markdown('<h1 class="main-header">🌸 Iris Flower Classification ML Application</h1>', unsafe_allow_html=True)
    st.markdown("**An Interactive Machine Learning Dashboard for Iris Species Classification**")

    df, target_names = load_data()

    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Go to", [
        "📊 Data Exploration",
        "🤖 Model Training",
        "📈 Model Comparison",
        "🔮 Prediction Interface",
        "🔍 Feature Analysis"
    ])

    if tab == "📊 Data Exploration":
        st.write("### Coming Soon: Data Exploration Tab")
    elif tab == "🤖 Model Training":
        st.write("### Coming Soon: Model Training Tab")
    elif tab == "📈 Model Comparison":
        st.write("### Coming Soon: Model Comparison Tab")
    elif tab == "🔮 Prediction Interface":
        st.write("### Coming Soon: Prediction Interface Tab")
    elif tab == "🔍 Feature Analysis":
        st.write("### Coming Soon: Feature Analysis Tab")

if __name__ == "__main__":
    main()
