import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_confusion_matrix_plot(cm, class_names, title):
    """Create an interactive confusion matrix plot using Plotly"""
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        hoverongaps=False,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=400,
        height=400
    )
    
    return fig

def create_feature_importance_plot(importance_dict, feature_names):
    """Create a feature importance comparison plot"""
    
    # Prepare data for plotting
    plot_data = []
    for model_name, importances in importance_dict.items():
        for i, importance in enumerate(importances):
            plot_data.append({
                'Model': model_name,
                'Feature': feature_names[i],
                'Importance': importance
            })
    
    df = pd.DataFrame(plot_data)
    
    fig = px.bar(df, x='Feature', y='Importance', color='Model',
                 title="Feature Importance Comparison Across Models",
                 barmode='group')
    
    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Importance",
        legend_title="Model"
    )
    
    return fig

def create_correlation_heatmap(correlation_matrix):
    """Create an interactive correlation heatmap"""
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        width=600,
        height=600
    )
    
    return fig

def calculate_model_metrics(y_true, y_pred):
    """Calculate comprehensive model evaluation metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics

def create_learning_curve_plot(train_sizes, train_scores, val_scores, title):
    """Create a learning curve plot"""
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig = go.Figure()
    
    # Training scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue'),
        error_y=dict(type='data', array=train_std, visible=True)
    ))
    
    # Validation scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=val_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='red'),
        error_y=dict(type='data', array=val_std, visible=True)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Training Set Size",
        yaxis_title="Score",
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig

def create_prediction_confidence_plot(probabilities, class_names):
    """Create a prediction confidence visualization"""
    
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=probabilities,
            text=[f"{prob:.3f}" for prob in probabilities],
            textposition='auto',
            marker_color=['lightblue' if prob != max(probabilities) else 'darkblue' 
                         for prob in probabilities]
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Species",
        yaxis_title="Probability",
        showlegend=False
    )
    
    return fig