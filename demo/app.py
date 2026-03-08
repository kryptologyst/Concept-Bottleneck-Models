"""Interactive Streamlit demo for Concept Bottleneck Models."""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.cbm import (
    ConceptBottleneckModel, 
    load_iris_dataset, 
    create_synthetic_dataset,
    prepare_data, 
    get_device, 
    set_seed
)
from models.trainer import CBMTrainer, CBMEvaluator
from explainers.concept_explainer import ConceptExplainer


# Page configuration
st.set_page_config(
    page_title="Concept Bottleneck Models Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Concept Bottleneck Models Demo</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Important Disclaimer</h4>
    <p><strong>This demo is for research and educational purposes only.</strong></p>
    <ul>
        <li>XAI outputs may be unstable or misleading</li>
        <li>Not a substitute for human judgment</li>
        <li>Should NOT be used for regulated decisions without human review</li>
        <li>Concept interpretations require domain expertise</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("Configuration")

# Dataset selection
dataset_option = st.sidebar.selectbox(
    "Select Dataset",
    ["Iris", "Synthetic"],
    help="Choose between the classic Iris dataset or a synthetic dataset"
)

# Model parameters
st.sidebar.subheader("Model Parameters")
concept_dim = st.sidebar.slider(
    "Number of Concepts",
    min_value=2,
    max_value=10,
    value=4,
    help="Number of interpretable concepts in the bottleneck"
)

hidden_dims = st.sidebar.multiselect(
    "Hidden Layer Dimensions",
    [16, 32, 64, 128, 256],
    default=[64, 32],
    help="Dimensions of hidden layers"
)

dropout_rate = st.sidebar.slider(
    "Dropout Rate",
    min_value=0.0,
    max_value=0.5,
    value=0.1,
    step=0.05,
    help="Dropout rate for regularization"
)

# Training parameters
st.sidebar.subheader("Training Parameters")
epochs = st.sidebar.slider(
    "Epochs",
    min_value=10,
    max_value=200,
    value=50,
    help="Number of training epochs"
)

learning_rate = st.sidebar.selectbox(
    "Learning Rate",
    [0.001, 0.01, 0.1],
    index=0,
    help="Learning rate for optimization"
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    [16, 32, 64, 128],
    index=1,
    help="Batch size for training"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None


def load_and_prepare_data():
    """Load and prepare dataset based on selection."""
    set_seed(42)
    
    if dataset_option == "Iris":
        X, y, feature_names, concept_names, target_names = load_iris_dataset()
    else:  # Synthetic
        X, y, feature_names, concept_names, target_names = create_synthetic_dataset(
            n_samples=500,
            n_features=10,
            n_classes=3,
            n_concepts=concept_dim,
            random_state=42
        )
    
    # Prepare data splits
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
        X, y, test_size=0.2, val_size=0.1, random_state=42, standardize=True
    )
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'feature_names': feature_names, 'concept_names': concept_names[:concept_dim],
        'target_names': target_names, 'scaler': scaler
    }


def create_model():
    """Create Concept Bottleneck Model."""
    data = st.session_state.data
    
    model = ConceptBottleneckModel(
        input_dim=data['X_train'].shape[1],
        concept_dim=concept_dim,
        task_dim=len(data['target_names']),
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    )
    
    return model


def train_model():
    """Train the model."""
    if st.session_state.data is None:
        st.error("Please load data first!")
        return
    
    # Create model
    model = create_model()
    device = get_device()
    
    # Create trainer
    trainer = CBMTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        concept_loss_weight=1.0,
        task_loss_weight=1.0
    )
    
    # Create data loaders
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(
        torch.FloatTensor(st.session_state.data['X_train']),
        torch.LongTensor(st.session_state.data['y_train'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(st.session_state.data['X_val']),
        torch.LongTensor(st.session_state.data['y_val'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train model
    with st.spinner("Training model..."):
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            patience=10,
            verbose=False
        )
    
    # Store in session state
    st.session_state.model = model
    st.session_state.trainer = trainer
    st.session_state.training_history = history
    
    return history


# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Model Training", "Concept Analysis", "Predictions"])

with tab1:
    st.header("Data Overview")
    
    if st.button("Load Data"):
        with st.spinner("Loading data..."):
            st.session_state.data = load_and_prepare_data()
        st.success("Data loaded successfully!")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(data['X_train']) + len(data['X_val']) + len(data['X_test']))
        with col2:
            st.metric("Features", data['X_train'].shape[1])
        with col3:
            st.metric("Classes", len(data['target_names']))
        
        # Feature statistics
        st.subheader("Feature Statistics")
        feature_df = pd.DataFrame(data['X_train'], columns=data['feature_names'])
        st.dataframe(feature_df.describe(), use_container_width=True)
        
        # Class distribution
        st.subheader("Class Distribution")
        class_counts = pd.Series(data['y_train']).value_counts().sort_index()
        class_counts.index = [data['target_names'][i] for i in class_counts.index]
        
        fig = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title="Training Set Class Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation
        st.subheader("Feature Correlation Matrix")
        corr_matrix = feature_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.header("Model Training")
    
    if st.session_state.data is None:
        st.warning("Please load data first in the Data Overview tab.")
    else:
        if st.button("Train Model"):
            history = train_model()
            st.success("Model trained successfully!")
        
        if st.session_state.training_history is not None:
            history = st.session_state.training_history
            
            # Training curves
            st.subheader("Training Progress")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Loss", "Accuracy", "Concept Loss", "Task Loss"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            epochs_range = range(len(history['train_loss']))
            
            # Loss
            fig.add_trace(
                go.Scatter(x=epochs_range, y=history['train_loss'], name='Train Loss', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs_range, y=history['val_loss'], name='Val Loss', line=dict(color='red')),
                row=1, col=1
            )
            
            # Accuracy
            fig.add_trace(
                go.Scatter(x=epochs_range, y=history['train_acc'], name='Train Acc', line=dict(color='blue')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=epochs_range, y=history['val_acc'], name='Val Acc', line=dict(color='red')),
                row=1, col=2
            )
            
            # Concept Loss
            fig.add_trace(
                go.Scatter(x=epochs_range, y=history['concept_loss'], name='Concept Loss', line=dict(color='green')),
                row=2, col=1
            )
            
            # Task Loss
            fig.add_trace(
                go.Scatter(x=epochs_range, y=history['task_loss'], name='Task Loss', line=dict(color='orange')),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Final metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Train Loss", f"{history['train_loss'][-1]:.4f}")
            with col2:
                st.metric("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
            with col3:
                st.metric("Final Train Acc", f"{history['train_acc'][-1]:.2f}%")
            with col4:
                st.metric("Final Val Acc", f"{history['val_acc'][-1]:.2f}%")


with tab3:
    st.header("Concept Analysis")
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the Model Training tab.")
    else:
        # Initialize evaluator and explainer
        if st.session_state.evaluator is None:
            device = get_device()
            st.session_state.evaluator = CBMEvaluator(st.session_state.model, device)
            st.session_state.explainer = ConceptExplainer(st.session_state.model, device)
        
        data = st.session_state.data
        
        # Create test data loader
        from torch.utils.data import DataLoader, TensorDataset
        test_dataset = TensorDataset(
            torch.FloatTensor(data['X_test']),
            torch.LongTensor(data['y_test'])
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Concept completeness
        st.subheader("Concept Completeness")
        completeness_scores = st.session_state.evaluator.evaluate_concept_completeness(
            test_loader, data['concept_names']
        )
        
        completeness_df = pd.DataFrame(
            list(completeness_scores.items()),
            columns=['Concept', 'Completeness Score']
        )
        completeness_df = completeness_df.sort_values('Completeness Score', ascending=False)
        
        fig = px.bar(
            completeness_df,
            x='Concept',
            y='Completeness Score',
            title='Concept Completeness Scores',
            color='Completeness Score',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Concept sensitivity
        st.subheader("Concept Sensitivity")
        sensitivity_scores = st.session_state.evaluator.evaluate_concept_sensitivity(
            test_loader, data['concept_names']
        )
        
        sensitivity_df = pd.DataFrame(
            list(sensitivity_scores.items()),
            columns=['Concept', 'Sensitivity Score']
        )
        sensitivity_df = sensitivity_df.sort_values('Sensitivity Score', ascending=False)
        
        fig = px.bar(
            sensitivity_df,
            x='Concept',
            y='Sensitivity Score',
            title='Concept Sensitivity Scores',
            color='Sensitivity Score',
            color_continuous_scale='plasma'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Concept interactions
        st.subheader("Concept Interactions")
        X_test_tensor = torch.FloatTensor(data['X_test'])
        interactions = st.session_state.explainer.analyze_concept_interactions(
            X_test_tensor, data['concept_names']
        )
        
        # Create interaction matrix
        interaction_matrix = np.zeros((len(data['concept_names']), len(data['concept_names'])))
        for i, concept_name in enumerate(data['concept_names']):
            for other_concept, score in interactions[concept_name]:
                j = data['concept_names'].index(other_concept)
                interaction_matrix[i, j] = score
        
        fig = px.imshow(
            interaction_matrix,
            x=data['concept_names'],
            y=data['concept_names'],
            title='Concept Interaction Matrix (Mutual Information)',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)


with tab4:
    st.header("Predictions & Explanations")
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the Model Training tab.")
    else:
        data = st.session_state.data
        
        # Sample selection
        st.subheader("Select Sample for Analysis")
        
        sample_idx = st.selectbox(
            "Choose a test sample",
            range(len(data['X_test'])),
            format_func=lambda x: f"Sample {x} (Class: {data['target_names'][data['y_test'][x]]})"
        )
        
        # Get sample data
        sample_x = torch.FloatTensor(data['X_test'][sample_idx:sample_idx+1])
        sample_y = data['y_test'][sample_idx]
        
        # Make predictions
        st.session_state.model.eval()
        with torch.no_grad():
            concepts, task_pred = st.session_state.model(sample_x)
            predicted_class = torch.argmax(task_pred, dim=1).item()
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("True Class", data['target_names'][sample_y])
        with col2:
            st.metric("Predicted Class", data['target_names'][predicted_class])
        with col3:
            confidence = task_pred[0][predicted_class].item()
            st.metric("Confidence", f"{confidence:.3f}")
        
        # Feature values
        st.subheader("Feature Values")
        feature_df = pd.DataFrame(
            [data['X_test'][sample_idx]],
            columns=data['feature_names']
        )
        st.dataframe(feature_df, use_container_width=True)
        
        # Concept predictions
        st.subheader("Concept Predictions")
        concept_df = pd.DataFrame(
            [concepts[0].numpy()],
            columns=data['concept_names']
        )
        st.dataframe(concept_df, use_container_width=True)
        
        # Concept visualization
        fig = px.bar(
            x=data['concept_names'],
            y=concepts[0].numpy(),
            title='Concept Activation Values',
            color=concepts[0].numpy(),
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_title="Concepts", yaxis_title="Activation Value")
        st.plotly_chart(fig, use_container_width=True)
        
        # Task prediction probabilities
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame(
            [task_pred[0].numpy()],
            columns=data['target_names']
        )
        st.dataframe(prob_df, use_container_width=True)
        
        fig = px.bar(
            x=data['target_names'],
            y=task_pred[0].numpy(),
            title='Class Prediction Probabilities',
            color=task_pred[0].numpy(),
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_title="Classes", yaxis_title="Probability")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Concept Bottleneck Models Demo - For Research and Educational Purposes Only</p>
    <p>⚠️ XAI outputs may be unstable or misleading. Not a substitute for human judgment.</p>
</div>
""", unsafe_allow_html=True)
