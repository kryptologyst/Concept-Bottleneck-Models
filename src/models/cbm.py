"""Modern Concept Bottleneck Model implementation using PyTorch 2.x."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class ConceptBottleneckModel(nn.Module):
    """
    A Concept Bottleneck Model that separates concept prediction from final classification.
    
    This model consists of two main components:
    1. Concept predictor: Maps input features to interpretable concepts
    2. Task predictor: Maps concepts to final task predictions
    
    Args:
        input_dim: Number of input features
        concept_dim: Number of concepts
        task_dim: Number of output classes
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout rate for regularization
        concept_activation: Activation function for concept layer
        task_activation: Activation function for task layer
    """
    
    def __init__(
        self,
        input_dim: int,
        concept_dim: int,
        task_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.1,
        concept_activation: str = "sigmoid",
        task_activation: str = "softmax",
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.concept_dim = concept_dim
        self.task_dim = task_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build concept predictor (input -> concepts)
        concept_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            concept_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        concept_layers.append(nn.Linear(prev_dim, concept_dim))
        
        if concept_activation == "sigmoid":
            concept_layers.append(nn.Sigmoid())
        elif concept_activation == "tanh":
            concept_layers.append(nn.Tanh())
        elif concept_activation == "relu":
            concept_layers.append(nn.ReLU())
            
        self.concept_predictor = nn.Sequential(*concept_layers)
        
        # Build task predictor (concepts -> task)
        self.task_predictor = nn.Sequential(
            nn.Linear(concept_dim, concept_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(concept_dim // 2, task_dim)
        )
        
        if task_activation == "softmax":
            self.task_activation = nn.Softmax(dim=-1)
        elif task_activation == "log_softmax":
            self.task_activation = nn.LogSoftmax(dim=-1)
        else:
            self.task_activation = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (concepts, task_predictions)
        """
        # Predict concepts
        concepts = self.concept_predictor(x)
        
        # Predict task from concepts
        task_logits = self.task_predictor(concepts)
        task_predictions = self.task_activation(task_logits)
        
        return concepts, task_predictions
    
    def predict_concepts(self, x: torch.Tensor) -> torch.Tensor:
        """Predict concepts only."""
        return self.concept_predictor(x)
    
    def predict_from_concepts(self, concepts: torch.Tensor) -> torch.Tensor:
        """Predict task from given concepts."""
        task_logits = self.task_predictor(concepts)
        return self.task_activation(task_logits)


class ConceptDataset:
    """Dataset wrapper for concept-based learning."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        concept_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
    ) -> None:
        self.X = X
        self.y = y
        self.concept_names = concept_names or [f"concept_{i}" for i in range(X.shape[1])]
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self.target_names = target_names or [f"class_{i}" for i in range(len(np.unique(y)))]
        
    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create PyTorch DataLoader."""
        X_tensor = torch.FloatTensor(self.X)
        y_tensor = torch.LongTensor(self.y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def create_synthetic_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 3,
    n_concepts: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """
    Create a synthetic dataset for concept bottleneck learning.
    
    Args:
        n_samples: Number of samples
        n_features: Number of input features
        n_classes: Number of output classes
        n_concepts: Number of concepts
        random_state: Random seed
        
    Returns:
        Tuple of (X, y, feature_names, concept_names, target_names)
    """
    np.random.seed(random_state)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    
    # Create meaningful names
    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    concept_names = [f"concept_{i:02d}" for i in range(n_concepts)]
    target_names = [f"class_{i}" for i in range(n_classes)]
    
    return X, y, feature_names, concept_names, target_names


def load_iris_dataset() -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """
    Load and prepare the Iris dataset for concept bottleneck learning.
    
    Returns:
        Tuple of (X, y, feature_names, concept_names, target_names)
    """
    data = load_iris()
    X = data.data
    y = data.target
    
    feature_names = data.feature_names
    concept_names = ["sepal_length_concept", "sepal_width_concept", "petal_length_concept", "petal_width_concept"]
    target_names = data.target_names.tolist()
    
    return X, y, feature_names, concept_names, target_names


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    standardize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare data for training with proper splits and preprocessing.
    
    Args:
        X: Input features
        y: Target labels
        test_size: Fraction of data for testing
        val_size: Fraction of data for validation
        random_state: Random seed
        standardize: Whether to standardize features
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    # Standardize features
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    
    logger.info(f"Data splits - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
