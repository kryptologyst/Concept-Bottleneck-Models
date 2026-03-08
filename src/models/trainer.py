"""Training and evaluation utilities for Concept Bottleneck Models."""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

from .cbm import ConceptBottleneckModel, get_device, set_seed

logger = logging.getLogger(__name__)


class CBMTrainer:
    """Trainer class for Concept Bottleneck Models."""
    
    def __init__(
        self,
        model: ConceptBottleneckModel,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        concept_loss_weight: float = 1.0,
        task_loss_weight: float = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.concept_loss_weight = concept_loss_weight
        self.task_loss_weight = task_loss_weight
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss functions
        self.concept_criterion = nn.BCELoss()
        self.task_criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'concept_loss': [],
            'task_loss': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_concept_loss = 0.0
        total_task_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc="Training"):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            concepts, task_pred = self.model(batch_x)
            
            # Create concept targets (simplified - using input features as concept proxies)
            concept_targets = torch.sigmoid(batch_x[:, :self.model.concept_dim])
            if concept_targets.shape[1] != self.model.concept_dim:
                # Pad or truncate to match concept dimension
                if concept_targets.shape[1] < self.model.concept_dim:
                    padding = torch.zeros(
                        concept_targets.shape[0], 
                        self.model.concept_dim - concept_targets.shape[1]
                    ).to(self.device)
                    concept_targets = torch.cat([concept_targets, padding], dim=1)
                else:
                    concept_targets = concept_targets[:, :self.model.concept_dim]
            
            # Compute losses
            concept_loss = self.concept_criterion(concepts, concept_targets)
            task_loss = self.task_criterion(task_pred, batch_y)
            total_loss_batch = (
                self.concept_loss_weight * concept_loss + 
                self.task_loss_weight * task_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += total_loss_batch.item()
            total_concept_loss += concept_loss.item()
            total_task_loss += task_loss.item()
            
            _, predicted = torch.max(task_pred.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        return {
            'loss': total_loss / len(train_loader),
            'concept_loss': total_concept_loss / len(train_loader),
            'task_loss': total_task_loss / len(train_loader),
            'accuracy': 100 * correct / total
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_concept_loss = 0.0
        total_task_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                concepts, task_pred = self.model(batch_x)
                
                # Create concept targets
                concept_targets = torch.sigmoid(batch_x[:, :self.model.concept_dim])
                if concept_targets.shape[1] != self.model.concept_dim:
                    if concept_targets.shape[1] < self.model.concept_dim:
                        padding = torch.zeros(
                            concept_targets.shape[0], 
                            self.model.concept_dim - concept_targets.shape[1]
                        ).to(self.device)
                        concept_targets = torch.cat([concept_targets, padding], dim=1)
                    else:
                        concept_targets = concept_targets[:, :self.model.concept_dim]
                
                # Compute losses
                concept_loss = self.concept_criterion(concepts, concept_targets)
                task_loss = self.task_criterion(task_pred, batch_y)
                total_loss_batch = (
                    self.concept_loss_weight * concept_loss + 
                    self.task_loss_weight * task_loss
                )
                
                # Statistics
                total_loss += total_loss_batch.item()
                total_concept_loss += concept_loss.item()
                total_task_loss += task_loss.item()
                
                _, predicted = torch.max(task_pred.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return {
            'loss': total_loss / len(val_loader),
            'concept_loss': total_concept_loss / len(val_loader),
            'task_loss': total_task_loss / len(val_loader),
            'accuracy': 100 * correct / total
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['concept_loss'].append(train_metrics['concept_loss'])
            self.history['task_loss'].append(train_metrics['task_loss'])
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if verbose and epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch:3d}: "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                    f"Val Acc: {val_metrics['accuracy']:.2f}%"
                )
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        return self.history


class CBMEvaluator:
    """Evaluator class for Concept Bottleneck Models."""
    
    def __init__(self, model: ConceptBottleneckModel, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device
    
    def evaluate_concept_completeness(
        self,
        test_loader: DataLoader,
        concept_names: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate concept completeness - how well concepts capture the input information.
        
        Args:
            test_loader: Test data loader
            concept_names: List of concept names
            
        Returns:
            Dictionary of completeness metrics
        """
        self.model.eval()
        concept_predictions = []
        input_features = []
        
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                concepts = self.model.predict_concepts(batch_x)
                concept_predictions.append(concepts.cpu().numpy())
                input_features.append(batch_x.cpu().numpy())
        
        concept_predictions = np.vstack(concept_predictions)
        input_features = np.vstack(input_features)
        
        # Calculate reconstruction quality
        from sklearn.metrics import r2_score
        
        completeness_scores = {}
        for i, concept_name in enumerate(concept_names):
            if i < input_features.shape[1]:
                # Use input features to predict concepts
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()
                reg.fit(input_features, concept_predictions[:, i])
                pred_concepts = reg.predict(input_features)
                completeness_scores[concept_name] = r2_score(concept_predictions[:, i], pred_concepts)
            else:
                completeness_scores[concept_name] = 0.0
        
        return completeness_scores
    
    def evaluate_concept_sensitivity(
        self,
        test_loader: DataLoader,
        concept_names: List[str],
        perturbation_std: float = 0.1,
    ) -> Dict[str, float]:
        """
        Evaluate concept sensitivity - how much concepts change with input perturbations.
        
        Args:
            test_loader: Test data loader
            concept_names: List of concept names
            perturbation_std: Standard deviation of perturbations
            
        Returns:
            Dictionary of sensitivity metrics
        """
        self.model.eval()
        sensitivities = {}
        
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                
                # Original concepts
                original_concepts = self.model.predict_concepts(batch_x)
                
                # Perturbed concepts
                noise = torch.randn_like(batch_x) * perturbation_std
                perturbed_x = batch_x + noise
                perturbed_concepts = self.model.predict_concepts(perturbed_x)
                
                # Calculate sensitivity (mean absolute difference)
                sensitivity = torch.mean(torch.abs(original_concepts - perturbed_concepts), dim=0)
                
                for i, concept_name in enumerate(concept_names):
                    if i < len(sensitivity):
                        if concept_name not in sensitivities:
                            sensitivities[concept_name] = []
                        sensitivities[concept_name].append(sensitivity[i].item())
        
        # Average across batches
        avg_sensitivities = {}
        for concept_name, values in sensitivities.items():
            avg_sensitivities[concept_name] = np.mean(values)
        
        return avg_sensitivities
    
    def evaluate_task_performance(
        self,
        test_loader: DataLoader,
        target_names: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate task performance metrics.
        
        Args:
            test_loader: Test data loader
            target_names: List of target class names
            
        Returns:
            Dictionary of performance metrics
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_concepts = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                concepts, task_pred = self.model(batch_x)
                
                _, predicted = torch.max(task_pred, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                all_concepts.extend(concepts.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Classification report
        report = classification_report(
            all_targets, all_predictions, 
            target_names=target_names, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets,
            'concepts': np.array(all_concepts)
        }
    
    def compare_with_baselines(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_names: List[str],
    ) -> Dict[str, float]:
        """
        Compare CBM performance with baseline models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            target_names: List of target class names
            
        Returns:
            Dictionary of baseline accuracies
        """
        # Logistic Regression baseline
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_test, X_test)  # Simplified - would need proper training
        lr_pred = lr_model.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_pred)
        
        # Decision Tree baseline
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_test, y_test)
        dt_pred = dt_model.predict(X_test)
        dt_acc = accuracy_score(y_test, dt_pred)
        
        return {
            'logistic_regression': lr_acc,
            'decision_tree': dt_acc
        }


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Concept loss
    axes[1, 0].plot(history['concept_loss'], label='Concept Loss')
    axes[1, 0].set_title('Concept Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Task loss
    axes[1, 1].plot(history['task_loss'], label='Task Loss')
    axes[1, 1].set_title('Task Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, target_names: List[str], save_path: Optional[str] = None) -> None:
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
