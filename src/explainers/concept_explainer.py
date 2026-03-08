"""Concept explanation methods for Concept Bottleneck Models."""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients, Saliency, GradientShap
from captum.attr import visualization as viz
import logging

from .cbm import ConceptBottleneckModel

logger = logging.getLogger(__name__)


class ConceptExplainer:
    """Explainer class for Concept Bottleneck Models."""
    
    def __init__(self, model: ConceptBottleneckModel, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def explain_concept_importance(
        self,
        X: torch.Tensor,
        concept_idx: int,
        method: str = "integrated_gradients",
    ) -> torch.Tensor:
        """
        Explain which input features are most important for a specific concept.
        
        Args:
            X: Input tensor
            concept_idx: Index of the concept to explain
            method: Attribution method ('integrated_gradients', 'saliency', 'gradient_shap')
            
        Returns:
            Attribution scores for each input feature
        """
        X = X.to(self.device)
        
        def concept_model(x):
            return self.model.predict_concepts(x)[:, concept_idx:concept_idx+1]
        
        if method == "integrated_gradients":
            ig = IntegratedGradients(concept_model)
            attributions = ig.attribute(X, target=0)
        elif method == "saliency":
            saliency = Saliency(concept_model)
            attributions = saliency.attribute(X, target=0)
        elif method == "gradient_shap":
            gs = GradientShap(concept_model)
            baseline = torch.zeros_like(X)
            attributions = gs.attribute(X, baselines=baseline, target=0)
        else:
            raise ValueError(f"Unknown attribution method: {method}")
        
        return attributions
    
    def explain_task_from_concepts(
        self,
        concepts: torch.Tensor,
        concept_names: List[str],
    ) -> Dict[str, float]:
        """
        Explain which concepts are most important for task prediction.
        
        Args:
            concepts: Concept tensor
            concept_names: List of concept names
            
        Returns:
            Dictionary of concept importance scores
        """
        concepts = concepts.to(self.device)
        
        # Get task predictions
        with torch.no_grad():
            task_pred = self.model.predict_from_concepts(concepts)
        
        # Calculate concept importance using gradients
        concepts.requires_grad_(True)
        task_pred = self.model.predict_from_concepts(concepts)
        
        # For each class, calculate concept importance
        concept_importance = {}
        
        for class_idx in range(task_pred.shape[1]):
            # Compute gradients w.r.t. concepts
            grad = torch.autograd.grad(
                outputs=task_pred[:, class_idx].sum(),
                inputs=concepts,
                retain_graph=True,
                create_graph=False
            )[0]
            
            # Average importance across samples
            importance = torch.mean(torch.abs(grad), dim=0)
            
            for i, concept_name in enumerate(concept_names):
                if i < len(importance):
                    key = f"{concept_name}_class_{class_idx}"
                    concept_importance[key] = importance[i].item()
        
        return concept_importance
    
    def analyze_concept_interactions(
        self,
        X: torch.Tensor,
        concept_names: List[str],
        top_k: int = 5,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Analyze interactions between concepts.
        
        Args:
            X: Input tensor
            concept_names: List of concept names
            top_k: Number of top interactions to return
            
        Returns:
            Dictionary of concept interactions
        """
        X = X.to(self.device)
        
        with torch.no_grad():
            concepts = self.model.predict_concepts(X)
        
        concepts_np = concepts.cpu().numpy()
        
        # Calculate mutual information between concepts
        interactions = {}
        
        for i, concept_name in enumerate(concept_names):
            concept_i = concepts_np[:, i]
            concept_interactions = []
            
            for j, other_concept_name in enumerate(concept_names):
                if i != j:
                    concept_j = concepts_np[:, j]
                    # Discretize for mutual information calculation
                    concept_i_discrete = np.digitize(concept_i, bins=np.linspace(0, 1, 10))
                    concept_j_discrete = np.digitize(concept_j, bins=np.linspace(0, 1, 10))
                    
                    mi_score = mutual_info_score(concept_i_discrete, concept_j_discrete)
                    concept_interactions.append((other_concept_name, mi_score))
            
            # Sort by mutual information and take top_k
            concept_interactions.sort(key=lambda x: x[1], reverse=True)
            interactions[concept_name] = concept_interactions[:top_k]
        
        return interactions
    
    def concept_sensitivity_analysis(
        self,
        X: torch.Tensor,
        concept_names: List[str],
        perturbation_std: float = 0.1,
        n_samples: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze sensitivity of concepts to input perturbations.
        
        Args:
            X: Input tensor
            concept_names: List of concept names
            perturbation_std: Standard deviation of perturbations
            n_samples: Number of perturbation samples
            
        Returns:
            Dictionary of sensitivity metrics
        """
        X = X.to(self.device)
        
        with torch.no_grad():
            original_concepts = self.model.predict_concepts(X)
        
        sensitivity_results = {}
        
        for i, concept_name in enumerate(concept_names):
            concept_sensitivities = []
            
            for _ in range(n_samples):
                # Add noise to input
                noise = torch.randn_like(X) * perturbation_std
                perturbed_X = X + noise
                
                with torch.no_grad():
                    perturbed_concepts = self.model.predict_concepts(perturbed_X)
                
                # Calculate sensitivity
                sensitivity = torch.abs(original_concepts[:, i] - perturbed_concepts[:, i])
                concept_sensitivities.append(sensitivity.mean().item())
            
            sensitivity_results[concept_name] = {
                'mean_sensitivity': np.mean(concept_sensitivities),
                'std_sensitivity': np.std(concept_sensitivities),
                'max_sensitivity': np.max(concept_sensitivities),
                'min_sensitivity': np.min(concept_sensitivities)
            }
        
        return sensitivity_results
    
    def concept_completeness_analysis(
        self,
        X: torch.Tensor,
        concept_names: List[str],
    ) -> Dict[str, float]:
        """
        Analyze how well concepts capture the input information.
        
        Args:
            X: Input tensor
            concept_names: List of concept names
            
        Returns:
            Dictionary of completeness scores
        """
        X = X.to(self.device)
        
        with torch.no_grad():
            concepts = self.model.predict_concepts(X)
        
        X_np = X.cpu().numpy()
        concepts_np = concepts.cpu().numpy()
        
        completeness_scores = {}
        
        for i, concept_name in enumerate(concept_names):
            # Try to reconstruct input features from concepts
            concept_i = concepts_np[:, i:i+1]
            
            # Use linear regression to predict input features from concept
            reg = LinearRegression()
            reg.fit(concept_i, X_np)
            predicted_X = reg.predict(concept_i)
            
            # Calculate R² score
            from sklearn.metrics import r2_score
            r2 = r2_score(X_np, predicted_X)
            completeness_scores[concept_name] = r2
        
        return completeness_scores


def visualize_concept_attributions(
    attributions: torch.Tensor,
    feature_names: List[str],
    concept_name: str,
    save_path: Optional[str] = None,
) -> None:
    """Visualize concept attributions."""
    attributions_np = attributions.cpu().numpy()
    
    # Average across samples
    avg_attributions = np.mean(np.abs(attributions_np), axis=0)
    
    # Sort by importance
    sorted_indices = np.argsort(avg_attributions)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_names)), avg_attributions[sorted_indices])
    plt.xticks(range(len(feature_names)), 
               [feature_names[i] for i in sorted_indices], 
               rotation=45, ha='right')
    plt.title(f'Feature Importance for {concept_name}')
    plt.xlabel('Features')
    plt.ylabel('Attribution Score')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_concept_interactions(
    interactions: Dict[str, List[Tuple[str, float]]],
    save_path: Optional[str] = None,
) -> None:
    """Visualize concept interactions."""
    # Create interaction matrix
    concept_names = list(interactions.keys())
    n_concepts = len(concept_names)
    interaction_matrix = np.zeros((n_concepts, n_concepts))
    
    for i, concept_name in enumerate(concept_names):
        for other_concept, score in interactions[concept_name]:
            j = concept_names.index(other_concept)
            interaction_matrix[i, j] = score
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix, 
                xticklabels=concept_names,
                yticklabels=concept_names,
                annot=True, fmt='.3f', cmap='viridis')
    plt.title('Concept Interaction Matrix (Mutual Information)')
    plt.xlabel('Concepts')
    plt.ylabel('Concepts')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_concept_sensitivity(
    sensitivity_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """Visualize concept sensitivity analysis."""
    concept_names = list(sensitivity_results.keys())
    mean_sensitivities = [sensitivity_results[name]['mean_sensitivity'] for name in concept_names]
    std_sensitivities = [sensitivity_results[name]['std_sensitivity'] for name in concept_names]
    
    plt.figure(figsize=(12, 6))
    plt.bar(concept_names, mean_sensitivities, yerr=std_sensitivities, capsize=5)
    plt.title('Concept Sensitivity Analysis')
    plt.xlabel('Concepts')
    plt.ylabel('Mean Sensitivity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_concept_explanation_report(
    explainer: ConceptExplainer,
    X: torch.Tensor,
    concept_names: List[str],
    feature_names: List[str],
    target_names: List[str],
    save_dir: str = "assets/",
) -> Dict[str, Any]:
    """
    Create a comprehensive concept explanation report.
    
    Args:
        explainer: ConceptExplainer instance
        X: Input tensor
        concept_names: List of concept names
        feature_names: List of feature names
        target_names: List of target names
        save_dir: Directory to save plots
        
        Returns:
            Dictionary containing all explanation results
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    report = {}
    
    # Concept importance analysis
    logger.info("Analyzing concept importance...")
    with torch.no_grad():
        concepts = explainer.model.predict_concepts(X)
    
    concept_importance = explainer.explain_task_from_concepts(concepts, concept_names)
    report['concept_importance'] = concept_importance
    
    # Concept interactions
    logger.info("Analyzing concept interactions...")
    interactions = explainer.analyze_concept_interactions(X, concept_names)
    report['concept_interactions'] = interactions
    
    # Concept sensitivity
    logger.info("Analyzing concept sensitivity...")
    sensitivity = explainer.concept_sensitivity_analysis(X, concept_names)
    report['concept_sensitivity'] = sensitivity
    
    # Concept completeness
    logger.info("Analyzing concept completeness...")
    completeness = explainer.concept_completeness_analysis(X, concept_names)
    report['concept_completeness'] = completeness
    
    # Feature attributions for each concept
    logger.info("Computing feature attributions...")
    feature_attributions = {}
    for i, concept_name in enumerate(concept_names):
        attributions = explainer.explain_concept_importance(X, i)
        feature_attributions[concept_name] = attributions
        
        # Save attribution plot
        visualize_concept_attributions(
            attributions, feature_names, concept_name,
            save_path=f"{save_dir}/attributions_{concept_name}.png"
        )
    
    report['feature_attributions'] = feature_attributions
    
    # Save interaction plot
    visualize_concept_interactions(
        interactions, save_path=f"{save_dir}/concept_interactions.png"
    )
    
    # Save sensitivity plot
    visualize_concept_sensitivity(
        sensitivity, save_path=f"{save_dir}/concept_sensitivity.png"
    )
    
    return report
