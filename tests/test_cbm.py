"""Unit tests for Concept Bottleneck Models."""

import pytest
import torch
import numpy as np
from sklearn.datasets import make_classification
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.cbm import (
    ConceptBottleneckModel,
    create_synthetic_dataset,
    prepare_data,
    get_device,
    set_seed
)
from models.trainer import CBMTrainer, CBMEvaluator
from explainers.concept_explainer import ConceptExplainer


class TestConceptBottleneckModel:
    """Test cases for ConceptBottleneckModel."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = ConceptBottleneckModel(
            input_dim=10,
            concept_dim=5,
            task_dim=3,
            hidden_dims=[32, 16],
            dropout_rate=0.1
        )
        
        assert model.input_dim == 10
        assert model.concept_dim == 5
        assert model.task_dim == 3
        assert len(model.hidden_dims) == 2
        assert model.dropout_rate == 0.1
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = ConceptBottleneckModel(
            input_dim=4,
            concept_dim=3,
            task_dim=2
        )
        
        # Create dummy input
        x = torch.randn(5, 4)  # batch_size=5, input_dim=4
        
        # Forward pass
        concepts, task_pred = model(x)
        
        # Check output shapes
        assert concepts.shape == (5, 3)  # batch_size=5, concept_dim=3
        assert task_pred.shape == (5, 2)  # batch_size=5, task_dim=2
        
        # Check concept values are in [0, 1] (sigmoid activation)
        assert torch.all(concepts >= 0) and torch.all(concepts <= 1)
        
        # Check task predictions sum to 1 (softmax activation)
        assert torch.allclose(task_pred.sum(dim=1), torch.ones(5), atol=1e-6)
    
    def test_concept_prediction_only(self):
        """Test concept prediction without task prediction."""
        model = ConceptBottleneckModel(
            input_dim=4,
            concept_dim=3,
            task_dim=2
        )
        
        x = torch.randn(3, 4)
        concepts = model.predict_concepts(x)
        
        assert concepts.shape == (3, 3)
        assert torch.all(concepts >= 0) and torch.all(concepts <= 1)
    
    def test_task_prediction_from_concepts(self):
        """Test task prediction from given concepts."""
        model = ConceptBottleneckModel(
            input_dim=4,
            concept_dim=3,
            task_dim=2
        )
        
        concepts = torch.randn(3, 3)
        task_pred = model.predict_from_concepts(concepts)
        
        assert task_pred.shape == (3, 2)
        assert torch.allclose(task_pred.sum(dim=1), torch.ones(3), atol=1e-6)


class TestDataUtilities:
    """Test cases for data utility functions."""
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        X, y, feature_names, concept_names, target_names = create_synthetic_dataset(
            n_samples=100,
            n_features=5,
            n_classes=3,
            n_concepts=4,
            random_state=42
        )
        
        assert X.shape == (100, 5)
        assert len(y) == 100
        assert len(feature_names) == 5
        assert len(concept_names) == 4
        assert len(target_names) == 3
        assert len(set(y)) == 3  # 3 unique classes
    
    def test_prepare_data(self):
        """Test data preparation and splitting."""
        # Create dummy data
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_classes=3,
            random_state=42
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
            X, y, test_size=0.2, val_size=0.1, random_state=42
        )
        
        # Check shapes
        assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == 100
        assert len(y_train) + len(y_val) + len(y_test) == 100
        
        # Check that all classes are represented in each split
        assert len(set(y_train)) == 3
        assert len(set(y_val)) == 3
        assert len(set(y_test)) == 3
        
        # Check scaler is fitted
        assert scaler is not None
        assert hasattr(scaler, 'mean_')
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        np_rand1 = np.random.randn(5)
        torch_rand1 = torch.randn(5)
        
        # Reset seed and generate again
        set_seed(42)
        np_rand2 = np.random.randn(5)
        torch_rand2 = torch.randn(5)
        
        # Should be identical
        assert np.allclose(np_rand1, np_rand2)
        assert torch.allclose(torch_rand1, torch_rand2)


class TestTrainer:
    """Test cases for CBMTrainer."""
    
    @pytest.fixture
    def dummy_data(self):
        """Create dummy data for testing."""
        set_seed(42)
        X, y = make_classification(
            n_samples=50,
            n_features=4,
            n_classes=3,
            random_state=42
        )
        
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        return loader
    
    def test_trainer_initialization(self, dummy_data):
        """Test trainer initialization."""
        model = ConceptBottleneckModel(input_dim=4, concept_dim=3, task_dim=3)
        device = torch.device('cpu')
        
        trainer = CBMTrainer(
            model=model,
            device=device,
            learning_rate=0.001,
            weight_decay=1e-4
        )
        
        assert trainer.model == model
        assert trainer.device == device
        assert trainer.learning_rate == 0.001
        assert trainer.weight_decay == 1e-4
    
    def test_train_epoch(self, dummy_data):
        """Test training for one epoch."""
        model = ConceptBottleneckModel(input_dim=4, concept_dim=3, task_dim=3)
        device = torch.device('cpu')
        
        trainer = CBMTrainer(model=model, device=device)
        
        # Train for one epoch
        metrics = trainer.train_epoch(dummy_data)
        
        # Check that metrics are returned
        assert 'loss' in metrics
        assert 'concept_loss' in metrics
        assert 'task_loss' in metrics
        assert 'accuracy' in metrics
        
        # Check that metrics are reasonable
        assert metrics['loss'] > 0
        assert metrics['accuracy'] >= 0 and metrics['accuracy'] <= 100
    
    def test_validation(self, dummy_data):
        """Test validation."""
        model = ConceptBottleneckModel(input_dim=4, concept_dim=3, task_dim=3)
        device = torch.device('cpu')
        
        trainer = CBMTrainer(model=model, device=device)
        
        # Validate
        metrics = trainer.validate(dummy_data)
        
        # Check that metrics are returned
        assert 'loss' in metrics
        assert 'concept_loss' in metrics
        assert 'task_loss' in metrics
        assert 'accuracy' in metrics


class TestEvaluator:
    """Test cases for CBMEvaluator."""
    
    @pytest.fixture
    def dummy_model_and_data(self):
        """Create dummy model and data for testing."""
        model = ConceptBottleneckModel(input_dim=4, concept_dim=3, task_dim=3)
        device = torch.device('cpu')
        
        X, y = make_classification(
            n_samples=30,
            n_features=4,
            n_classes=3,
            random_state=42
        )
        
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        return model, device, loader
    
    def test_evaluator_initialization(self, dummy_model_and_data):
        """Test evaluator initialization."""
        model, device, loader = dummy_model_and_data
        
        evaluator = CBMEvaluator(model, device)
        
        assert evaluator.model == model
        assert evaluator.device == device
    
    def test_concept_completeness_evaluation(self, dummy_model_and_data):
        """Test concept completeness evaluation."""
        model, device, loader = dummy_model_and_data
        
        evaluator = CBMEvaluator(model, device)
        concept_names = ['concept_0', 'concept_1', 'concept_2']
        
        completeness_scores = evaluator.evaluate_concept_completeness(loader, concept_names)
        
        assert len(completeness_scores) == 3
        assert all(name in completeness_scores for name in concept_names)
        assert all(0 <= score <= 1 for score in completeness_scores.values())
    
    def test_concept_sensitivity_evaluation(self, dummy_model_and_data):
        """Test concept sensitivity evaluation."""
        model, device, loader = dummy_model_and_data
        
        evaluator = CBMEvaluator(model, device)
        concept_names = ['concept_0', 'concept_1', 'concept_2']
        
        sensitivity_scores = evaluator.evaluate_concept_sensitivity(loader, concept_names)
        
        assert len(sensitivity_scores) == 3
        assert all(name in sensitivity_scores for name in concept_names)
        assert all(score >= 0 for score in sensitivity_scores.values())
    
    def test_task_performance_evaluation(self, dummy_model_and_data):
        """Test task performance evaluation."""
        model, device, loader = dummy_model_and_data
        
        evaluator = CBMEvaluator(model, device)
        target_names = ['class_0', 'class_1', 'class_2']
        
        results = evaluator.evaluate_task_performance(loader, target_names)
        
        assert 'accuracy' in results
        assert 'classification_report' in results
        assert 'confusion_matrix' in results
        assert 'predictions' in results
        assert 'targets' in results
        assert 'concepts' in results
        
        assert 0 <= results['accuracy'] <= 1
        assert len(results['predictions']) == len(results['targets'])


class TestConceptExplainer:
    """Test cases for ConceptExplainer."""
    
    @pytest.fixture
    def dummy_model_and_data(self):
        """Create dummy model and data for testing."""
        model = ConceptBottleneckModel(input_dim=4, concept_dim=3, task_dim=3)
        device = torch.device('cpu')
        
        X = torch.randn(10, 4)
        concept_names = ['concept_0', 'concept_1', 'concept_2']
        feature_names = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
        
        return model, device, X, concept_names, feature_names
    
    def test_explainer_initialization(self, dummy_model_and_data):
        """Test explainer initialization."""
        model, device, X, concept_names, feature_names = dummy_model_and_data
        
        explainer = ConceptExplainer(model, device)
        
        assert explainer.model == model
        assert explainer.device == device
    
    def test_concept_importance_explanation(self, dummy_model_and_data):
        """Test concept importance explanation."""
        model, device, X, concept_names, feature_names = dummy_model_and_data
        
        explainer = ConceptExplainer(model, device)
        
        # Test with integrated gradients
        attributions = explainer.explain_concept_importance(X, concept_idx=0, method="integrated_gradients")
        
        assert attributions.shape == X.shape
        assert torch.isfinite(attributions).all()
    
    def test_task_from_concepts_explanation(self, dummy_model_and_data):
        """Test task explanation from concepts."""
        model, device, X, concept_names, feature_names = dummy_model_and_data
        
        explainer = ConceptExplainer(model, device)
        
        with torch.no_grad():
            concepts = model.predict_concepts(X)
        
        importance = explainer.explain_task_from_concepts(concepts, concept_names)
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
    
    def test_concept_interactions_analysis(self, dummy_model_and_data):
        """Test concept interactions analysis."""
        model, device, X, concept_names, feature_names = dummy_model_and_data
        
        explainer = ConceptExplainer(model, device)
        
        interactions = explainer.analyze_concept_interactions(X, concept_names)
        
        assert isinstance(interactions, dict)
        assert len(interactions) == len(concept_names)
        
        for concept_name, interaction_list in interactions.items():
            assert isinstance(interaction_list, list)
            for other_concept, score in interaction_list:
                assert isinstance(other_concept, str)
                assert isinstance(score, float)
                assert score >= 0
    
    def test_concept_sensitivity_analysis(self, dummy_model_and_data):
        """Test concept sensitivity analysis."""
        model, device, X, concept_names, feature_names = dummy_model_and_data
        
        explainer = ConceptExplainer(model, device)
        
        sensitivity = explainer.concept_sensitivity_analysis(X, concept_names, n_samples=5)
        
        assert isinstance(sensitivity, dict)
        assert len(sensitivity) == len(concept_names)
        
        for concept_name, metrics in sensitivity.items():
            assert 'mean_sensitivity' in metrics
            assert 'std_sensitivity' in metrics
            assert 'max_sensitivity' in metrics
            assert 'min_sensitivity' in metrics
            
            assert all(isinstance(v, float) for v in metrics.values())
            assert all(v >= 0 for v in metrics.values())
    
    def test_concept_completeness_analysis(self, dummy_model_and_data):
        """Test concept completeness analysis."""
        model, device, X, concept_names, feature_names = dummy_model_and_data
        
        explainer = ConceptExplainer(model, device)
        
        completeness = explainer.concept_completeness_analysis(X, concept_names)
        
        assert isinstance(completeness, dict)
        assert len(completeness) == len(concept_names)
        
        for concept_name, score in completeness.items():
            assert isinstance(score, float)
            assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main([__file__])
