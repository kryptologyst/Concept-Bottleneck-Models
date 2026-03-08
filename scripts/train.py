#!/usr/bin/env python3
"""Main training script for Concept Bottleneck Models."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from omegaconf import OmegaConf

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
from models.trainer import CBMTrainer, CBMEvaluator, plot_training_history, plot_confusion_matrix
from explainers.concept_explainer import ConceptExplainer, create_concept_explanation_report


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO").upper())
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Create logs directory
    log_file = log_config.get("file", "logs/training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_dataset(config: Dict[str, Any]) -> tuple:
    """Prepare dataset based on configuration."""
    data_config = config["data"]
    dataset_name = data_config["dataset"]
    
    if dataset_name == "iris":
        X, y, feature_names, concept_names, target_names = load_iris_dataset()
    elif dataset_name == "synthetic":
        synthetic_config = data_config["synthetic"]
        X, y, feature_names, concept_names, target_names = create_synthetic_dataset(
            n_samples=synthetic_config["n_samples"],
            n_features=synthetic_config["n_features"],
            n_classes=synthetic_config["n_classes"],
            n_concepts=synthetic_config["n_concepts"],
            random_state=config["training"]["random_seed"]
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Update model config with actual dimensions
    config["model"]["input_dim"] = X.shape[1]
    config["model"]["task_dim"] = len(set(y))
    
    return X, y, feature_names, concept_names, target_names


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Concept Bottleneck Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for results")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training and only run evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(config["training"]["random_seed"])
    
    # Get device
    if config["device"]["auto_detect"]:
        device = get_device()
    else:
        device = torch.device(config["device"]["device"])
    
    logger.info(f"Using device: {device}")
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    X, y, feature_names, concept_names, target_names = prepare_dataset(config)
    
    # Prepare data splits
    data_config = config["data"]
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
        X, y,
        test_size=data_config["test_size"],
        val_size=data_config["val_size"],
        random_state=config["training"]["random_seed"],
        standardize=data_config["standardize"]
    )
    
    # Create model
    model_config = config["model"]
    model = ConceptBottleneckModel(
        input_dim=model_config["input_dim"],
        concept_dim=model_config["concept_dim"],
        task_dim=model_config["task_dim"],
        hidden_dims=model_config["hidden_dims"],
        dropout_rate=model_config["dropout_rate"],
        concept_activation=model_config["concept_activation"],
        task_activation=model_config["task_activation"]
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create data loaders
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    
    # Train model
    if not args.skip_training:
        logger.info("Starting training...")
        trainer = CBMTrainer(
            model=model,
            device=device,
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            concept_loss_weight=config["training"]["concept_loss_weight"],
            task_loss_weight=config["training"]["task_loss_weight"]
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config["training"]["epochs"],
            patience=config["training"]["patience"]
        )
        
        # Save training history
        plot_training_history(history, save_path=f"{args.output_dir}/training_history.png")
        
        # Save model
        torch.save(model.state_dict(), f"{args.output_dir}/model.pth")
        logger.info(f"Model saved to {args.output_dir}/model.pth")
    else:
        # Load existing model
        model_path = f"{args.output_dir}/model.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.error(f"Model not found at {model_path}")
            return
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = CBMEvaluator(model, device)
    
    # Task performance
    task_results = evaluator.evaluate_task_performance(test_loader, target_names)
    logger.info(f"Test Accuracy: {task_results['accuracy']:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        task_results['confusion_matrix'], 
        target_names,
        save_path=f"{args.output_dir}/confusion_matrix.png"
    )
    
    # Concept completeness
    if config["evaluation"]["concept_completeness"]:
        logger.info("Evaluating concept completeness...")
        completeness_scores = evaluator.evaluate_concept_completeness(test_loader, concept_names)
        logger.info(f"Concept completeness scores: {completeness_scores}")
    
    # Concept sensitivity
    if config["evaluation"]["concept_sensitivity"]:
        logger.info("Evaluating concept sensitivity...")
        sensitivity_scores = evaluator.evaluate_concept_sensitivity(test_loader, concept_names)
        logger.info(f"Concept sensitivity scores: {sensitivity_scores}")
    
    # Baseline comparison
    if config["evaluation"]["baseline_comparison"]:
        logger.info("Comparing with baselines...")
        baseline_scores = evaluator.compare_with_baselines(X_test, y_test, target_names)
        logger.info(f"Baseline accuracies: {baseline_scores}")
    
    # Concept explanations
    logger.info("Generating concept explanations...")
    explainer = ConceptExplainer(model, device)
    
    # Create explanation report
    explanation_report = create_concept_explanation_report(
        explainer=explainer,
        X=torch.FloatTensor(X_test),
        concept_names=concept_names,
        feature_names=feature_names,
        target_names=target_names,
        save_dir=f"{args.output_dir}/explanations/"
    )
    
    # Save results summary
    results_summary = {
        "config": config,
        "task_performance": {
            "accuracy": task_results['accuracy'],
            "classification_report": task_results['classification_report']
        },
        "concept_completeness": completeness_scores if config["evaluation"]["concept_completeness"] else None,
        "concept_sensitivity": sensitivity_scores if config["evaluation"]["concept_sensitivity"] else None,
        "baseline_comparison": baseline_scores if config["evaluation"]["baseline_comparison"] else None,
        "concept_importance": explanation_report["concept_importance"],
        "concept_interactions": explanation_report["concept_interactions"]
    }
    
    # Save results
    import json
    with open(f"{args.output_dir}/results.json", 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info(f"Results saved to {args.output_dir}/results.json")
    logger.info("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
