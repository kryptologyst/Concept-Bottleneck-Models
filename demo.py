#!/usr/bin/env python3
"""Demonstration script for Concept Bottleneck Models."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
from models.cbm import (
    ConceptBottleneckModel,
    load_iris_dataset,
    prepare_data,
    get_device,
    set_seed
)
from models.trainer import CBMTrainer, CBMEvaluator
from explainers.concept_explainer import ConceptExplainer


def main():
    """Demonstrate the Concept Bottleneck Models implementation."""
    print("🧠 Concept Bottleneck Models - Demonstration")
    print("=" * 50)
    print("⚠️  DISCLAIMER: For research and educational purposes only!")
    print("   XAI outputs may be unstable or misleading.")
    print("   Not a substitute for human judgment.")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load and prepare data
    print("\n📊 Loading Iris dataset...")
    X, y, feature_names, concept_names, target_names = load_iris_dataset()
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
        X, y, test_size=0.2, val_size=0.1, random_state=42, standardize=True
    )
    
    print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Classes: {target_names}")
    print(f"   Concepts: {concept_names}")
    
    # Get device
    device = get_device()
    print(f"\n🖥️  Using device: {device}")
    
    # Create model
    print("\n🏗️  Creating Concept Bottleneck Model...")
    model = ConceptBottleneckModel(
        input_dim=X_train.shape[1],
        concept_dim=4,
        task_dim=len(target_names),
        hidden_dims=[32, 16],
        dropout_rate=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {total_params:,} parameters")
    
    # Create data loaders
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Train model
    print("\n🚀 Training model...")
    trainer = CBMTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        concept_loss_weight=1.0,
        task_loss_weight=1.0
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        patience=10,
        verbose=False
    )
    
    print(f"✅ Training completed in {len(history['train_loss'])} epochs")
    print(f"   Final train accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"   Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    evaluator = CBMEvaluator(model, device)
    
    # Task performance
    task_results = evaluator.evaluate_task_performance(test_loader, target_names)
    print(f"✅ Test accuracy: {task_results['accuracy']:.4f}")
    
    # Concept completeness
    completeness_scores = evaluator.evaluate_concept_completeness(test_loader, concept_names)
    print(f"\n🔍 Concept Completeness Scores:")
    for concept, score in completeness_scores.items():
        print(f"   {concept}: {score:.4f}")
    
    # Concept sensitivity
    sensitivity_scores = evaluator.evaluate_concept_sensitivity(test_loader, concept_names)
    print(f"\n🎯 Concept Sensitivity Scores:")
    for concept, score in sensitivity_scores.items():
        print(f"   {concept}: {score:.4f}")
    
    # Concept explanations
    print("\n🔬 Analyzing concept explanations...")
    explainer = ConceptExplainer(model, device)
    
    # Analyze concept interactions
    X_test_tensor = torch.FloatTensor(X_test)
    interactions = explainer.analyze_concept_interactions(X_test_tensor, concept_names)
    
    print(f"\n🔗 Top Concept Interactions:")
    for concept, interaction_list in interactions.items():
        if interaction_list:
            top_interaction = interaction_list[0]
            print(f"   {concept} ↔ {top_interaction[0]}: {top_interaction[1]:.4f}")
    
    # Sample prediction
    print(f"\n🎲 Sample Prediction Analysis:")
    sample_idx = 0
    sample_x = X_test_tensor[sample_idx:sample_idx+1]
    sample_y = y_test[sample_idx]
    
    model.eval()
    with torch.no_grad():
        concepts, task_pred = model(sample_x)
        predicted_class = torch.argmax(task_pred, dim=1).item()
        confidence = task_pred[0][predicted_class].item()
    
    print(f"   Sample {sample_idx}:")
    print(f"   True class: {target_names[sample_y]}")
    print(f"   Predicted class: {target_names[predicted_class]}")
    print(f"   Confidence: {confidence:.4f}")
    
    print(f"\n   Concept activations:")
    for i, concept_name in enumerate(concept_names):
        print(f"   {concept_name}: {concepts[0][i]:.4f}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Demonstration completed successfully!")
    print("\nKey Features Demonstrated:")
    print("✅ Modern PyTorch 2.x implementation")
    print("✅ Concept bottleneck architecture")
    print("✅ Comprehensive evaluation metrics")
    print("✅ Concept completeness and sensitivity analysis")
    print("✅ Concept interaction analysis")
    print("✅ Individual prediction explanations")
    
    print("\nNext Steps:")
    print("1. Run interactive demo: streamlit run demo/app.py")
    print("2. Explore notebook: jupyter notebook notebooks/quickstart.ipynb")
    print("3. Run full training: python scripts/train.py")
    print("4. Run tests: pytest tests/")
    
    print("\n⚠️  Remember: This is for research and educational purposes only!")
    print("   XAI outputs may be unstable or misleading.")
    print("   Not a substitute for human judgment.")


if __name__ == "__main__":
    main()
