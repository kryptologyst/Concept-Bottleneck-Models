 # Concept Bottleneck Models

## DISCLAIMER

**IMPORTANT**: This project is for research and educational purposes only. XAI outputs may be unstable, misleading, or incorrect. This tool is NOT a substitute for human judgment and should NOT be used for regulated decisions without human review.

## Overview

This project implements modern Concept Bottleneck Models (CBMs) for explainable AI, focusing on interpretable concept-based predictions. CBMs separate reasoning (concept prediction) from decision-making (final prediction), enabling transparent AI systems.

## Features

- **Modern PyTorch 2.x implementation** with device fallback (CUDA → MPS → CPU)
- **Comprehensive evaluation** with concept completeness, sensitivity, and faithfulness metrics
- **Interactive demos** with Streamlit for concept visualization
- **Reproducible pipelines** with deterministic seeding and proper data splits
- **Production-ready structure** with type hints, comprehensive testing, and CI/CD

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run the interactive demo
streamlit run demo/app.py

# Train and evaluate models
python scripts/train.py --config configs/default.yaml

# Run tests
pytest tests/
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # CBM implementations
│   ├── data/              # Data loading and preprocessing
│   ├── explainers/        # Concept explanation methods
│   ├── metrics/           # Evaluation metrics
│   ├── viz/               # Visualization utilities
│   └── utils/             # Common utilities
├── data/                  # Datasets and metadata
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit tests
├── assets/                # Generated plots and outputs
└── demo/                  # Interactive Streamlit demo
```

## Limitations

- **Concept Stability**: Concept predictions may vary across different random seeds
- **Concept Completeness**: Not all relevant concepts may be captured
- **Human Interpretation**: Concept meanings require human validation
- **Domain Specificity**: Concepts are dataset-dependent and may not generalize

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
# Concept-Bottleneck-Models
