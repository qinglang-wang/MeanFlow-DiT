# Improved MeanFlow training for better 1-step generation 

## Announcement
Parts of the code implementation references the [haidog-yaqub/MeanFlow](https://github.com/haidog-yaqub/MeanFlow).

## Project Structure

The file structure of the project is organized as follows:

```text
mean_flow/
├── configs/                # Configuration files for training and sampling
│   ├── sampler/            # Specific configurations for different sampling strategies
│   ├── baseline.yaml       # Baseline training config
│   └── debug.yaml          # Debug training config
├── core/                   # Core modules and components
│   ├── engine.py           # Training engine and loop logic
│   ├── loss.py             # Loss function definitions
│   ├── normalizer.py       # Data normalization utilities
│   └── scheduler.py        # Training sampling strategy 
├── models/                 # Network
│   └── dit.py              # Diffusion Transformer (DiT)
├── utils/                  # Utility scripts
│   ├── evaluator.py        # Evaluation metrics and scripts
│   ├── visualization.py    # Visualization tools for generated samples
│   └── utils.py            # General utility functions
├── results/                # Directory for experiment outputs, logs, and generated images. For the size limitation, checkpoints are deleted.
├── analysis.ipynb          # Jupyter notebook for analyzing results and sampling strategies
├── main.py                 # Main entry point for the application
├── trainer.py              # Script handling the training procedure
├── LICENSE                 # License
└── README.md               # Documentation
```