# ChestMNIST

## Structure of the project

The project follows a modular and scalable structure designed to support the full machine learning lifecycle.

```plaintext
project/
│
├── data/                      # Raw and processed datasets
├── docs/                      # Project documentation
│   └── conceptual_model/      # Data conceptual model and supporting documents
│
├── logs/                      # Logs generated during execution
│
├── notebooks/                 # Exploratory Data Analysis notebooks
│   └── eda
│
├── reports/                   # Generated reports, metrics, and visualizations
│
├── src/                       # Source code (core of the project)
│   ├── config/                # Environment config: device, paths, dataset flag
│   ├── data/                  # Data ingestion and DataLoader factories
│   ├── eda/                   # EDA visualization functions (save to reports/)
│   ├── entity/                # Experiment config schemas (OptunaConfig, TrainingConfig)
│   ├── features/              # Transform pipelines (torchvision augmentation)
│   ├── models/                # Architecture, loss, trainer, evaluator
│   ├── pipelines/             # Runnable pipelines (hparam_search, train, eval, gradcam, pipeline_final)
│   └── utils/                 # Plotting utilities
│
├── tests/                     # Unit and integration tests
├── trained_models/            # Saved trained models (artifacts)
│
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
└── requirements.txt           # Project dependencies