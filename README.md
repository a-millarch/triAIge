# Prehospital prediction of surgical needs

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├───src
│   │   predict_model.py
│   │   train_model.py
│   │   __init__.py
│   │
│   ├───common
│   │       log_config.py
│   │
│   ├───custom
│   │       custom_fusion_model.py
│   │       tsai_custom.py
│   │
│   ├───data
│   │   │   creators.py
│   │   │   datasets.py
│   │   │   datasets_legacy.py
│   │   │   list_dumps.py
│   │   │   make_data.py
│   │   │   mapping.py
│   │   │   tools.py
│   │   │   __init__.py
│   │   │
│   │   └───tmp_del
│   │           1.0-pre-processing.py
│   │           archive.py
│   │           dataloader.py
│   │
│   ├───evaluation
│   │       metrics.py
│   │
│   ├───features
│   │       fusion.py
│   │       fusion_legacy.py
│   │       loader.py
│   │       __init__.py
│   │
│   ├───models
│   │       fastbinary.py
│   │       ml_utils.py
│   │       modelcomponents.py
│   │       timeseries_classification.py
│   │       tree_utils.py
│   │
│   ├───scripts
│   │       pre_process.py
│   │       training.py
│   │       utils.py
│   │       __init__,py
│   │
│   └───visualization
│           visualize.py
│
└── LICENSE              <- Open-source license if one is chosen
```
