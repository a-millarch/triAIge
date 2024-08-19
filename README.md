# TriAIge
Code accompanying the paper titled:<br>

__Prehospital Triage of Trauma Patients: Predicting Major Surgery using Artificial Intelligence__
<br>_by Andreas Skov Millarch MSc., Fredrik Folke MD PhD, Søren S. Rudolph MD, Haytham M. Kaafarani MD MPH and Martin Sillesen MD PhD_

## Repository usage
This package is build for Danish pre-hospital data where the expected raw format is akin:

| EventCodeName | CreationTime | ValueFloat | ValueString | ValueDateTime | ValueBool | JournalID |
|:-|-:|-:|:-|-:|-:|-:|
| OMI00002 | 2019-05-31 15:23:11.000000 | NaN | 102.0 | NaN | NaN | 0124FKSK-2311-45AF-0093-342BDA3FA | 
|...|...|...|...|...|...|...|

### Data availability statement
Due to confidentiality issues, we are not at liberty to share the Electronic Health Record (pre- and in-hospital) data used in this study. Access to Danish health data is granted through The Danish Patient Safety Authority. EHR data can then be obtained by reasonable request through the relevant region.

However, the project can be adapted for any raw files containing a timestamp, value and identifier bypassing or modifying the constructors in src/data/creators.py.

### Get started 
🟢 Install the project using "pip install -e ." from project root directory. 

🟢 If possible, use Make commands e.g. "make data", "make train", "make multirun" (for gridsearch hyper-paramater tuning using Optima and Hydra).
    See Makefile for all options and Python-files to run if Make is unavailable



## Project structure

The directory structure of the project looks like this:

```txt
│   .gitignore
│   .pre-commit-config.yaml
│   docker-compose.yaml
│   LICENSE
│   Makefile
│   pyproject.toml
│   README.md
│   requirements.txt
│   requirements_dev.txt
│   
├───.github
│   └───workflows
│           .gitkeep
│
├───configs
│   │   default.yaml
│   │   default_10_20min.yaml
│   │
│   ├───data
│   │       basic_multilabel.yaml
│   │       data.yaml
│   │       data_pretrain.yaml
│   │
│   ├───evaluation
│   │       eval.yaml
│   │
│   ├───experiment
│   │       experiment.yaml
│   │
│   ├───model
│   │       de-novo-clean.yaml
│   │       tabfusion.yaml
│   │
│   └───study
│           10min.yaml
│           20min.yaml
│           5min.yaml
│
├───data
│   │   .gitkeep
│   │
│   ├───data_dumps
│   │       .gitkeep
│   │
│   ├───interim
│   │       .gitkeep
│   │
│   ├───processed
│   │       .gitkeep
│   │
│   ├───raw
│   │       .gitkeep
│   │
│   └───reports
│           .gitkeep
│
├───logging
│       .gitkeep
│
├───mlruns
│       .gitkeep
│
├───models
│       .gitkeep
│
├───notebooks
│       .gitkeep
│
├───reports
│   │   .gitkeep
│   │
│   ├───figures
│   │       .gitkeep
│   │
│   └───tables
│           .gitkeep
│
└───src
    │   entrypoint.py
    │   predict_model.py
    │   train_model.py
    │   __init__.py
    │
    ├───common
    │       log_config.py
    │
    ├───custom
    │       custom_fusion_model.py
    │       tsai_custom.py
    │
    ├───data
    │       creators.py
    │       datasets.py
    │       datasets_legacy.py
    │       list_dumps.py
    │       make_data.py
    │       mapping.py
    │       tools.py
    │       __init__.py
    │
    ├───evaluation
    │       metrics.py
    │
    ├───features
    │       fusion.py
    │       fusion_legacy.py
    │       loader.py
    │       __init__.py
    │
    ├───models
    │       fastbinary.py
    │       ml_utils.py
    │       modelcomponents.py
    │       timeseries_classification.py
    │       tree_utils.py
    │
    ├───scripts
    │       pre_process.py
    │       training.py
    │       utils.py
    │       __init__,py
    │
    └───visualization
            visualize.py
```
