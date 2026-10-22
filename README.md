# Reproducible Machine Learning and Deep Learning for MOOC Dropout Prediction

## Overview

This repository provides source code for reproducing and extending experiments on **Massive Open Online Course (MOOC) student dropout prediction**.

It builds on the benchmark introduced by Prenkaj, Velardi, Distante, and Faralli in their study, *A Reproducibility Study of Deep and Surface Machine Learning Methods for Human-Related Trajectory Prediction* (CIKM 2020).

The original benchmark investigates machine learning and deep learning approaches for predicting risks from sequential interaction data. This repository retains the benchmark-oriented code structure and extends the model collection with additional machine learning, ensemble learning, and attention-based deep learning approaches.

The experiments focus on two MOOC datasets:

- **KDDCup15**
- **XuetangX**

To keep the repository lightweight, **only the source code is distributed here**. The datasets must be downloaded separately from the original benchmark resource page.

## Research objectives

The project aims to:

- Reproduce baseline experiments for MOOC dropout prediction.
- Compare traditional machine learning, ensemble learning, and deep learning approaches.
- Investigate sequential models that capture students' temporal interaction patterns.
- Evaluate attention-based architectures for dropout prediction.
- Facilitate reproducible experimentation through centralized configuration, repeated runs, and saved performance results.

## Repository structure

The repository contains the `src/` directory:

```text
repository/
└── src/
    ├── config/
    │   └── conf.py
    ├── models/
    │   ├── ml/
    │   ├── dl/
    │   └── ensemble/
    ├── test_model/
    ├── train_models/
    ├── main.py
    ├── reader.py
    ├── utils.py
    └── data_splitting.py
```

The exact filenames and module organization may depend on the version of the source code committed to the repository.

### Main components

| Component | Purpose |
|---|---|
| `config/` | Defines model selection, hyperparameters, dataset paths, and training settings. |
| `models/ml/` | Contains traditional machine learning models. |
| `models/ensemble/` | Contains ensemble learning approaches. |
| `models/dl/` | Contains deep learning architectures, including sequential and attention-based models. |
| `train_models/` | Implements model training procedures. |
| `test_model/` | Implements evaluation and testing procedures. |
| `reader.py` | Loads and prepares the time-series data. |
| `data_splitting.py` | Provides data splitting utilities. |
| `utils.py` | Provides model construction, directory management, and supporting utilities. |
| `main.py` | Coordinates the experiment workflow. |

## Datasets

### KDDCup15

KDDCup15 contains learner interaction records from online courses and is used for MOOC dropout prediction research.

### XuetangX

XuetangX contains learner activity trajectories from online courses and supports the evaluation of predictive models on a second MOOC dataset.

### Download the datasets

The datasets are not included in this repository because of their size. Download the benchmark resources from the original project page:

**Benchmark resource:** https://iim.di.uniroma1.it/benchmark.html

After downloading and extracting the resources, locate the `kddcup15` and `xuetang` data directories and configure the paths to match the structure expected by the source code.

The original benchmark package uses a structure similar to:

```text
resources/
└── data/
    ├── kddcup15/
    └── xuetang/
```

The `resources/` directory is not required to be committed to this repository.

**Important:** Check the original benchmark instructions and the repository's data reader to determine the exact expected files and preprocessing requirements. Do not assume that the raw datasets can be passed directly to the models without preparation.

## Implemented models

The codebase includes models from the original benchmark and additional models introduced in this extension.

### Traditional machine learning

Depending on the classes included in the committed source code, the model collection includes:

- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Linear Discriminant Analysis (LDA)

### Ensemble learning

The extended implementation includes the following ensemble approaches where available:

- Random Forest
- AdaBoost
- XGBoost
- Kotsiantis-style ensemble

### Deep learning

The original benchmark includes architectures such as:

- Simple CNN
- Simple LSTM
- Multilayer DNN
- CFIN
- ConRec
- Hierarchical Attention Network (HAN)

Additional architectures developed for this project include:

- **LSTM-ATT:** an LSTM with temporal attention.
- **CNN-LSTM:** a convolutional and recurrent architecture for combining local patterns with temporal dependencies.
- **CNN-LSTM-ATT:** a CNN-LSTM architecture enhanced with positional encoding and multi-head attention.

The definitive list of available models is determined by the model classes and imports in the source code.

## Requirements

The project uses Python and machine learning/deep learning libraries. The required packages may include:

- Python 3
- NumPy
- pandas
- scikit-learn
- PyTorch
- imbalanced-learn
- SciPy
- SHAP (for supported explainability workflows)
- XGBoost (if the XGBoost model is enabled)

Install the dependencies using the project's requirements file if one is provided:

```bash
pip install -r requirements.txt
```

If no `requirements.txt` is included, install the required packages individually after checking the imports and version constraints in the source code. Use a compatible PyTorch installation for your operating system and hardware.

## Configuration

Before running an experiment, open:

```text
src/config/conf.py
```

Configure the dataset path, model selection, training parameters, and output directories according to the definitions in the actual configuration file.

The selected model is controlled by the `MODEL` variable. For example, the configuration may select a traditional machine learning model:

```python
DEEP_LEARN = False
MODEL = baselines.LogisticRegressor
```

Or a deep learning model:

```python
DEEP_LEARN = True
MODEL = cnn_lstm_att.CNN_LSTM_ATT
```

These examples assume the indicated classes are present and importable in the checked-out version.

Ensure that `DEEP_LEARN` is consistent with the selected architecture and that model-specific hyperparameters match the constructor arguments.

## Running an experiment

From the repository root, run the main script with the dataset name expected by the configured data directory:

```bash
python src/main.py kddcup15
```

For XuetangX, use the corresponding directory name expected by your configuration:

```bash
python src/main.py xuetang
```

These commands assume that `conf.DATA_DIR` points to the directory containing the extracted datasets and that the folder names match the arguments.

If the configuration expects a different working directory or dataset name, adjust the commands accordingly. The script may also need to be launched from `src/` if its imports are resolved relative to that directory.

## Experimental workflow

The experiment pipeline generally consists of:

1. Loading the selected dataset and time-series data.
2. Preparing the input features and dropout labels.
3. Splitting the data into training and testing sets.
4. Optionally applying oversampling to address class imbalance.
5. Initializing and training the selected model.
6. Evaluating predictions on the test set.
7. Saving the resulting performance records.

For reproducible comparisons, document the random seeds, data splitting protocol, preprocessing steps, model hyperparameters, and number of runs used in each experiment.

Oversampling must be applied to the training set only. The test set should remain untouched to avoid data leakage.

## Results and outputs

The code uses configurable directories for saved models and performance results. The exact output locations depend on the settings in `conf.py` and the directory-management functions in `utils.py`.

Performance files may contain the true test labels and predicted labels. These can be used for subsequent evaluation, comparison, and analysis.

For reliable reporting, retain the configuration associated with each experiment and calculate the evaluation metrics using the predictions from the appropriate test split.

## Explainability

The source code includes supporting utilities for model interpretation, including feature-importance extraction and SHAP-based explanations for compatible models.

The applicability of each explanation method depends on the selected architecture. Tree-based feature importance, linear coefficients, SHAP values, and gradient-based explanations are not interchangeable and should be interpreted according to their assumptions.

## Reproducibility guidelines

For reproducible experiments:

- Use fixed random seeds where supported.
- Keep the same train/test splitting protocol across compared models.
- Fit preprocessing transformations using training data only.
- Apply oversampling to training data only.
- Record the dataset version, model configuration, and software dependencies.
- Save predictions and metrics for each run.
- Report whether results correspond to the original benchmark implementation or to an extended model.

Reproducing a published result may require matching the original preprocessing, hyperparameters, evaluation protocol, and software environment—not simply using the same model name.

## Original benchmark and attribution

This project builds on the benchmark and research published by:

Bardh Prenkaj, Paola Velardi, Damiano Distante, and Stefano Faralli.  
**A Reproducibility Study of Deep and Surface Machine Learning Methods for Human-Related Trajectory Prediction.** Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM 2020).

- Benchmark and resources: https://iim.di.uniroma1.it/benchmark.html
- DOI: https://doi.org/10.1145/3340531.3412088

The original benchmark should be cited when using its code, experimental design, or associated resources. Additional models and modifications in this repository should be identified as extensions of the original implementation.

Please consult the original project and publication for the applicable terms governing reuse and redistribution. The inclusion of extensions in this repository does not imply that the original authors endorse them.



## Acknowledgements

We acknowledge the authors of the original benchmark for providing the research and resources that support reproducible studies of machine learning and deep learning for trajectory-based risk prediction.
