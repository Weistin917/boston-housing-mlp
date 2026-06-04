# Boston Housing ML Pipeline

Final project for the Artificial Intelligence course — Bachelor's Degree in Systems Engineering, Computer Science, and Information Technology, Universidad Galileo.

**Members**
- Samuel Ernesto Caal Ramirez — 23001127
- Wei-Chung Cheng — 23001851
- José Carlos Gordillo Alarcón — 23003115

---

## Objective

Predict the median house value (`MEDV`) in Boston neighborhoods using the classic Boston Housing dataset. Four regression models are trained, evaluated, and compared.

---

## Models

| Model | Type |
|---|---|
| MLP | Multilayer Perceptron (TensorFlow/Keras) |
| SVR | Support Vector Regressor (scikit-learn) |
| Random Forest | Ensemble tree regressor (scikit-learn) |
| XGBoost | Gradient Boosting (XGBoost) |

---

## Results

| Model | Best checkpoint |
|---|---|
| MLP | Epoch 59 — `2026-06-31_3` |
| SVR | `kfold_model-svr_kernel-rbf` — `2026-05-31_1` |
| Random Forest | `kfold_classical_model-random_forest` — `2026-05-31_2` |
| XGBoost | `kfold_classical_model-xgb` — `2026-06-02` |

---

## Project Structure

```
.
├── Boston_Housing.ipynb       # Main notebook
├── config_files/              # YAML experiment configs
│   ├── ModelNameExperiment.yaml
│   └── ModelNameEvaluation.yaml
├── checkpoints/               # Saved model weights
│   └── Experiments/
├── runs/                      # TensorBoard logs
│   └── Experiments/
└── boston.dataset             # Raw dataset file
```

---

## Setup

**Requirements**
```
tensorflow
scikit-learn
xgboost
pandas
numpy
matplotlib
seaborn
pyyaml
joblib
```

Install:
```bash
pip install tensorflow scikit-learn xgboost pandas numpy matplotlib seaborn pyyaml joblib
```

This project runs on **Google Colab**. Mount your Drive and set the `PATH` variable in the notebook to your working directory before running.

---

## Pipeline

1. **Dataset** — Downloaded from [CMU StatLib](http://lib.stat.cmu.edu/datasets/boston). Custom parser handles the dataset's two-line-per-entry format.
2. **Preprocessing** — Normalization of selected columns (`CRIM`, `ZN`, `AGE`, `DIS`, `RAD`, `TAX`, `LSTAT`). Configurable via YAML.
3. **Split** — train-test split + K-Fold cross validation (k=10).
4. **Training** — Controlled by YAML config. MLP supports hyperparameter search over learning rates and batch sizes, with checkpoint saving and resume support.
5. **Evaluation** — Metrics computed on the held-out test set. Results logged to file via `Tee` logger.
6. **Visualization** — Actual vs. Predicted scatter plots and residual error histograms for all four models.

---

## Configuration

Experiments are driven by YAML files in `config_files/`. Set `config_name` in the notebook to switch between experiment and evaluation configs.

Key fields:
```yaml
model:
  name: mlp  # mlp | svr | random_forest | xgb
  params: ...

training:
  hyperparams:
    learning_rates: [0.0001, 0.001]
    search_method: random

eval:
  loss: ...
  metrics: ...

split:
  test_size: 0.2
  split_seed: 51
  fold_n: 10

test: false          # set true to skip training and go straight to evaluation
log_dir: runs/Experiments
ckpt_dir: checkpoints/Experiments
```
