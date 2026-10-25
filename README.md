# ENGE707: cat and dog facial-feature classification

This project classifies eight cat/dog facial features using a reference CNN,
frozen MobileNetV2 features and partial fine-tuning. The fixed grouped,
class-stratified split contains 50,927 training and 12,733 validation images.
Identical decoded images stay together. Related crops or animals without
source metadata may still overlap.

| Experiment | Best epoch | Validation accuracy | Macro F1 | Kaggle score* |
| --- | ---: | ---: | ---: | ---: |
| E0 Reference CNN | 15 | 83.78% | 0.842719 | — |
| E1 Frozen MobileNetV2 | 10 | 95.08% | 0.949144 | 0.9218 |
| E2 Partial fine-tuning | 8 | 98.63% | 0.987329 | 0.9397 |

*The table contains the recorded Kaggle submission scores. Validation
accuracy and Kaggle scores are separate. E2 validation has 12,558 correct predictions,
175 errors, cross-entropy 0.040039216507934836 and weighted F1 0.9862631262638263.

## Run locally or on Kaggle

Use Python 3.14 and the existing CPU dependency set for local development:

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name enge707-assessment3 --display-name "ENGE707 Assessment 3"
python -m jupyter lab
```

`requirements-lock.txt` records the original local Windows environment.
Kaggle uses its provided GPU packages: do not install the CPU requirements
or lock file there. Attach the project files and competition data. Set
`PROJECT_ROOT` to the extracted project folder and `DATA_ROOT` to the
competition dataset mount in the notebook configuration cell. Use notebook
**06** to restore the final checkpoint without retraining.

Place the external dataset under ignored `data/raw/`, retaining
`train_labels.csv`, `sample_submission.csv`, `train/train/` and `test/test/`.
On Kaggle, the default dataset mount is
`/kaggle/input/datasets/leocaoprojects/enge707-2026-s2/`.
Nested image directories are discovered; an existing training cache is
an optional explicit override. No dataset copying happens automatically.

Run notebooks in order: **01** data exploration, **02** preprocessing,
**03** reference CNN, **04** frozen MobileNetV2, **05** fine-tuning,
**06** evaluation/submission and **07** E1 error analysis before fine-tuning.
Each starts independently. Training and test prediction default to disabled.

To use the final checkpoint without retraining, open notebook **06**, keep
`EXPERIMENT="E2"`, and enable only evaluation or prediction as required.
Complete checkpoints load strictly with no pretrained download. Only a new
E1 training run requires official torchvision ImageNet V2 weights.

Original evidence lives in `outputs/metrics/<run>/`, `outputs/models/`,
`outputs/figures/` and `outputs/submissions/`. The final CSV is
`outputs/submissions/submission_E2.csv`. Original configurations and hashes
record the historical code. New runs use exclusive `outputs/reproduction/`
(or `/kaggle/working/reproduction/`) directories and never overwrite evidence.
