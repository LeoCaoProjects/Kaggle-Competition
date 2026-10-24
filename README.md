# ENGE707 Assessment 3

Setup for an individual eight-class cat/dog image classification assessment.
Class spellings come from the supplied competition labels. Development and
checks use the local Python 3.14.3 CPU environment; E0 and E1 training
was completed on Kaggle using one Tesla T4 (`cuda:0`). Only the supplied
competition dataset is used. Kaggle also provides eventual prediction submission.

## Windows PowerShell setup

Use the existing **64-bit Python 3.14.3**. Do not reuse another assignment's
environment. From the repository root, create a fresh environment if `.venv`
does not already exist:

```powershell
& 'C:\Python314\python.exe' -m venv .venv
.\.venv\Scripts\Activate.ps1
& '.\.venv\Scripts\python.exe' --version
```

Activation is optional: all commands below explicitly use the project Python.
If PowerShell blocks activation, continue with these explicit commands without
changing system execution policy.

Install the stable CPU pair from the official PyTorch index first:

```powershell
& '.\.venv\Scripts\python.exe' -m pip install 'torch==2.14.1+cpu' 'torchvision==0.29.1+cpu' --index-url https://download.pytorch.org/whl/cpu
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' -m pip check
& '.\.venv\Scripts\python.exe' -m ipykernel install --user --name enge707-assessment3 --display-name 'Python (ENGE707 Assessment 3)'
```

`requirements.txt` records the installed direct dependencies. To reproduce the
full resolved environment, install the CPU pair above, then use
`-r requirements-lock.txt` instead of `-r requirements.txt`. Both files include
the official CPU index as an additional source so the `+cpu` pins resolve.
To refresh the lock after an intentional successful dependency change:

```powershell
& '.\.venv\Scripts\python.exe' -c "import pathlib,subprocess,sys; p=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True); pathlib.Path('requirements-lock.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cpu\n'+p,encoding='utf-8')"
```

Do not install TensorFlow, nightly builds, CUDA, WSL, drivers, or another Python
version for this setup. Stable Windows CPython 3.14 CPU wheels were selected
from the [official PyTorch index](https://download.pytorch.org/whl/cpu).

## VS Code and notebooks

1. Open this repository folder in VS Code.
2. Install Microsoft's **Python** (`ms-python.python`) and **Jupyter**
   (`ms-toolsai.jupyter`) extensions.
3. Run **Python: Select Interpreter** from the Command Palette and select
   `.venv\Scripts\python.exe`. The checked-in workspace preference uses a
   relative path; existing VS Code selections may need this manual update.
4. Open `notebooks/00_Environment_Check.ipynb`.
5. Click **Select Kernel**, choose **Jupyter Kernel → Python (ENGE707
   Assessment 3)**, or select the matching `.venv` under **Python Environments**.
6. Click **Run All**. Confirm the executable points into this repository's
   `.venv`, all path checks pass, and the tensor result is `10.0`.

Alternatively, launch JupyterLab using:

```powershell
& '.\.venv\Scripts\python.exe' -m jupyterlab
```

Execute and save the check notebook from PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=enge707-assessment3 --ExecutePreprocessor.timeout=120 notebooks/00_Environment_Check.ipynb
```

Repeat from `notebooks/` with filename `00_Environment_Check.ipynb` to check
that working directory too. The notebook locates the project using structural
markers rather than assuming a particular repository folder name.

Run `01_Data_Access_Check.ipynb` with the same kernel to verify the supplied
data contract; its summary is saved to `outputs/metrics/data_access_check.json`.

Continue in numbered order: 02 training-data audit, 03 fixed split/preprocessing,
04 augmentation/loaders, 05 CPU reference-CNN checks, 06 Kaggle GPU setup, and
07 Kaggle reference-CNN training. Notebooks 06–07 use hosted packages and the
verified transfer bundle, not the local CPU requirements. Notebook 07 keeps
`RUN_FULL_TRAINING=False` by default; its smoke checks do not start a full run.

`08_MobileNetV2_Setup.ipynb` contains E1 setup and guarded training cells for the
existing Kaggle session. It freezes pretrained MobileNetV2 features/BatchNorm,
trains only the eight-class head,
and adds ImageNet normalisation to the existing resize/pad tensors. It deliberately
does not use the weights' centre-crop preset. Full E1 training is disabled by default.

The final notebook08 cell prepares E1 test predictions after a successful saved
E1 evaluation. Paste only that new cell into the existing Kaggle session; leave
`RUN_TEST_PREDICTION=False` until deliberately enabling inference. It preserves
original sample IDs and writes `/kaggle/working/submission_E1.csv` plus
`submission_E1_check.json`. Existing outputs stop the run; no competition upload
is performed. Do not restart, Run All or recopy images for this step.

## Directory purposes and conventions

| Path | Purpose |
| --- | --- |
| `notebooks/00_Environment_Check.ipynb` | Numbered setup checks with Markdown explanations and commented code |
| `notebooks/01_Data_Access_Check.ipynb` | CSV schema, training labels, image counts and exact ID-to-filename checks; no image contents opened |
| `notebooks/02_Training_Data_Exploration.ipynb` | Training class balance, seeded gallery, decoding audit and exact duplicate memberships |
| `data/raw/` | Supplied competition CSVs and nested `train/train/` and `test/test/` images, extracted from the provided ZIP |
| `data/processed/` | Fixed split and class mapping retained; training audit manifest and other derived data ignored |
| `outputs/figures/` | Small genuine figure evidence, deliberately retained when useful |
| `outputs/metrics/` | Small genuine metric and experiment evidence |
| `outputs/models/` | The specific preserved E0/E1 checkpoints are eligible for Git; other weights remain ignored |

Keep credentials, datasets, image archives, environments and large weights out
of Git. Only portable VS Code settings are shared; machine-specific editor state
is ignored. Review outputs before deliberately adding them. ChatGPT Work planning
documents stay outside this repository. Preprocessing, the fixed split, reference
CNN implementation and completed E0 evidence are retained; raw images are excluded.

The prior Phase I/II repository's requested README, requirements, ignore file and
two representative notebooks could not be fetched during inspection. This setup
uses the supplied verified summary: local `.venv`, numbered notebooks, Markdown,
commented code, shared `data/` and `outputs/`, and saved evidence. It copies no
medical data, eLCS dependencies or balancing methods.

## CPU and verification

The local environment is a CPU-only PyTorch installation. `torch.cuda.is_available()` is expected
to be `False`, and `torch.version.cuda` should be `None`. That does not imply the
computer has no GPU. Hardware inspection found an NVIDIA RTX 2060 Max-Q with
6144 MiB VRAM and driver 512.78; E0 used the separate Kaggle GPU runtime.

See the executed notebook and `outputs/metrics/environment_setup.json` for actual
versions, checks, hardware observations and setup errors. Environment metadata
is setup evidence, not dataset analysis or experimental results.

## Saved E0 experiment

`E0_reference_cnn_seed42_attempt2` completed 15 epochs; epoch 15 had the lowest
validation loss (0.4878515617073935), with validation accuracy 0.8378229796591534.
The original exported bytes are preserved in:

- `outputs/metrics/E0_reference_cnn_seed42_attempt2/history.csv`
- `outputs/metrics/E0_reference_cnn_seed42_attempt2/configuration.json`
- `outputs/models/E0_reference_cnn_seed42_attempt2.pt`
- `outputs/figures/loss_accuracy_curves.png`

The completed evaluation of all 12,733 saved validation images is preserved in:

- `outputs/metrics/E0_reference_cnn_seed42_attempt2/validation_metrics.json`
- `outputs/metrics/E0_reference_cnn_seed42_attempt2/classification_report.csv`
- `outputs/metrics/E0_reference_cnn_seed42_attempt2/validation_predictions.csv`
- `outputs/figures/E0_reference_cnn_confusion_matrix.png`

## Saved E1 experiment

`E1_mobilenetv2_frozen_seed42` completed 10 epochs with frozen pretrained features
and classifier-only training. Epoch 10 had the lowest validation loss
(0.1460895589143588), with validation accuracy 0.9508364093300872.
The original exported bytes are preserved in:

- `outputs/metrics/E1_mobilenetv2_frozen_seed42/history.csv`
- `outputs/metrics/E1_mobilenetv2_frozen_seed42/configuration.json`
- `outputs/models/E1_mobilenetv2_frozen_seed42.pt`
- `outputs/figures/E1_mobilenetv2_loss_accuracy_curves.png`
- `outputs/metrics/mobilenetv2_setup_check.json`

The completed E1 evaluation retains all 12,733 validation images and is saved in:

- `outputs/metrics/E1_mobilenetv2_frozen_seed42/validation_metrics.json`
- `outputs/metrics/E1_mobilenetv2_frozen_seed42/classification_report.csv`
- `outputs/metrics/E1_mobilenetv2_frozen_seed42/validation_predictions.csv`
- `outputs/figures/E1_mobilenetv2_confusion_matrix.png`

The first submission and its verification record are preserved in
`outputs/submissions/submission_E1.csv` and
`outputs/metrics/submission_E1_check.json`.
User-confirmed first Kaggle score: **0.9218**.

Run `notebooks/09_Error_Analysis.ipynb` locally with the existing project kernel
to compare saved E0/E1 per-class F1 and inspect 12 reproducibly selected E1
validation errors. It uses saved predictions without model inference and writes
`outputs/figures/E1_validation_error_gallery.png` and
`outputs/metrics/E1_error_analysis_check.json`.

## Saved E2 experiment

`E2_mobilenetv2_finetuned_seed42` completed eight epochs of partial fine-tuning
from the saved E1 checkpoint. Epoch 8 had the lowest recorded validation loss
(0.040039216507934836), with validation accuracy 0.9862561847168774.
The original exported bytes are preserved in:

- `outputs/metrics/E2_mobilenetv2_finetuned_seed42/configuration.json`
- `outputs/metrics/E2_mobilenetv2_finetuned_seed42/history.csv`
- `outputs/metrics/E2_mobilenetv2_finetuned_seed42/completion_status.json`
- `outputs/models/E2_mobilenetv2_finetuned_seed42.pt`
- `outputs/figures/E2_mobilenetv2_loss_accuracy_curves.png`

The verified hosted diagnostic and first E2 submission are preserved in:

- `outputs/metrics/mobilenetv2_finetuning_setup_check.json`
- `outputs/submissions/submission_E2.csv`
- `outputs/metrics/submission_E2_check.json`

The user reports an accepted E2 Kaggle score of **0.9397 (93.97%)**, distinct
from the **98.63% validation accuracy**. This is user-reported, without
independent leaderboard verification or a final-rank claim.

The completed E2 evaluation of all 12,733 saved validation images is preserved in:

- `outputs/metrics/E2_mobilenetv2_finetuned_seed42/validation_metrics.json`
- `outputs/metrics/E2_mobilenetv2_finetuned_seed42/classification_report.csv`
- `outputs/metrics/E2_mobilenetv2_finetuned_seed42/validation_predictions.csv`
- `outputs/figures/E2_mobilenetv2_confusion_matrix.png`

Local recomputation from saved predictions confirmed 12,558 correct predictions
and 175 errors, accuracy 0.9862561847168774, macro F1 0.9873287581357201 and
weighted F1 0.9862631262638263. Per-class metrics and confusion counts match the
saved reports, and the confusion PNG was visually checked. Validation loss
matches epoch-8 history exactly; it was not recomputed from predicted classes.
