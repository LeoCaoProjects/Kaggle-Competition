# ENGE707 Assessment 3 — local environment

Setup for an individual eight-class cat/dog image classification assessment.
Class spellings will come from the competition labels. Only the supplied
competition dataset may be used; pretrained weights are permitted later.
Kaggle is required for prediction submission, but notebooks can run locally.
This repository currently contains environment checks, data-access checks and a
training-only image audit.

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

## Directory purposes and conventions

| Path | Purpose |
| --- | --- |
| `notebooks/00_Environment_Check.ipynb` | Numbered setup checks with Markdown explanations and commented code |
| `notebooks/01_Data_Access_Check.ipynb` | CSV schema, training labels, image counts and exact ID-to-filename checks; no image contents opened |
| `notebooks/02_Training_Data_Exploration.ipynb` | Training class balance, seeded gallery, decoding audit and exact duplicate memberships |
| `data/raw/` | Supplied competition CSVs and nested `train/train/` and `test/test/` images, extracted from the provided ZIP |
| `data/processed/` | Ignored derived data, including the complete training audit manifest |
| `outputs/figures/` | Small genuine figure evidence, deliberately retained when useful |
| `outputs/metrics/` | Small genuine metric and experiment evidence |
| `outputs/models/` | Future local model weights, ignored by Git |

Keep credentials, datasets, image archives, environments and large weights out
of Git. Only portable VS Code settings are shared; machine-specific editor state
is ignored. Review outputs before deliberately adding them. ChatGPT Work planning
documents stay outside this repository. No models, preprocessing, splits,
training, report paragraphs or Kaggle submission are included here.

The prior Phase I/II repository's requested README, requirements, ignore file and
two representative notebooks could not be fetched during inspection. This setup
uses the supplied verified summary: local `.venv`, numbered notebooks, Markdown,
commented code, shared `data/` and `outputs/`, and saved evidence. It copies no
medical data, eLCS dependencies or balancing methods.

## CPU and verification

This is a CPU-only PyTorch installation. `torch.cuda.is_available()` is expected
to be `False`, and `torch.version.cuda` should be `None`. That does not imply the
computer has no GPU. Hardware inspection found an NVIDIA RTX 2060 Max-Q with
6144 MiB VRAM and driver 512.78; GPU training will be assessed separately.

See the executed notebook and `outputs/metrics/environment_setup.json` for actual
versions, checks, hardware observations and setup errors. Environment metadata
is setup evidence, not dataset analysis or experimental results.
