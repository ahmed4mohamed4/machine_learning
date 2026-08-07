# Machine Learning

A hands-on collection of notebooks, datasets, and mini-projects created while learning machine learning with Python.

## Learning path

| Step | Topic  |
| --- | --- |
| 1 | [Statistics](1_statistics/notebooks/main.ipynb) |
| 2 | [Python basics](2_python_basics/notebooks/main.ipynb) |

## Repository layout

```text
machine_learning/
├── 1_statistics/                         # Statistics foundations
├── 2_python_basics/                      # Core Python
└── requirements.txt                      # Python dependencies
└── README.md
```

Most learning modules use the same structure:

- `notebooks/` — guided examples and experiments.
- `data/` — datasets used by that module, when needed.when available.

## Getting started

1. Clone the repository and open its folder.
```bash
git clone https://github.com/ahmed4mohamed4/machine_learning.git
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install the dependencies.

```bash
pip install -r requirements.txt
```

## Requirements

See [requirements.txt](requirements.txt)
