# ProCAST
A projection-based framework that adjusts any forecasts to satisfy coupled aggregate constraints.

## Quick Start
> Requires driver supports CUDA 13.0+, or you will have to edit `pyproject.toml` manually.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Update the project's environment.
   ```
   uv sync
   ```
3. Main Result (without BU, MinT):
   ```
   uv run nf_main.py
   ```
4. Main Result (BU, MinT):
   ```
   uv run hierarchical_main.py
   ```

## Datasets
Included in this repository.
```
.
├─datasets
│  ├─e-commerce
|  |  ├─category.json
|  |  └─data.csv
│  └─RH
|     └─data.pkl
```
