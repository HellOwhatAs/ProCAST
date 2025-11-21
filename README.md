# ProCAST
A projection-based framework that adjusts any forecasts to satisfy coupled aggregate constraints.

<picture>
  <source srcset="https://github.com/user-attachments/assets/797dcf64-20a0-4d80-b384-0cdb665fa861" media="(prefers-color-scheme: dark)">
  <img align="right" width="100%" src="https://github.com/user-attachments/assets/68d8c5d1-71d1-4f76-a7cf-e47b3be252f1"/>
</picture>

## Quick Start
> Requires driver supports CUDA 13.0+, or you will have to edit `pyproject.toml` manually.

<picture>
  <source srcset="https://github.com/user-attachments/assets/cc42440a-15ff-46ee-9ac4-746bd54e2d10" media="(prefers-color-scheme: dark)">
  <img align="right" width="60%" src="https://github.com/user-attachments/assets/340cd562-25db-4f08-af9f-21cde43bced9"/>
</picture>

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

## Main Results
<picture>
  <source srcset="https://github.com/user-attachments/assets/bb84f505-43ec-47e8-a176-fff22ede0f79" media="(prefers-color-scheme: dark)">
  <img src="https://github.com/user-attachments/assets/e8e346c4-f613-4de0-89b1-24dfdc1e74dc" width="100%"/>
</picture>

## Datasets
Included in this repository.
```
.
├─datasets
│  ├─e-commerce
│  │  ├─category.json
│  │  └─data.csv
│  └─RH
│     └─data.pkl
```

<picture>
  <source srcset="https://github.com/user-attachments/assets/243d4e35-2f62-4c1a-a175-3bc6417b22e7" media="(prefers-color-scheme: dark)">
  <img src="https://github.com/user-attachments/assets/25af5abe-feb8-4af5-a980-1b46b2433125" width="100%"/>
</picture>
