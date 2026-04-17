# Multimodal Entrance Detection

**CS566 Deep Learning Project (Spring 2026)**

This repository contains the codebase for identifying building entrance locations by fusing visual signals (aerial/street-level imagery) with behavioral patterns (human trajectory data).

## 🚀 Getting Started

### 1. Install Miniconda
We use **Miniconda** to manage our Python environment to ensure everyone uses the same dependencies.

**If you don't have Conda installed:**
1. Download the installer for your OS here: [Miniconda Download](https://docs.conda.io/en/latest/miniconda.html)
2. Run the installer and restart your terminal.

### 2. Create and Activate Environment
Run the following commands in your terminal to set up the environment:

```bash
# Create a new environment named 'entrance-det' with Python 3.10
conda create -n entrance-det python=3.10 -y

# Activate the environment
conda activate entrance-det
```

### 3. Install Dependencies
We use a `requirements.txt` file to manage packages.

**Step A: Install PyTorch (Specific to your Hardware)** Before installing the rest, install the correct version of PyTorch for your system (CUDA for Nvidia GPUs, MPS for Mac M1/M2, or CPU).

- Visit [Start Locally | PyTorch](https://pytorch.org/get-started/locally/) to get your specific command.
- *Example (Linux with CUDA 13.0):*
    ```bash
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    ```

**Step B: Install Project Dependencies**
```bash
pip install -r requirements.txt
```

### 4. Setup Pre-Commit Hooks (Crucial!)
We use pre-commit to automatically format code (Black) and sort imports (Isort) before you can commit. This keeps our code looking clean and prevents merge conflicts.
```bash
# Install the git hooks
pre-commit install
```
*Now, every time you run `git commit`, it will check your code. If it fails, it will auto-fix the formatting. Just add the changes and commit again.*

---

## 🛠 Repository Structure
```plaintext
entrance-detection/
├── data/                   # IGNORED by git. Put raw data here.
│   ├── raw/                # OSM, WorldTrace, Mapillary downloads
│   └── processed/          # Tensors/Intermediate files
├── notebooks/              # Jupyter Notebooks for EDA (naming: 01_desc.ipynb)
├── src/                    # Main source code
│   ├── baselines/          # Random Forest & simple models (Team A)
│   ├── method_v1/          # ViT + LoRA implementation (Team B)
│   ├── data_loader.py      # Data fetching scripts
│   └── utils.py            # Helper functions
├── scripts/                # Shell scripts for training/eval
├── .gitignore              # Files to exclude from git
├── .pre-commit-config.yaml # Formatting rules
├── README.md               # This file
├── config.yaml             # Configurations
└── requirements.txt        # Python dependencies
```

## 🤝 Collaboration Workflow
To avoid chaos with 7 people, please follow this **Feature Branch Workflow**.

### 1. Start a New Task
Never push directly to `main`. Always create a branch.
```bash
# 1. Switch to main and get the latest changes
git checkout main
git pull origin main

# 2. Create a new branch for your specific task
# Naming convention: type/short-description
# Types: feat (feature), fix (bug fix), docs (documentation), exp (experiment)
git checkout -b feat/vit-encoder
```

### 2. Work and Commit
Work on your code. When you are ready to save:
```
git add .
git commit -m "Add ViT encoder class with patch embedding"
```
*Note: If `pre-commit` fails, it has likely auto-formatted your file. Simply run `git add .` and `git commit ...` again.*

### 3. Keep Your Branch Updated (Rebase vs. Merge)
We use **Rebase**. This keeps our history clean and linear. If others have pushed to `main` while you were working, you need to sync:
```bash
# 1. Fetch latest updates from the remote
git fetch origin

# 2. Rebase your branch on top of origin/main
git rebase origin/main
```
*If there are conflicts, Git will pause. Fix the files, then run `git add <file>` and `git rebase --continue`.*

### 4. Open a Pull Request (PR)
1. Push your branch:
    ```Bash
    git push -u origin feat/vit-encoder
    ```
2. Go to GitHub and click **"Compare & pull request"**.

3. **Link your Issue**: In the description, type `Closes #12` (where 12 is your issue number). This moves the card on the Project Board automatically.

4. **Assign a Reviewer**: Pick a teammate to review your code.

### 5. Merge
Once approved, use the **"Squash and Merge"** button on GitHub. This combines all your small commits into one clean commit on the main branch.

---

## 📊 Data Management
Do not commit large data files! Upload them to [this Google Drive](https://drive.google.com/drive/folders/1wuoTEfzsUUk7XEbKTxojzOkrX8KvrQMd?usp=sharing) under `Datasets/`.

The `data/` folder is in `.gitignore`.

Download datasets locally and place them in `data/raw/`.

If you generate a new necessary config file or small CSV, you may commit it, but avoid binary files >10MB.

### Processed Dataset Schema

The preprocessing pipeline writes data under `config.yaml -> processed_data` with this structure:

```plaintext
{processed_data}/
├── splits.json
├── {city}/
│   ├── metadata.gpkg                 # layer: Filtered_Buildings
│   ├── aerial/
│   │   └── building_{idx}.tif
│   ├── streetview/
│   │   └── building_{idx}/
│   │       └── {image_id}.jpg
│   ├── streetview_metadata/
│   │   └── building_{idx}.csv
│   └── gps/
│       └── building_{idx}.csv
```

Where `{idx}` is the row index in `Filtered_Buildings` for that city.

#### `metadata.gpkg` (`Filtered_Buildings` layer)

This layer starts from OSM building attributes (including `geometry`, `osm_id`, etc.) and is augmented by preprocessing scripts.

Columns added by each step:

1. **`1_filter_buildings_with_entrances.py`**
     - `entrance_geometries`: list of entrance point geometries for the building.

2. **`2_add_aerial_images.py`**
     - `aerial_image_path`: relative path like `aerial/building_{idx}.tif`, or empty string if missing.

3. **`3_add_streetview_images.py`**
     - `streetview_image_dir`: relative path like `streetview/building_{idx}`, or empty string.
     - `n_streetview_images`: integer count of copied street-view images.

4. **`4_add_gps_traces.py`**
     - `n_traj_within_50m`: integer trajectory count from WorldTrace stats.
     - `n_traj_inside`: integer trajectory count from WorldTrace stats.
     - `gps_trace_path`: relative path like `gps/building_{idx}.csv`, or empty string.

5. **`6_add_candidates.py`**
     - `candidates`: dictionary-like object with keys:
         - `positive`: list of true entrance Points.
         - `negative`: list of sampled negative Points on the building boundary.
         - `labels`: list of ints (`1` for positive, `0` for negative).
         - `all_points`: concatenation of `positive + negative`.

#### `splits.json`

Created by `5_split_train_val_test.py` with per-city mapping:

```json
{
    "nyc": {
        "building_0": "train",
        "building_1": "val",
        "building_2": "test"
    },
    "la": {
        "building_0": "train"
    }
}
```

Split keys use `building_{idx}`, where `{idx}` is the same index used by the per-city `metadata.gpkg` layer.

#### Modality File Schemas

- **Aerial (`aerial/building_{idx}.tif`)**
    - GeoTIFF cropped around building footprint with ~50m buffer.
    - Read by loader as `np.ndarray` with shape `(C, H, W)`.

- **Streetview (`streetview/building_{idx}/*.jpg`)**
    - Zero or more JPEG files copied from Mapillary top-k results.
    - Read by loader as a list of PIL RGB images.

- **GPS (`gps/building_{idx}.csv`)**
    - Subset of nearby trajectory points from WorldTrace trajectory files.
    - Includes original trajectory columns plus inserted `traj_id` column.
    - Read by loader as a pandas DataFrame.

### How to Obtain Data
You can obtain the dataset in two ways:

1. **[Recommended] Download processed data from [Google Drive](https://drive.google.com/drive/folders/1pVvP5Uq2jgdho4e2VfsFThKqo14N91ha?usp=drive_link)**

2. **Run code to process data from their raw form**
    At root of this repository, using the `entrance-det` virtual environment set up earlier:
    1. Update `config.yaml` with the correct `raw_data` path, and the desired location to store the processed data: `processed_data`
    2. Activate the conda environment you created `conda activate entrance-det`
    3. Grant preprocessing script execute-permission: `chmod +x scripts/preprocess.sh`
    4. Run the pre-processing script `./scripts/preprocess.sh` (runs steps 1-6)

---

## 📚 Using the Data Loader and Metrics

### Data Loader (`src/data_loader.py`)

The dataset is structured as a **binary classification problem**: for each building, you will classify candidate entrance points as either true entrances (label=1) or negative samples (label=0).

#### Quick Start

```python
from src.data_loader import BuildingEntranceDataset, collate_fn
from torch.utils.data import DataLoader

# Load the training dataset for NYC with all modalities
train_dataset = BuildingEntranceDataset(
    city="nyc",
    split="train",
    modalities=["streetview", "aerial", "gps"],
    config_path="config.yaml"
)

# Create a DataLoader with the custom collate function
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    collate_fn=collate_fn,
    shuffle=True
)

# Example: iterate through batches
for batch in train_loader:
    building_ids = batch["building_id"]        # List of (city, building_idx) tuples
    geometries = batch["geometry"]               # List of building geometries
    candidates = batch["candidates"]             # List of candidate coordinates
    labels = batch["labels"]                     # List of binary labels (1 or 0)

    # Modality-specific data (if requested)
    streetview_imgs = batch["streetview_images"]  # List of PIL Image lists
    aerial_imgs = batch["aerial_image"]           # List of numpy arrays (C, H, W)
    gps_traces = batch["gps_traces"]              # List of DataFrames

    # Your model training code here
    pass
```

#### Dataset Output Format

Each item from `__getitem__` returns a dictionary:

```python
{
    "building_id": (city, building_idx),           # Tuple identifying the building
    "geometry": Polygon,                            # Building footprint (Shapely geometry)
    "candidates": [(lon, lat), (lon, lat), ...],  # List of candidate entrance coordinates
    "labels": [1, 0, 1, 0, ...],                  # Binary labels for each candidate (1=true entrance, 0=negative)
    "streetview_images": [PIL.Image, ...],        # Optional: list of street-level images
    "aerial_image": np.ndarray (C, H, W),         # Optional: aerial/satellite image
    "gps_traces": pd.DataFrame,                   # Optional: GPS trajectory points
}
```

#### Key Features

- **Variable-length sequences**: Buildings have different numbers of candidates (typically ~32 per building). Use the custom `collate_fn` to handle this.
- **Multimodal flexibility**: Load only the modalities you need (e.g., aerial + GPS only).
- **Train/val/test splits**: Automatically handled via `splits.json` created during preprocessing.
- **Multiple cities**: Load data from any city in the processed dataset.

#### Usage Examples

```python
# Example 1: Load only some modalities
dataset = BuildingEntranceDataset(
    city="la",
    split="val",
    modalities=["aerial", "gps"]  # Skip streetview
)

# Example 2: Get metadata about a building
for i in range(len(dataset)):
    sample = dataset[i]
    n_candidates = len(sample["candidates"])
    n_positive = sum(sample["labels"])
    print(f"Building {i}: {n_candidates} candidates ({n_positive} positive)")

# Example 3: Access full metadata
metadata = dataset.get_building_metadata(0)
print(metadata.columns)  # See all available fields
```

---

### Metrics and Evaluation (`src/metrics.py`)

After training your model, use this module to evaluate predictions against ground truth labels.

#### Evaluation Pipeline

```python
from src.metrics import evaluate_predictions
import yaml

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Evaluate predictions
results = evaluate_predictions(
    processed_path=config["processed_data"],
    predictions_file="path/to/your/predictions.csv",
    cities=config["cities"],
    output_file="path/to/evaluation_results.json"
)
```

#### Prediction File Format

Your model predictions should be saved as a CSV file with columns:
```
city,building_idx,candidate_idx,predicted_prob
nyc,0,0,0.95
nyc,0,1,0.12
nyc,0,2,0.87
...
```

Where:
- `city`: City name (string)
- `building_idx`: Building index (integer)
- `candidate_idx`: Candidate index within the building (0 to n_candidates-1)
- `predicted_prob`: Predicted probability that this candidate is a true entrance (0.0 to 1.0)

#### Computed Metrics

All candidates from every building in every city are pooled into a single
`(y_true, y_pred_proba)` set, and metrics are computed once on that pool:

- **Accuracy, Precision, Recall, F1**: reported at the threshold that maximizes
  F1 over a 0.00–1.00 sweep (step 0.01). If no threshold yields F1 > 0, the
  values fall back to the threshold=0.5 baseline.
- **AUROC, AUPR**: threshold-free ranking metrics. Both require y_true to
  contain both classes; otherwise they are reported as `null`.

Per-building entries retain only `n_candidates`, `n_positive`, `n_negative`.

#### Example Usage

```python
# After your model generates predictions for all test buildings:

# Save predictions to CSV
import pandas as pd
predictions_data = []
for batch in test_loader:
    for building_id, candidates, probs in zip(
        batch["building_id"],
        batch["candidates"],
        batch["predicted_probs"]  # Your model output
    ):
        city, building_idx = building_id
        for candidate_idx, prob in enumerate(probs):
            predictions_data.append({
                "city": city,
                "building_idx": building_idx,
                "candidate_idx": candidate_idx,
                "predicted_prob": float(prob)
            })

predictions_df = pd.DataFrame(predictions_data)
predictions_df.to_csv("predictions.csv", index=False)

# Evaluate
from src.metrics import evaluate_predictions
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

results = evaluate_predictions(
    processed_path=config["processed_data"],
    predictions_file="predictions.csv",
    cities=config["cities"],
    output_file="evaluation_results.json"
)
```

#### Output

The function generates a comprehensive JSON report with:
- Per-city results
- Per-building results
- Aggregated statistics

Example output structure:
```json
{
  "aggregated": {
    "total_buildings": 1000,
    "total_candidates": 32000,
    "total_positive": 2500,
    "total_negative": 29500,
    "pooled_metrics": {
      "accuracy": 0.92,
      "precision": 0.78,
      "recall": 0.81,
      "f1": 0.79,
      "auroc": 0.94,
      "aupr": 0.82
    }
  }
}
```

---

## 🔧 Development Workflow

### Step 1: Load Data
```python
from src.data_loader import BuildingEntranceDataset, collate_fn

train_data = BuildingEntranceDataset(city="nyc", split="train")
```

### Step 2: Build Your Model
Your model should take multimodal inputs and output a probability score for each candidate being a true entrance.

### Step 3: Generate Predictions
Make predictions on test/val sets and save in the required CSV format.

### Step 4: Evaluate
```python
from src.metrics import evaluate_predictions

results = evaluate_predictions(
    processed_path=config["processed_data"],
    predictions_file="predictions.csv",
    cities=config["cities"],
    output_file="results.json"
)
```

This workflow is designed to be followed by all method teams (baselines, ViT-based, etc.).
