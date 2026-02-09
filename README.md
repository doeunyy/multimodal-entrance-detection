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
