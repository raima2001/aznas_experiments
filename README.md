

AZ-NAS Experiments (CIFAR-10 / CIFAR-100)**

This repository contains three GPU-accelerated experiments for **AZ-NAS** (Architectural Zero-shot Neural Architecture Search) implemented using **PyTorch**.
All scripts include two special modes:

* **`--fast_full`** → ~20 minute full experiment (recommended)
* **`--quick_test`** → 5–10 minute smoke test

The experiments require **teacher model checkpoints**, and the folder structure must follow the format below.

---

# 🌲 **Folder Structure**

```
aznas_experiments/
├── 9_aznas_loss_c100.py
├── 9_aznas_component_ablation.py
├── 9_aznas_overfitting_test.py
├── requirements.txt
├── README.md
├── checkpoints/
│   ├── cifar100_teacher.pth
│   └── cifar10_teacher_ablation.pth   # optional, auto-trained if missing
└── data/      # CIFAR files auto-downloaded here by torchvision
```

> **Important:**
> The teacher weights **must be inside `checkpoints/`**
> → `checkpoints/cifar100_teacher.pth`

---

# 📦 **1. Installation & Environment Setup**

## (A) Create & activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

---

## (B) Install **GPU-accelerated PyTorch**

⚠️ Installing PyTorch from `requirements.txt` does **not** install the GPU version.

Find the correct CUDA command for your GPU here:

👉 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Or choose one of these common installs:

### CUDA 12.1

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### CUDA 11.8

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### CPU-only (not recommended for these experiments)

```bash
pip install torch torchvision
```

---

## (C) Install remaining dependencies

Your `requirements.txt`:

```txt
torch
torchvision
numpy
Pillow
tqdm
```

Install with:

```bash
pip install -r requirements.txt
```

---

# 🧪 **2. Verify GPU Availability**

Run:

```bash
python - <<EOF
import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
EOF
```

Expected:

```
CUDA Available: True
GPU Name: NVIDIA A100-SXM4-40GB
```

If CUDA is unavailable, reinstall PyTorch using the correct GPU wheel.

---

# 🎒 **3. Teacher Checkpoints (Required Files)**

The scripts expect:

### CIFAR-100 teacher

Required for **ALL scripts**:

```
checkpoints/cifar100_teacher.pth
```

Format may be:

```python
{"state_dict": <weights>, "val_acc": <float>}
```

or a raw `state_dict` → both work.

### CIFAR-10 teacher

Used only in **overfitting test**, optional:

```
checkpoints/cifar10_teacher_ablation.pth
```

If missing → the script trains it automatically.

---

# 🚀 **4. How to Run Each Script**

Run everything from inside the root folder:

```
aznas_experiments/
```

---

## ✅ **A. 9_aznas_loss_c100.py**

**Purpose:**
Compare pruning baselines on CIFAR-100:

* Random
* L2-norm
* KD-only encoder
* Full AZ-NAS (E+P+C)

**Fast full (~20 min):**

```bash
python 9_aznas_loss_c100.py --fast_full
```

**Quick test (~5–10 min):**

```bash
python 9_aznas_loss_c100.py --quick_test
```

**Outputs:**

* `checkpoints/cifar100_teacher.pth`
* `checkpoints/aznas_loss_comparison_results.json`

---

## ✅ **B. 9_aznas_component_ablation.py**

**Purpose:**
Ablate AZ-NAS loss components on CIFAR-100:

* Expressivity only
* Progressivity only
* Complexity only
* Expr + Prog
* Full (E+P+C)

**Fast full (~20 min):**

```bash
python 9_aznas_component_ablation.py --fast_full
```

**Quick test (~5–10 min):**

```bash
python 9_aznas_component_ablation.py --quick_test
```

**Outputs:**

* `checkpoints/aznas_component_ablation_results.json`

---

## ✅ **C. 9_aznas_overfitting_test.py**

**Purpose:**

1. Train AZ-NAS policy on CIFAR-10
2. Apply this *old policy* to CIFAR-100 (transfer test)
3. Train new policy directly on CIFAR-100
4. Compare performance:

   * If **new >> old** → overfitting
   * If **new ≈ old** → generalization

**Fast full (~20 min):**

```bash
python 9_aznas_overfitting_test.py --fast_full
```

**Quick test (~5–10 min):**

```bash
python 9_aznas_overfitting_test.py --quick_test
```

**Outputs:**

* CIFAR-10 teacher (auto-trained if absent)
  → `checkpoints/cifar10_teacher_ablation.pth`
* `checkpoints/aznas_overfitting_test_results.json`

---

# 📘 **5. Recommended Execution Order**

If everything is being run fresh:

```bash
# 1. Train CIFAR-100 teacher + loss baselines
python 9_aznas_loss_c100.py --fast_full

# 2. Run AZ-NAS component ablation on CIFAR-100
python 9_aznas_component_ablation.py --fast_full

# 3. Run overfitting test: CIFAR-10 → CIFAR-100
python 9_aznas_overfitting_test.py --fast_full
```

For quick sanity checks:

```bash
python 9_aznas_loss_c100.py --quick_test
python 9_aznas_component_ablation.py --quick_test
python 9_aznas_overfitting_test.py --quick_test
```

## Running the Automated Setup Scripts

After placing the scripts in the project folder, make them executable:

```bash
chmod +x setup_quick_test.sh setup_fast_full.sh
````

### Run the **Quick Test** version (~5–10 minutes):

```bash
./setup_quick_test.sh
```

### Run the **Fast-Full** version (~20 minutes):

```bash
./setup_fast_full.sh
```

