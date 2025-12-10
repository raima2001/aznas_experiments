#!/usr/bin/env bash
set -e

echo "==============================================="
echo " AZ-NAS QUICK TEST SETUP & RUN"
echo "==============================================="

# -------- 1. Create & activate virtual environment --------
if [ ! -d ".venv" ]; then
  echo "[+] Creating virtual environment (.venv)..."
  python -m venv .venv
else
  echo "[=] Virtual environment (.venv) already exists."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[+] Upgrading pip..."
pip install --upgrade pip

# -------- 2. Install PyTorch (try CUDA build, fallback to CPU) --------
echo "[+] Installing PyTorch (CUDA 12.1 wheel if possible)..."
if ! pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121; then
  echo "[!] CUDA wheel install failed, falling back to default PyTorch (CPU or system CUDA)..."
  pip install torch torchvision
fi

# -------- 3. Install remaining dependencies from requirements.txt --------
if [ -f "requirements.txt" ]; then
  echo "[+] Installing remaining Python packages from requirements.txt..."
  pip install -r requirements.txt
else
  echo "[!] requirements.txt not found – skipping. Make sure dependencies are installed."
fi

# -------- 4. Ensure directories exist --------
echo "[+] Ensuring data/ and checkpoints/ directories exist..."
mkdir -p data
mkdir -p checkpoints

echo
echo "============================================================"
echo " IMPORTANT: Teacher Weights Required"
echo "  - Place your CIFAR-100 teacher at:"
echo "        checkpoints/cifar100_teacher.pth"
echo "  - (Optional) For overfitting test speedup:"
echo "        checkpoints/cifar10_teacher_ablation.pth"
echo "============================================================"
echo

# Simple check for CIFAR-100 teacher presence
if [ ! -f "checkpoints/cifar100_teacher.pth" ]; then
  echo "[X] ERROR: checkpoints/cifar100_teacher.pth not found."
  echo "    Please copy your CIFAR-100 teacher weights into:"
  echo "        checkpoints/cifar100_teacher.pth"
  echo "    Then re-run this script."
  exit 1
fi

# -------- 5. Show GPU status --------
python - <<EOF
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
EOF

# -------- 6. Run all experiments in QUICK TEST mode --------
echo
echo "==============================================="
echo " RUNNING QUICK TEST PIPELINE"
echo "==============================================="

echo
echo "[1/3] Running 9_aznas_loss_c100.py --quick_test ..."
python 9_aznas_loss_c100.py --quick_test

echo
echo "[2/3] Running 9_aznas_component_ablation.py --quick_test ..."
python 9_aznas_component_ablation.py --quick_test

echo
echo "[3/3] Running 9_aznas_overfitting_test.py --quick_test ..."
python 9_aznas_overfitting_test.py --quick_test

echo
echo "==============================================="
echo " QUICK TEST PIPELINE COMPLETE"
echo " - Results are in checkpoints/*.json"
echo "==============================================="
