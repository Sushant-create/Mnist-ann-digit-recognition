"""
================================================================================
 run.py  -  Direct launch file
--------------------------------------------------------------------------------
 One-command entry point for the whole project:
   1. Verifies the dataset is present in 'data/'.
   2. Installs dependencies from requirements.txt (if missing).
   3. Runs main.py (Tasks 1-4: load, preprocess, train, evaluate).
   4. Runs test_samples.py (visual sanity check on sample digits).

 Author  : Sushant Tiwari
 License : All Rights Reserved (see LICENSE)

 USAGE
 -----
     python run.py
================================================================================
"""

import os
import subprocess
import sys

DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "mnist_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "mnist_test.csv")


def check_dataset():
    if not (os.path.exists(TRAIN_CSV) or os.path.exists(TEST_CSV)):
        print("=" * 78)
        print("ERROR: Dataset not found.")
        print("Download 'mnist_train.csv' and 'mnist_test.csv' from:")
        print("  https://www.kaggle.com/datasets/oddrationale/mnist-in-csv")
        print(f"and place them inside the '{DATA_DIR}/' folder before running again.")
        print("=" * 78)
        sys.exit(1)


def install_requirements():
    print("Checking/installing dependencies from requirements.txt ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
        check=True,
    )


def run_step(script_name):
    print(f"\n{'='*78}\nRunning {script_name} ...\n{'='*78}")
    subprocess.run([sys.executable, script_name], check=True)


def main():
    check_dataset()
    install_requirements()
    run_step("main.py")
    run_step("test_samples.py")
    print("\nAll done! Check the 'outputs/' folder for results and 'saved_model/' "
          "for the trained model.")


if __name__ == "__main__":
    main()
