"""Central configuration. Everything that defines the submitted model is frozen here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
ARTIFACT_DIR = ROOT / "artifacts"
EXP_DIR = ROOT / "experiments"
SUBMISSION_CSV = ROOT / "submission.csv"

ID_COL = "id"
TARGET = "target_popularity"
CLASSES = ["A", "B", "C", "D", "E"]

# ---------------------------------------------------------------- validation protocol
HOLDOUT_FRAC = 0.15          # confirmation holdout, stratified, never used for tuning
HOLDOUT_SEED = 2026
N_FOLDS = 5
CV_REPEAT_SEEDS = [11, 22, 33]   # development CV = 5 folds x 3 repeats on the dev set

# Two evaluation regimes (see README "Train/test shift"):
#   'train'    -> macro F1 on the training class mix
#   'balanced' -> macro F1 with rows re-weighted so every class has equal mass
#                 (adversarial validation shows the test set is ~class-balanced)
PRIMARY_REGIME = "balanced"

# ---------------------------------------------------------------- frozen final model
# Filled in from experiments/champion.json at freeze time; train.py reads the champion file.
CHAMPION_FILE = EXP_DIR / "champion.json"
