"""Configuration of paths, cohort definitions, features, models and evaluation settings."""
from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PRIVATE_CONFIG = Path(os.environ.get("GBM_PRIVATE_CONFIG",
                                     BASE_DIR / "private_config.json"))
_PRIVATE = (json.loads(PRIVATE_CONFIG.read_text(encoding="utf-8"))
            if PRIVATE_CONFIG.exists() else {})


def _path(key: str, default: str) -> Path:
    path = Path(_PRIVATE.get("paths", {}).get(key) or default)
    return path if path.is_absolute() else BASE_DIR / path


DATA_CLINICAL = _path("data_clinical", "Database_pazienti_clinico.ods")
DATA_RADIOMIC = _path("data_radiomic", "radiomiche_quibim.csv")
EXTERNAL_CLINICAL = _path("external_clinical", "File_GBM_Gemelli.xlsx")
EXTERNAL_RADIOMIC = _path("external_radiomic", "radiomiche_roma.csv")

RESULTS_DIR = _path("results_dir", "results_cox")
FIGURES_DIR = _path("figures_dir", "figures_cox")
CACHE_DIR = _path("cache_dir", "cache_cox")
CKPT_DIR = _path("checkpoints_dir", "checkpoints_cox")
LOGS_DIR = _path("logs_dir", "logs")
MODEL_PATH = _path("frozen_model", "model/frozen_model.json")

for _d in (RESULTS_DIR, FIGURES_DIR, CACHE_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATA_LOCK = "2026-09-10"

TARGET_FIXES: dict[int, int] = {int(k): int(v) for k, v in
                                _PRIVATE.get("target_fixes", {}).items()}
EXCLUDE_IDS: tuple[int, ...] = tuple(int(i) for i in _PRIVATE.get("exclude_ids", []))
LOST_TO_FOLLOWUP_IDS: tuple[int, ...] = tuple(
    int(i) for i in _PRIVATE.get("lost_to_followup_ids", []))

REQUIRE_IDH_WILDTYPE = True

ID_COL = "ID_PAZIENTE"
TARGET_COL = "target"
DURATION_COL = "duration"
EVENT_COL = "event"

CLINICAL_FEATURES = [
    "Eta",
    "GENERE",
    "RESEZIONE",
    "Ki67",
    "MGMT_METILAZIONE",
    "p53_ESPRESSIONE",
]

BINARY_CLINICAL = ["GENERE", "RESEZIONE", "p53_ESPRESSIONE"]


MGMT_BINARY_THRESHOLD: float | None = 10.0
if MGMT_BINARY_THRESHOLD is not None:
    BINARY_CLINICAL = BINARY_CLINICAL + ["MGMT_METILAZIONE"]

DEEP_PATTERN = r"TALAMO|NUCLEO|CC|TRIGONO|\+D|\bD-|MESIAL"

P53_ABERRANT_CATEGORIES = ("assente", "rare cellule", "intensa", "diffusa")


SHAPE_FEATURES = [
    "Elongation",
    "Major_Axis_in_3D_Length",
    "Minor_Axis_in_3D_Length",
    "Sphericity",
    "Flatness",
    "Least_Axis_in_3D_Length",
    "Maximum_2D_diameter_column",
    "Maximum_2D_diameter_row",
    "Maximum_2D_diameter_slice",
    "Maximum_3D_Diameter_of_a_Mesh",
    "Volume_of_Mesh",
    "Surface_Area_of_Mesh",
    "Surface_to_Volume_Ratio",
    "Volume_from_Voxel_Summation",
]

FIRSTORDER_PREFIXES = ("IH_", "FO_")


def feature_class(name: str) -> str:
    """Classify a radiomic column by its suffix after the region prefix."""
    suffix = name.split("_", 1)[1] if "_" in name else name
    if suffix in SHAPE_FEATURES:
        return "shape"
    if suffix.startswith(FIRSTORDER_PREFIXES):
        return "firstorder"
    return "texture"


REGIONS = {"enh": "Enhancing tumour", "nec": "Necrotic core", "ede": "Peritumoral edema"}

ADD_VOLUME_RATIOS = True
VOLUME_FEATURE = "Volume_of_Mesh"

SEED = 42
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 5
N_REPEATS = 10

L1_RATIOS = [0.5, 0.7, 0.9, 1.0]
N_ALPHAS = 50
ALPHA_MIN_RATIO = 0.01

SCANNER_CORRECTION = "combat"
FIELD_ID_CUTOFF = _PRIVATE.get("field_id_cutoff")
SCANNER_FDR = 0.05

CORRELATION_THRESHOLD = 0.90
VARIANCE_PERCENTILE = 5

FEATURE_SPACES = {
    "clinical": {"clinical": True, "regions": [], "classes": None},
    "radiomic_shape": {"clinical": False, "regions": ["enh", "nec", "ede"],
                       "classes": ["shape"]},
    "mixed_shape": {"clinical": True, "regions": ["enh", "nec", "ede"],
                    "classes": ["shape"]},
}

SECONDARY_SPACES = {
    "enhancing_only": {"clinical": False, "regions": ["enh"], "classes": None},
    "mixed_enhancing": {"clinical": True, "regions": ["enh"], "classes": None},
    "necrosis_only": {"clinical": False, "regions": ["nec"], "classes": None},
    "edema_only": {"clinical": False, "regions": ["ede"], "classes": None},
    "radiomic_all": {"clinical": False, "regions": ["enh", "nec", "ede"],
                     "classes": None},
    "mixed_all": {"clinical": True, "regions": ["enh", "nec", "ede"],
                  "classes": None},
    "radiomic_shape_fo": {"clinical": False, "regions": ["enh", "nec", "ede"],
                          "classes": ["shape", "firstorder"]},
}

RUN_SECONDARY_SPACES = True

N_BOOTSTRAP_OPTIMISM = 200
N_BOOTSTRAP_CI = 2000
N_PERMUTATIONS = 1000

TIME_HORIZONS = [6.0, 12.0, 24.0]

N_JOBS = 10
USE_CACHE = True

RUN_SHAP = True

FIG_DPI = 300
FIG_FORMAT = "png"

SPACE_LABELS = {
    "clinical": "Clinical",
    "radiomic_shape": "Radiomic",
    "mixed_shape": "Mixed",
    "radiomic_shape_fo": "Radiomic (shape + first-order)",
    "radiomic_all": "Radiomic (all features)",
    "mixed_all": "Mixed (all features)",
    "enhancing_only": "Enhancing only",
    "mixed_enhancing": "Mixed (enhancing, all features)",
    "necrosis_only": "Necrosis only",
    "edema_only": "Edema only",
}

PRETTY_CLINICAL = {
    "Eta": "Age",
    "GENERE": "Female sex",
    "RESEZIONE": "Gross total resection",
    "Ki67": "Ki67",
    "MGMT_METILAZIONE": "MGMT methylation (per %)",
    "p53_ESPRESSIONE": "p53 wild-type pattern",
    "DEEP": "Deep structure involvement",
    "rad_score": "Radiomic score",
}


def describe() -> str:
    """Configuration summary, written to the log at the start of every run."""
    return (
        f"Base directory     : {BASE_DIR}\n"
        f"Data lock          : {DATA_LOCK}\n"
        f"Endpoint           : time to death (Cox)\n"
        f"Regions            : {', '.join(REGIONS)}\n"
        f"Volume ratios      : {ADD_VOLUME_RATIOS}\n"
        f"CV                 : {N_REPEATS} x {N_SPLITS_OUTER}-fold "
        f"({N_REPEATS * N_SPLITS_OUTER} folds)\n"
        f"l1_ratio grid      : {L1_RATIOS}\n"
        f"Permutations       : {N_PERMUTATIONS}\n"
        f"Optimism bootstrap : {N_BOOTSTRAP_OPTIMISM}\n"
        f"Parallel jobs      : {N_JOBS}\n"
        f"Seed               : {SEED}"
    )


def cache_size_mb() -> float:
    if not CACHE_DIR.exists():
        return 0.0
    return sum(p.stat().st_size for p in CACHE_DIR.rglob("*") if p.is_file()) / 1024 / 1024


def clear_cache() -> None:
    """Empty the transformer cache."""
    import shutil

    if CACHE_DIR.exists():
        size = cache_size_mb()
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Cache cleared ({size:.0f} MB)")
