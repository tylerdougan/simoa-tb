from pathlib import Path

import pandas as pd

INPUT_DIR = Path("..") / "input"
HDX_RESULTS_DIR = INPUT_DIR / "simoa"
ALERE_RESULTS_DIR = INPUT_DIR / "alere"
INPUT_EXT_DIR = INPUT_DIR / "external"
CLINICAL_MBV_FILES = [
    INPUT_EXT_DIR / s
    for s in [
        "FIND_BWH_Clinical_data_2019-04-08 original.xlsx",
        "blinded urine_BWH_WYSS_Clinical_data_LAM_levels_2019-12-19 original_received_doc.xlsx",
    ]
]
CLINICAL_TEST_FILE = INPUT_EXT_DIR / "WYSS_SAMPLES_Clinical_data_2021-05-10.xlsx"
META_ANALYSIS_FILE = INPUT_EXT_DIR / "meta-analysis2.xlsx"

PROCESSED_DIR = Path("..") / "processed"
OUTPUT_DIR = Path("..") / "output"


LOD_COL_FMT = pd.DataFrame(
    {
        "LOD_curve": {
            "488 Ag85B 182": 0.2,
            "750 LAM G3": 1.0,
            "700 LAM FIND28": 60.0,
            "647 LAM S4-20": 3.0,
        },
        "LOD_samples": {
            "488 Ag85B 182": 0.4,
            "750 LAM G3": 2.0,
            "700 LAM FIND28": 120.0,
            "647 LAM S4-20": 6.0,
        },
        "color": {
            "488 Ag85B 182": "#2E9AB6",
            "750 LAM G3": "#7D3571",
            "700 LAM FIND28": "#DD3C4A",
            "647 LAM S4-20": "#F2AB0D",
        },
        "fmt": {
            "488 Ag85B 182": "d",
            "750 LAM G3": "^",
            "700 LAM FIND28": "s",
            "647 LAM S4-20": "o",
        },
    }
)

FIND28_ISINGLE_MULTIPLIER = 0.6
