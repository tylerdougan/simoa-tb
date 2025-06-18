import numpy as np
import pandas as pd
import waltlabtools as wlt

from config import FIND28_ISINGLE_MULTIPLIER
from utils import get_isingle_aeb

q_raw = pd.read_csv("file.csv")
nonzero_calibrators = (q_raw["Sample Type"] == "Calibrator") & (
    q_raw["Replicate Conc."] > 0
)

cal_curve_cols = {
    7: "Ag85B only",
    9: "LAM only",
    11: "Both",
}

q_raw["Curve"] = "Both"
q_raw.loc[nonzero_calibrators, "Curve"] = (
    q_raw.loc[nonzero_calibrators, "Location"]
    .map(lambda s: int(s.split()[-1][1:]))
    .map(cal_curve_cols)
)

# READ FILES
q = wlt.HDX(raw=q_raw, assay_defining_cols=["Curve", "Plex"])
plexes = sorted(q.raw.Plex.unique())

# CALCULATE AEBS AND CONCENTRATIONS
q.raw = get_isingle_aeb(
    q.raw, {"700 LAM FIND28": FIND28_ISINGLE_MULTIPLIER}, max_aeb=np.nan
)
for plex in sorted(q.raw.Plex.unique()):
    analyte = plex.split()[1]
    df = q.raw[
        (q.raw["Plex"] == plex)
        & (q.raw["Sample Type"] == "Calibrator")
        & (q.raw["Replicate AEB"].notna())
        & (q.raw["Curve"].isin(["Both", f"{analyte} only"]))
    ]
    q.cal_curves["Both", plex] = wlt.CalCurve(model="linear").fit(
        X=df["Replicate Conc."], y=df["Replicate AEB"]
    )


# Concentrations
def get_conc(row):
    return q.cal_curves["Both", row["Plex"]].conc(row["Replicate AEB"])


q.raw.loc[q.raw["Sample Type"] == "Specimen", "Replicate Conc."] = q.raw[
    q.raw["Sample Type"] == "Specimen"
].apply(get_conc, axis=1)

Xs = q.raw[q.raw["Sample Type"] == "Specimen"].pivot_table(
    columns="Plex",
    index="Sample Barcode",
    values="Replicate Conc.",
    aggfunc=["median", "min", "max", "count"],
)


def get_dilution_df(plex: str, aggfunc: str) -> pd.DataFrame:
    df = (
        Xs[Xs[("meta", "Purpose")] == "Dilution Linearity"].pivot_table(
            index=[("meta", "Dilution Factor")],
            columns=[("meta", "Specimen Barcode")],
            values=[(aggfunc, plex)],
        )
    )[(aggfunc, plex)]
    inf_fold = df.loc[np.inf, "DilutionLinearity"]
    df.loc[np.inf] = inf_fold
    return df.dropna(thresh=2, axis=1)


diluted_conc = {plex: get_dilution_df(plex, "median") for plex in plexes}
