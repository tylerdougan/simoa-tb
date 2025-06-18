import functools
import re
import warnings
from collections import namedtuple

import matplotlib
import numpy as np
import optuna
import pandas as pd
import scipy.sparse
from sklearn.base import ClassifierMixin
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.model_selection._split import GroupsConsumerMixin, _BaseKFold
from sklearn.utils import indexable
from tqdm import tqdm


# Monkey patch because pyGAM relies on the deprecated scipy.sparse.spmatrix.A
def to_array(self):
    return self.toarray()


scipy.sparse.spmatrix.A = property(to_array)
import pygam
from pygam.utils import OptimizationError

optuna.logging.set_verbosity(optuna.logging.WARNING)
matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def weight_digital(fon_):
    return (1 - fon_) ** 2 / ((1 - fon_) ** 2 + fon_**6.75)


def get_isingle_aeb(df, isingle_plexes=None, max_aeb=30, max_fon=0.5):
    # Isingle when the lowest fraction on was > 0.2
    if isingle_plexes is not None:
        for plex, isingle_multiplier in isingle_plexes.items():
            is_plex = df["Plex"] == plex
            isingle = (
                df[is_plex & (df["Fraction On"] < max_fon)]
                .groupby("Batch Name")
                .apply(
                    lambda df: (
                        df["Fraction On"] * df["Ibead"] / df["Digital AEB"]
                    ).mean()
                    * isingle_multiplier,
                    include_groups=False,
                )
            )
            df.loc[is_plex, "Isingle"] = df.loc[is_plex, "Batch Name"].map(isingle)

            # Analog AEB when the lowest fraction on was > 0.2
            df.loc[is_plex, "Analog AEB"] = (
                df.loc[is_plex, "Fraction On"]
                * df.loc[is_plex, "Ibead"]
                / df.loc[is_plex, "Isingle"]
            )

    # Replicate AEB is a weighted combination of digital and analog AEB
    # Anal. Chem. 2023, 95, 22, 8613–8620 (2023)
    # https://doi.org/10.1021/acs.analchem.3c00918
    w = weight_digital(df["Fraction On"])
    df["Replicate AEB"] = w * df["Digital AEB"] + (1 - w) * df["Analog AEB"]
    # Fill high AEBs with MAX_AEB
    too_much_fluorescence = (
        df["Replicate AEB"].isna()
        & df["Errors"].str.contains("TooMuchFluorescenceInResorufinChannelException")
        & (df["Sample Type"] == "Specimen")
    )
    df.loc[too_much_fluorescence, ["Replicate AEB"]] = max_aeb
    return df


BarcodeBase = namedtuple(
    "BarcodeBase",
    ("prefix", "study", "site", "patient", "specimen", "aliquot"),
    defaults=("", "00", "00", "0000", "U", "00"),
)


class Barcode2(BarcodeBase):
    __slots__ = ()

    def __new__(cls, x):
        x = cls._process_x(x)
        s0, s1, s2, s3, s4 = cls._find_splits(x)
        return super().__new__(cls, *cls._parse(x))

    @staticmethod
    def _process_x(x: str | int | float) -> str:
        if isinstance(x, float):
            x = int(x)
        if isinstance(x, int):
            x = str(x)
        if not isinstance(x, str):
            raise TypeError(f"Expected str or numeric, got {type(x)}: {x}")
        return x.replace(" ", "-").strip()

    @staticmethod
    def _first_digit(x: str):
        s = re.search(r"\d", x)
        if s is not None:
            return s.start()
        else:
            raise ValueError(f"Invalid barcode: {x} has no digits")

    @classmethod
    def _find_splits(cls, x: str) -> tuple[int, int, int, int, int, int]:
        len_gt_12 = int(len(x) > 12)
        s0 = cls._first_digit(x)
        leading_zero = int(x[s0] == "0")
        s1 = s0 + 1 + leading_zero + len_gt_12
        s2 = s1 + 2
        s3 = s2 + 4
        s4 = -2 - len_gt_12
        return (s0, s1, s2, s3, s4)

    @classmethod
    def _parse(cls, x: str):
        s1a = cls._first_digit(x)
        s0b = s1a - 1 if (s1a > 0) and (x[s1a - 1] == "-") else s1a
        s1b = x.find("-", s1a, s1a + 4)
        if s1b > -1:  # there is a delimiter between study and site
            s2a = s1b + 1
        elif x[s1a : s1a + 3] == "026":  # 3-digit study ID
            s1b = s2a = s1a + 3
        elif x[s1a] == "0":  # 2-digit study ID
            s1b = s2a = s1a + 2
        else:  # 1-digit study ID (leading zero removed in int conversion)
            s1b = s2a = s1a + 1
        s2b = s2a + 2  # site ID is always 2 digits
        s3a = s2b + (x[s2b] == "-")  # beginning of patient ID
        s3b = s3a + 4
        if "-" in x[s3a:s3b]:
            x = x[:s3a] + x[s3a:s3b].replace("-", "") + x[s3b:]
        s4a = s3b + (x[s3b] == "-")  # beginning of specimen type ID
        s4b = x.find("-", s4a)
        if s4b > -1:
            s5a = s4b + 1
        else:
            s4b = s5a = len(x) - 2
        return (x[:s0b], x[s1a:s1b], x[s2a:s2b], x[s3a:s3b], x[s4a:s4b], x[s5a:])

    def __str__(self) -> str:
        return " ".join(self).strip()

    def __repr__(self) -> str:
        return f"Barcode2('{str(self)}')"

    def standard_form(self) -> str:
        standardized = (
            self.study[-2:].rjust(2, "0"),
            self.site[-2:],
            self.patient[-4:],
            "U",
            self.aliquot[-2:],
        )
        return " ".join(standardized)

    def any_aliquot(self) -> str:
        return " ".join((self.study[-2:].rjust(2, "0"), self.site, self.patient))

    def to_numbers(self) -> tuple[int, int, int, int]:
        return (int(self.study), int(self.site), int(self.patient), int(self.aliquot))

    def __hash__(self) -> int:
        return hash(self.to_numbers()[:3])

    def _compare(self, other, comparison: str):
        if not isinstance(other, Barcode2):
            try:
                return self._compare(Barcode2(other), comparison)
            except (ValueError, TypeError):
                return False
        self_n = self.to_numbers()
        other_n = other.to_numbers()
        if comparison[3] == "e":
            self_n = self_n[:3]
            other_n = other_n[:3]
        return all(getattr(s, comparison)(o) for s, o in zip(self_n, other_n))

    def __eq__(self, other) -> bool:
        return self._compare(other, "__eq__")

    def __lt__(self, other) -> bool:
        return self._compare(other, "__lt__")

    def __le__(self, other) -> bool:
        return self._compare(other, "__le__")

    def __gt__(self, other) -> bool:
        return self._compare(other, "__gt__")

    def __ge__(self, other) -> bool:
        return self._compare(other, "__ge__")


class LogisticGAM(pygam.LogisticGAM, ClassifierMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        super().fit(X, y, *args, **kwargs)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        y_score = super().predict_proba(X)
        return np.vstack([1 - y_score, y_score]).T


def make_gam(params, cols):
    terms = []
    for i in cols:
        lam = params[f"lam_{i}"]
        n_splines = params[f"n_splines_{i}"]
        mono = [None, "monotonic_inc"][params[f"mono_{i}"]]
        if n_splines == 2:
            terms.append(pygam.l(i, lam, "derivative"))
        else:
            terms.append(
                pygam.s(i, n_splines, min(n_splines - 1, 3), lam, "derivative", mono)
            )
    return LogisticGAM(terms=pygam.terms.TermList(*terms))


def evaluate_model(model, X_tv, y_tv, inner_cv, scorers, groups=None, weights=None):
    score_df = pd.DataFrame(
        columns=scorers, index=range(inner_cv.get_n_splits()), dtype=float
    )
    group_kws = {} if groups is None else {"groups": groups}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fold, (train_index, val_index) in enumerate(
            inner_cv.split(X_tv, y_tv, **group_kws)
        ):
            X_train, X_val = X_tv.iloc[train_index], X_tv.iloc[val_index]
            y_train, y_val = y_tv[train_index], y_tv[val_index]
            weight_kws = {} if weights is None else {"weights": weights[train_index]}
            try:
                model.fit(X_train, y_train, **weight_kws)
            except OptimizationError:
                score_df.loc[fold] = 0
            else:
                score_df.loc[fold] = [
                    sc(model, X_val, y_val) for sc in scorers.values()
                ]
    return tuple(score_df.mean())


def gam_objective(trial, X_tv, y_tv, inner_cv, scorers, groups=None, weights=None):
    params = {}
    cols = range(X_tv.shape[1])
    for i in cols:
        params[f"lam_{i}"] = trial.suggest_float(f"lam_{i}", 0.2, 800, log=True)
        params[f"n_splines_{i}"] = trial.suggest_int(f"n_splines_{i}", 2, 32, log=True)
        params[f"mono_{i}"] = trial.suggest_int(f"mono_{i}", 0, 1)
    model = make_gam(params, cols)
    return evaluate_model(
        model, X_tv, y_tv, inner_cv, scorers, groups=groups, weights=weights
    )


class NestedCV:
    def __init__(
        self, outer_cv, n_trials, scorers, inner_repeats=1, negligible=0.01, n_jobs=1
    ):
        self.outer_cv = outer_cv
        self.n_trials = n_trials
        self.inner_repeats = inner_repeats
        self.negligible = negligible
        self.n_jobs = n_jobs
        self.scorers = scorers

    def fit(self, X, y, sample_weight=None, groups=None):
        # CV
        # Initialize empty structures to store results
        range_splits = range(self.outer_cv.get_n_splits())
        cols = range(X.shape[1])
        chosen_params = pd.DataFrame(
            index=range_splits,
            columns=list(self.scorers)
            + [f"lam_{i}" for i in cols]
            + [f"n_splines_{i}" for i in cols]
            + [f"mono_{i}" for i in cols],
        )
        predicted_proba = pd.DataFrame(columns=range_splits, index=X.index, dtype=float)
        cv_gams = {}
        kws = {} if groups is None else {"groups": groups}

        for fold, (tv_index, test_index) in enumerate(
            tqdm(list(self.outer_cv.split(X, y, **kws)))
        ):
            X_tv, X_test = X.iloc[tv_index], X.iloc[test_index]
            y_tv, y_test = y[tv_index], y[test_index]
            weight_kws = (
                {} if sample_weight is None else {"weights": sample_weight[tv_index]}
            )
            group_kws = {} if groups is None else {"groups": groups[tv_index]}
            study = optuna.create_study(
                directions=["maximize"] * len(self.scorers),
                sampler=optuna.samplers.TPESampler(seed=fold),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                study.set_metric_names(list(self.scorers))
            inner_cv = self.outer_cv.__class__(
                n_splits=self.outer_cv.cvargs["n_splits"],
                n_repeats=self.inner_repeats,
                random_state=fold,
            )
            objective = functools.partial(
                gam_objective,
                X_tv=X_tv,
                y_tv=y_tv,
                inner_cv=inner_cv,
                scorers=self.scorers,
                **weight_kws,
                **group_kws,
            )
            study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
            best_trials = study.best_trials
            for i in range(len(self.scorers) - 1):
                thresh = max(trial.values[i] for trial in best_trials) - self.negligible
                best_trials = [
                    trial for trial in best_trials if trial.values[i] >= thresh
                ]
            best_trial = max(best_trials, key=lambda t: t.values[-1])
            chosen_params.loc[fold] = {
                **{
                    name: value
                    for name, value in zip(self.scorers.keys(), best_trial.values)
                },
                **best_trial.params,
            }
            cv_gams[fold] = make_gam(best_trial.params, cols).fit(
                X_tv, y_tv, **weight_kws
            )
            predicted_proba.loc[X_test.index, fold] = cv_gams[fold].predict_proba(
                X_test
            )[:, 1]
        self.chosen_params = chosen_params.convert_dtypes()
        self.predicted_proba = predicted_proba
        self.cv_gams = cv_gams
        return self


class StratifiedGroupKFoldFirst(GroupsConsumerMixin, _BaseKFold):
    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state=None,
        n_repeats: int = 1,
        first_only: bool = True,
    ):
        if n_repeats == 1:
            self.skf = StratifiedKFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )
        else:
            self.skf = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
            )
        self.first_only = first_only

    def split(self, X, y, groups):
        X, y, groups = indexable(X, y, groups)
        unique_groups, first_indices = np.unique(groups, return_index=True)

        group_y = y[first_indices]
        for train_index, test_index in self.skf.split(X=group_y, y=group_y):
            train = np.nonzero(np.isin(groups, unique_groups[train_index]))[0]
            if self.first_only:
                test = first_indices[test_index]
            else:
                test = np.nonzero(np.isin(groups, unique_groups[test_index]))[0]
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.skf.get_n_splits(X, y, groups)
