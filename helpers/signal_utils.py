import numpy as np
import pandas as pd


def trailing_return(r_m: pd.Series | pd.DataFrame, k: int) -> pd.Series | pd.DataFrame:
    tr = (1 + r_m).rolling(k).apply(np.prod, raw=True) - 1

    if isinstance(tr, pd.Series):
        tr.name = f"trailing_{k}m"

    return tr


def sign_signal(trailing_r: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    sig = np.sign(trailing_r)

    if isinstance(sig, pd.Series):
        sig.name = "signal"

    return sig


def _weighted_mean(
    pieces: dict[int, pd.Series | pd.DataFrame],
    weights: dict[int, float] | None = None,
) -> pd.Series | pd.DataFrame:
    if not pieces:
        raise ValueError("pieces must be a non-empty dict")

    keys = list(pieces.keys())
    if weights is None:
        weights = {k: 1.0 / len(keys) for k in keys}
    else:
        missing = set(keys) - set(weights)
        if missing:
            raise ValueError(f"weights missing horizons: {sorted(missing)}")
        w_sum = sum(weights[k] for k in keys)
        if w_sum <= 0:
            raise ValueError("weights must sum to a positive number")
        weights = {k: weights[k] / w_sum for k in keys}

    total = None
    for k in keys:
        piece = pieces[k].astype(float) * weights[k]
        total = piece if total is None else total.add(piece, fill_value=np.nan)
    return total


def multi_horizon_score(
    signals: dict[int, pd.Series | pd.DataFrame],
    weights: dict[int, float] | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Weighted average of sign signals across horizons.

    Equal weights (default) with four horizons → values in
    {-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1}.
    """
    score = _weighted_mean(signals, weights=weights)
    if isinstance(score, pd.Series):
        score.name = "score"
    return score


def raw_return_score(
    trailing: dict[int, pd.Series | pd.DataFrame],
    weights: dict[int, float] | None = None,
    as_sign: bool = True,
) -> pd.Series | pd.DataFrame:
    """
    Average trailing *returns* across horizons, optionally take sign.

    as_sign=True  → discrete ±1 / 0 signal from mean return
    as_sign=False → continuous mean return (then typically clip/sign downstream)
    """
    avg = _weighted_mean(trailing, weights=weights)
    out = np.sign(avg) if as_sign else avg
    if isinstance(out, pd.Series):
        out.name = "raw_score"
    return out


def majority_vote(
    signals: dict[int, pd.Series | pd.DataFrame],
    min_agree: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Sign of the sum of horizon signs. Optional threshold: flat unless
    at least `min_agree` horizons share the same non-zero sign.
    """
    if not signals:
        raise ValueError("signals must be a non-empty dict")

    total = None
    for sig in signals.values():
        piece = sig.astype(float)
        total = piece if total is None else total.add(piece, fill_value=np.nan)

    vote = np.sign(total)
    if min_agree is not None:
        if min_agree < 1:
            raise ValueError("min_agree must be >= 1")
        # |sum of signs| >= min_agree means at least that many agree net
        vote = vote.where(total.abs() >= min_agree, 0.0)

    if isinstance(vote, pd.Series):
        vote.name = "majority"
    return vote


def threshold_score(
    score: pd.Series | pd.DataFrame,
    min_abs: float = 0.5,
) -> pd.Series | pd.DataFrame:
    """Keep score only when |score| >= min_abs; otherwise flat (0)."""
    out = score.where(score.abs() >= min_abs, 0.0)
    if isinstance(out, pd.Series):
        out.name = "threshold_score"
    return out


def agreement_label(
    score: pd.Series | pd.DataFrame,
    high: float = 0.75,
    mid: float = 0.25,
) -> pd.Series | pd.DataFrame:
    """
    Bucket |score| into high / mid / low agreement.

    high: |score| >= high
    mid:  mid <= |score| < high
    low:  |score| < mid
    """
    abs_score = score.abs()

    def _label(x: float) -> str:
        if np.isnan(x):
            return np.nan
        if x >= high:
            return "high"
        if x >= mid:
            return "mid"
        return "low"

    if isinstance(abs_score, pd.Series):
        out = abs_score.map(_label)
        out.name = "agreement"
        return out

    return abs_score.apply(lambda col: col.map(_label))
