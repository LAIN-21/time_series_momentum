import pandas as pd
import numpy as np

def trailing_return(r_m: pd.Series, k: int) -> pd.Series:
    tr = (1 + r_m).rolling(k).apply(np.prod, raw=True) - 1
    tr.name = f"trailing_{k}m"
    return tr

def sign_signal(trailing_r: pd.Series) -> pd.Series:
    sig = pd.Series(
        np.where(trailing_r > 0, 1, np.where(trailing_r < 0, -1, 0)),
        index=trailing_r.index
    )
    sig.name = "signal"
    return sig