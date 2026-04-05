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