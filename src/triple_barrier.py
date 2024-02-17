import numpy as np
import pandas as pd

def triple_barrier_labels(
    df: pd.DataFrame,
    t: int,
    devs: float,
    upper: float = None,
    lower: float = None,
    span: int = 100
) -> pd.Series:

    def compute_std(
        df: pd.DataFrame,
        span: int = 100
    ) -> pd.Series:
        df = df.ffill()
        returns = df.pct_change()
        return returns.ewm(span=span).std()
    
    df = df.ffill()
    returns = df.pct_change()

    labels = pd.Series(index=df.index, name='Label', dtype=int)

    for idx in range(len(df) - 1 - t):
        interval = returns.iloc[idx:idx + t]
        cumsum_values = interval.cumsum().values

        # Check for NaNs and infs
        if not np.all(np.isfinite(cumsum_values)):
            labels.iloc[idx] = np.nan
            continue

        std = compute_std(df.iloc[:idx + t], span)

        u = std.iloc[idx] * devs if upper is None else upper
        l = -std.iloc[idx] * devs if lower is None else lower

        if np.any(cumsum_values >= u):
            labels.iloc[idx] = 1
        elif np.any(cumsum_values <= l):
            labels.iloc[idx] = -1
        else:
            labels.iloc[idx] = 0

    return labels