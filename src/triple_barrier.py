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
    """
    Compute triple barrier labels for a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        t (int): The look forward period for computing the triple barrier.
        devs (float): The number of standard deviations for the barrier.
        upper (float, optional): The upper barrier value. If None, it is computed based on the standard deviation.
        lower (float, optional): The lower barrier value. If None, it is computed based on the standard deviation.
        span (int, optional): The span for computing the standard deviation.

    Returns:
        pd.Series: The computed triple barrier labels.

    """
    def compute_std(df: pd.DataFrame, span: int = 100) -> pd.Series:
        """
        Compute the rolling standard deviation of returns.

        Args:
            df (pd.DataFrame): The input DataFrame.
            span (int, optional): The span for computing the standard deviation.

        Returns:
            pd.Series: The computed rolling standard deviation.

        """
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