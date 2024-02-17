from trading_data import get_data
from triple_barrier import triple_barrier_labels
from temporian_prep import temporian_eventset
from train_model import train
from prediction import create_predictions
from trader import execute_strategy

from datetime import datetime
import pandas as pd

def mdx(
    df: pd.DataFrame,
    barrier_length: int,
    barrier_std: float,
    days_lookback: int,
    split_date: str,
    model_summary: str,
    verbose: int,
    lower_prob: float,
    upper_prob: float
):

    # Calculate triple barrier
    labels = triple_barrier_labels(
        df=df['close'],
        t=barrier_length,
        devs=barrier_std
    )

    # Join tbl with df
    df = df.join(labels)
    df.dropna(inplace=True)

    # One hot encode labels
    one_hot_encoded = pd.get_dummies(df['Label'], prefix='label')

    # Concatenate the one-hot encoded columns to the original DataFrame
    df = pd.concat([df, one_hot_encoded], axis=1)
    del df['Label']

    (feature_names, label_names,
     test_dataset,
     train_dataset,
     tf_test_dataset,
     tf_train_dataset) = temporian_eventset(
        df=df,
        days_lookback=days_lookback,
        split_date=split_date
    )
    # Train model
    model = train(
        train_dataset=tf_train_dataset,
        label_names=label_names,
        model_summary=model_summary,
        verbose=verbose
    )

    # Create predictions
    result_df = create_predictions(
        tf_test_dataset=tf_test_dataset,
        model=model,
        pandas_test_dataset=test_dataset
    )

    # Execute trading
    overall_profit, n_trades = execute_strategy(
        data=result_df,
        lower_prob=lower_prob,
        upper_prob=upper_prob
    )

    return overall_profit, n_trades