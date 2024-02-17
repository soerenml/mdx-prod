import temporian as tp
from typing import List
import pandas as pd


def temporian_eventset(
    df: pd.DataFrame,
    days_lookback: int,
    split_date: str
):
    """
    Generate a temporian event set for a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the necessary columns.
        days_lookback (int): The number of days to look back for creating lagged features.
        split_date (str): The date to split the data into training and testing sets.

    Returns:
        Tuple: A tuple containing the following elements:
            - feature_names (List[str]): The names of the generated features.
            - label_names (List[str]): The names of the generated labels.
            - test_dataset (pd.DataFrame): The testing dataset.
            - train_dataset (pd.DataFrame): The training dataset.
            - tf_test_dataset (tf.data.Dataset): The testing dataset converted to TensorFlow Dataset.
            - tf_train_dataset (tf.data.Dataset): The training dataset converted to TensorFlow Dataset.
    """
    # Function code here...
def temporian_eventset(
    df: pd.DataFrame,
    days_lookback: int,
    split_date: str
):

    ### Detailed specification of EventSet
    f_data = tp.event_set(
        timestamps=df['timestamp'],
        features={
            "close": df['close'],
            "volume": df['volume'],
            "high": df['high'],
            "low": df['low'],
            "ticker": df['id'],
            "sell": df['label_-1.0'],
            "hold": df['label_0.0'],
            "buy": df['label_1.0']

          },
        indexes=['ticker']
    )

    ### Raw features ###
    all_features = [
        # LABELS
        f_data["close"].prefix("f_"),
        f_data["volume"].prefix("f_"),
        f_data["high"].prefix("f_"),
        f_data["low"].prefix("f_"),
    ]

    ### Labels ###
    labels = [
        f_data["sell"].prefix("l_"),
        f_data["hold"].prefix("l_"),
        f_data["buy"].prefix("l_"),
    ]

    # Glue features
    all_features = tp.glue(*all_features, *labels)


    ### Lookback ###
    range_days_lookback = [x for x in range(1,days_lookback)]

    feature_list=['close', 'volume', 'low', 'high']
    for var in feature_list:
        lagged_sales_list: List[tp.EventSet] = []

        for horizon in range_days_lookback:
            x = f_data[var].lag(tp.duration.days(horizon)) # change to days if needed.
            x = x.resample(f_data)
            x = x.rename(f"f_{var}_lag_{horizon}_d")
            lagged_sales_list.append(x)

        feature_lagged_sales = tp.glue(*lagged_sales_list)

    # Glue features
    all_features = tp.glue(all_features, feature_lagged_sales)

    ### Moving statistics ###
    feature_list=['close', 'volume', 'low', 'high']

    for var in feature_list:
        moving_stats_list: List[tp.EventSet] = []

        float_sales = f_data[var].cast(tp.float32)

        for win_day in [2, 3, 5, 7, 10, 15, 30, 60, 90, 144]:
            # Define the duration for the days
            win = tp.duration.days(win_day) # change to days if needed

            # Calculate moving average
            x = float_sales.simple_moving_average(win).prefix(
                f"f_{var}_ma_{win_day}_"
            )
            moving_stats_list.append(x)

            # Calculate moving standard devition
            x = float_sales.moving_standard_deviation(win).prefix(
                f"f_{var}_sd_{win_day}_"
            )
            moving_stats_list.append(x)

            # Calculate moving max
            x = float_sales.moving_max(win).prefix(
                f"f_{var}_max_{win_day}_"
            )
            moving_stats_list.append(x)

            # Calculate moving min
            x = float_sales.moving_min(win).prefix(
                f"f_{var}_min_{win_day}_"
            )
            moving_stats_list.append(x)

        feature_moving_stats = tp.glue(*moving_stats_list)

    # Glue features
    all_features = tp.glue(all_features, feature_moving_stats)


    ### Calendar dates ###
    calendar_list: List[tp.EventSet] = []
    calendar_list.append(f_data.calendar_day_of_month())
    calendar_list.append(f_data.calendar_day_of_week())
    calendar_list.append(f_data.calendar_month())
    feature_calendar = tp.glue(*calendar_list).prefix("f_")

    # Glue features
    all_features = tp.glue(all_features, feature_calendar)


    from datetime import datetime
    import tensorflow as tf
    import tensorflow_decision_forests as tfdf

    # Create combined EventSet
    tabular = tp.glue(all_features, f_data)

    # Create test & train EventSet
    tabular_test = tabular.after(split_date)
    tabular_train = tabular.before(split_date)

    # Convert to pd.DataFrame
    test_dataset = tp.to_pandas(tabular_test, timestamp_to_datetime=False)
    train_dataset = tp.to_pandas(tabular_train, timestamp_to_datetime=False)

    # Convert to tf.data.Dataset
    label_names = [x for x in test_dataset.columns if x.startswith("l_")]
    feature_names = [x for x in test_dataset.columns if x.startswith("f_")]

    def dataset_pandas_to_tensorflow_dataset(df):
        features = {k: df[k] for k in feature_names}
        labels = {k: df[k] for k in label_names}
        return tf.data.Dataset.from_tensor_slices((features, labels)).batch(100)


    tf_test_dataset = dataset_pandas_to_tensorflow_dataset(test_dataset)
    tf_train_dataset = dataset_pandas_to_tensorflow_dataset(train_dataset)

    return feature_names, label_names, test_dataset, train_dataset, tf_test_dataset, tf_train_dataset