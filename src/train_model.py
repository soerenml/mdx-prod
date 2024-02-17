import os, datetime
import tensorflow as tf
import tensorflow_decision_forests as tfdf


def train(
    train_dataset: tf.data.Dataset,
    verbose: int,
    label_names,
    model_summary,
    tensorboard_callback: bool = False,
):
    """
    Trains a gradient boosted trees model.

    Args:
        train_dataset (tf.data.Dataset): The training dataset.
        verbose (int): Verbosity level. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        label_names: The names of the labels.
        model_summary: Whether to enable model summary.
        tensorboard_callback (bool, optional): Whether to use TensorBoard callback. Defaults to False.

    Returns:
        tfdf.keras.GradientBoostedTreesModel: The trained model.
    """
    # Build model
    model = tfdf.keras.GradientBoostedTreesModel(
        verbose=verbose,
        multitask=[
            tfdf.keras.MultiTaskItem(x, task=tfdf.keras.Task.CLASSIFICATION)
            for x in label_names
        ],
    )

    # Fit model
    if tensorboard_callback:
        model.fit(train_dataset)
    else:
        model.fit(train_dataset, callbacks=[tensorboard_callback])
        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    # Enable model summary
    model.summary() if model_summary else None

    return model