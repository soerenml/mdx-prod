import os, datetime
import tensorflow as tf
import tensorflow_decision_forests as tfdf


def train(
    train_dataset: tf.data.Dataset,
    verbose: int,
    label_names,
    model_summary,
):
    """
    Trains a boosted tree model.

    Args:
        train_dataset (tf.data.Dataset): The training dataset.
        verbose (int): Verbosity level. Controls the amount of information printed during training.
        label_names: The names of the labels used for training.
        model_summary: Flag indicating whether to print the model summary.

    Returns:
        tfdf.keras.GradientBoostedTreesModel: The trained model.
    """
    # Tensorboard
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    # Build model
    model = tfdf.keras.GradientBoostedTreesModel(
        verbose=verbose,
        multitask=[
            tfdf.keras.MultiTaskItem(x, task=tfdf.keras.Task.CLASSIFICATION)
            for x in label_names
        ],
    )

    # Fit model
    model.fit(train_dataset, callbacks=[tensorboard_callback])

    # Enable model summary
    model.summary() if model_summary else None

    return model