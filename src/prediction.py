# Create predictions
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd

def create_predictions(
    tf_test_dataset: tf.data.Dataset,
    model: tf.keras.Model,
    pandas_test_dataset: pd.DataFrame
):
    """
    Run predictions
    """
    
    # Run predictions
    predictions = model.predict(tf_test_dataset, verbose=0)
    
    # Get predictions in form of a dataframe
    dicto = {'l_sell': [], 'l_hold': [], 'l_buy': []}
    
    for i in predictions:
        for j in predictions[i]:
            dicto[i].append(j[0])
    
    df_pred = pd.DataFrame(dicto)
    
    # Get the highest probability
    df_pred['prob_strat'] = df_pred.apply(lambda x: x.max(), axis=1)
    
    # Get the label name with the highest probability
    df_pred['strategy'] = pd.DataFrame(dicto).idxmax(axis=1)
    
    # Get the date
    df_pred['date'] = pd.to_datetime(pandas_test_dataset['timestamp'], unit='s')

    # Merge predictions with test dataset    
    result_df = pd.concat([df_pred, pandas_test_dataset], axis=1)
    
    # Select variables
    result_df = result_df[['date', 'close', 'strategy', 'prob_strat']]
    
    return result_df