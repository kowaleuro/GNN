from functions.data_gather import data_gather
from functions.get_min_max_dict import get_min_max_dict
import tensorflow as tf

def load_model(model_abs: tf.keras.Model, min_max_ds: tf.data.Dataset, ckpt_path: str, max_count = 100):

    model = model_abs

    model.set_min_max_scores(
        get_min_max_dict(
            min_max_ds,
            model.min_max_scores_fields,
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanAbsolutePercentageError(),
    )
    model.load_weights(tf.train.latest_checkpoint(ckpt_path))

    return model
