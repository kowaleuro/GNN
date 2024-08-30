from typing import Optional, Tuple
import tensorflow as tf
import numpy as np

def get_min_max_dict(
    ds: tf.data.Dataset, params: list[str], include_y: Optional[str] = None
) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Get the min and the max-min for different parameters of a dataset. Later used by the models for the min-max normalization.

    Parameters
    ----------
    ds : tf.data.Dataset
        Training dataset where to base the min-max normalization from.

    params : List[str]
        List of strings indicating the parameters to extract the features from.

    include_y : Optional[str], optional
        Indicates if to also extract the features of the output variable.
        Inputs indicate the string key used on the return dict. If None, it is not included.
        By default None.

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary containing the values needed for the min-max normalization.
        The first value is the min value of the parameter, and the second is 1 / (max - min).
    """

    # Use first sample to get the shape of the tensors
    iter_ds = iter(ds)
    sample, label = next(iter_ds)
    params_lists = {param: sample[param].numpy() for param in params}
    if include_y:
        params_lists[include_y] = label.numpy()

    # Include the rest of the samples
    for sample, label in iter_ds:
        for param in params:
            params_lists[param] = np.concatenate(
                (params_lists[param], sample[param].numpy()), axis=0
            )
        if include_y:
            params_lists[include_y] = np.concatenate(
                (params_lists[include_y], label.numpy()), axis=0
            )

    scores = dict()
    for param, param_list in params_lists.items():
        min_val = np.min(param_list, axis=0)
        min_max_val = np.max(param_list, axis=0) - min_val
        if min_max_val.size == 1 and min_max_val == 0:
            scores[param] = [min_val, 0]
            print(f"Min-max normalization Warning: {param} has a max-min of 0.")
        elif min_max_val.size > 1 and np.any(min_max_val == 0):
            min_max_val[min_max_val != 0] = 1 / min_max_val[min_max_val != 0]
            scores[param] = [min_val, min_max_val]
            print(
                f"Min-max normalization Warning: Several values of {param} has a max-min of 0."
            )
        else:
            scores[param] = [min_val, 1 / min_max_val]

    return scores