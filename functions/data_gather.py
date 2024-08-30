# @title Data Gather
from ipywidgets import IntProgress
from IPython.display import display
import tensorflow as tf
import numpy as np

def data_gather(ds,model,max_count = 20):
    f = IntProgress(min=0, max=max_count,
        description='Loading:',
        bar_style='success',
        orientation='horizontal')
    display(f) # display the bar
    # -----------////////////-----------------
    original = [[],[],[]]
    predicted = [[],[],[]]

    mean_load_list = []
    for ii, (sample_features, labels) in enumerate(iter(ds)):
        # graphVisualization(sample_features)
        prediction_list = individual_prediction(model, sample_features)
        mean_load_list += (getAvgPathLoad(sample_features))
        for increment,value in enumerate(prediction_list):
            if value == []:
                break
            predicted[increment] += value
            try:
                original[increment] += labels[increment].numpy().tolist()
            except TypeError:
                original[increment] += labels.numpy().tolist()
            f.value += 1
            if ii == max_count:
                break
# -----------////////////-----------------
    return original,predicted,mean_load_list
    

# @title Data Processing
def individual_prediction(model: tf.keras.Model, sample: any) -> np.ndarray:
    output = model(sample)
    gather = []
    if isinstance(output,tuple):
      for data in output:
        gather.append(data.numpy().reshape((-1,)).tolist())
    else:
        gather = [[]]
        gather[0] += (output.numpy().reshape((-1,)).tolist())
    return gather

def getAvgPathLoad(sample):
    link_capacity = sample['link_capacity']
    link_to_path = sample['link_to_path']
    path_gather_traffic = getTraffic(sample)
    load = tf.math.reduce_sum(path_gather_traffic, axis=1) / (link_capacity * 1e9)
    avgLoad_list = []
    mean_load_on_path = tf.math.reduce_mean(tf.gather(load, link_to_path),axis=1)
    return np.squeeze(mean_load_on_path.numpy()).tolist()


def getTraffic(sample):
    flow_traffic = sample['flow_traffic']
    path_to_link = sample['path_to_link']
    path_gather_traffic = (tf.gather(flow_traffic, path_to_link[:, :, 0]))
    return path_gather_traffic