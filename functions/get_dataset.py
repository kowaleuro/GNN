import tensorflow as tf

def combined_dataset(loaded_datasets):
  combined_dataset = loaded_datasets[0]
  for dataset in loaded_datasets[1:]:
      combined_dataset = combined_dataset.concatenate(dataset)
  return combined_dataset

def get_dataset(ds_path: str):
  ds_test_temp = []

  for i in range(0,5):
    ds_test_temp.append(tf.data.Dataset.load(f"{ds_path}/{i}/validation", compression="GZIP"))
    ds_test_temp.append(tf.data.Dataset.load(f"{ds_path}/{i}/training", compression="GZIP"))

  return combined_dataset(ds_test_temp)
