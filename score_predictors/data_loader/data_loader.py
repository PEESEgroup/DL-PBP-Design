import tensorflow as tf

def load_array(feature_arrays, label_arrays, batch_size, is_train=True, buffer_size=10000):
    feature_dataset = tf.data.Dataset.from_tensor_slices(feature_arrays)
    label_dataset = tf.data.Dataset.from_tensor_slices(label_arrays)
    dataset = tf.data.Dataset.zip((feature_dataset, label_dataset))
    if is_train:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=42)
    dataset = dataset.repeat().batch(batch_size)
    return dataset