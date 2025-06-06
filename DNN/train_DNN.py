import os
import sys
import json
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# solve the problem of "libdevice not found at ./libdevice.10.bc"
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/r10222035/.conda/envs/tf2'

def get_sample_size(y):
    if len(y.shape) == 1:
        ns = (y == 1).sum()
        nb = (y == 0).sum()
    else:
        ns = (y.argmax(axis=1) == 1).sum()
        nb = (y.argmax(axis=1) == 0).sum()
    print(ns, nb)
    return ns, nb


def get_highest_accuracy(y_true, y_pred):
    _, _, thresholds = roc_curve(y_true, y_pred)
    # compute highest accuracy
    thresholds = np.array(thresholds)
    if len(thresholds) > 1000:
        thresholds = np.percentile(thresholds, np.linspace(0, 100, 1001))
    accuracy_scores = []
    for threshold in thresholds:
        accuracy_scores.append(accuracy_score(y_true, y_pred>threshold))

    accuracies = np.array(accuracy_scores)
    return accuracies.max()


class DNN(tf.keras.Model):
    def __init__(self, name='DNN'):
        super(DNN, self).__init__(name=name)

        self.bn = tf.keras.layers.BatchNormalization()

        self.network = tf.keras.Sequential([
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

    @tf.function
    def call(self, inputs, training=False):

        inputs = self.bn(inputs)
        output = self.network(inputs)

        return output


def main():
    config_path = sys.argv[1]
    print(f'config_path: {config_path}')

    # Read config file
    with open(config_path, 'r') as f:
        config = json.load(f)

    train_path = config['train_npy_path']
    test_path = config['test_npy_path']
    six_b_test_path = config['6b_test_npy_path']
    batch_size = config['batch_size']

    model_name = config['model_name']
    sample_type = config['sample_type']

    r_train = 0.95
    X = np.load(train_path.replace('.npy', '-data.npy'))
    y = np.load(train_path.replace('.npy', '-label.npy'))

    # shuffle data
    np.random.seed(402)
    idx = np.random.permutation(len(y))
    X = X[idx]
    y = y[idx]

    try:
        train_sample_size = config['train_sample_size']
    except KeyError:
        train_sample_size = len(y)
    X = X[:train_sample_size]
    y = y[:train_sample_size]

    X_train = X[:int(train_sample_size * r_train)]
    X_val = X[int(train_sample_size * r_train):]

    y_train = y[:int(train_sample_size * r_train)]
    y_val = y[int(train_sample_size * r_train):]

    X_test = np.load(test_path.replace('.npy', '-data.npy'))
    y_test = np.load(test_path.replace('.npy', '-label.npy'))

    X_test_6b = np.load(six_b_test_path.replace('.npy', '-data.npy'))
    y_test_6b = np.load(six_b_test_path.replace('.npy', '-label.npy'))

    train_size = get_sample_size(y_train)
    val_size = get_sample_size(y_val)
    test_size = get_sample_size(y_test)

    BATCH_SIZE = batch_size
    with tf.device('CPU'):
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(y_train)).batch(BATCH_SIZE)

        valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        valid_dataset = valid_dataset.batch(BATCH_SIZE)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(BATCH_SIZE)


    # Training parameters
    train_epochs = 500
    patience = 10
    min_delta = 0.
    learning_rate = 1e-3
    save_model_name = f'DNN_models/last_model_{model_name}/'

    class_weight = {0: 1.0, 1: train_size[1] / train_size[0]}

    # Create the model  
    model = DNN()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, verbose=1, patience=patience)
    check_point    = tf.keras.callbacks.ModelCheckpoint(save_model_name, monitor='val_loss', verbose=1, save_best_only=True)

    history = model.fit(train_dataset, validation_data=valid_dataset, epochs=train_epochs, class_weight=class_weight,
                        callbacks=[early_stopping,
                                check_point,
                                ]
                        )
    
    # Load model
    loaded_model = tf.keras.models.load_model(save_model_name)
    results = loaded_model.evaluate(test_dataset)
    print(f'Testing Loss = {results[0]:.3}, Testing Accuracy = {results[1]:.3}')

    X, y = X_test, y_test
    y_pred = loaded_model.predict(X, batch_size=batch_size)
    auc = roc_auc_score(y, y_pred)
    acc = get_highest_accuracy(y, y_pred)

    X, y = X_test_6b, y_test_6b
    y_pred = loaded_model.predict(X, batch_size=batch_size)
    auc_6b = roc_auc_score(y, y_pred)
    acc_6b = get_highest_accuracy(y, y_pred)

    # Write results
    now = datetime.datetime.now()
    file_name = 'resonant_DNN_training_results.csv'
    data_dict = {
                'Train signal size': [train_size[0]],
                'Train background size': [train_size[1]],
                'Validation signal size': [val_size[0]],
                'Validation background size': [val_size[1]],
                'Test signal size': [test_size[0]],
                'Test background size': [test_size[1]],
                'Loss': [results[0]],
                'ACC': [acc],
                'AUC': [auc],
                'ACC_6b': [acc_6b],
                'AUC_6b': [auc_6b],
                'Sample Type': [sample_type],
                'Model Name': [model_name],
                'Training epochs': [len(history.history['loss']) + 1],
                'time': [now],
                }

    df = pd.DataFrame(data_dict)
    if os.path.isfile(file_name):
        training_results_df = pd.read_csv(file_name)
        pd.concat([training_results_df, df], ignore_index=True).to_csv(file_name, index=False)
    else:
        df.to_csv(file_name, index=False)


if __name__ == '__main__':
    main()