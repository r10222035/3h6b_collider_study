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

        # self.bn = tf.keras.layers.BatchNormalization()

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

        output = self.network(inputs)

        return output


def main():
    config_path = sys.argv[1]
    print(f'config_path: {config_path}')

    # Read config file
    with open(config_path, 'r') as f:
        config = json.load(f)

    sig_path = config['signal_npy_path']
    bkg_path = config['background_npy_path']
    batch_size = config['batch_size']

    model_name = config['model_name']
    sample_type = config['sample_type']

    # Load data
    ns, nb = 50000, 50000
    X_s = np.load(sig_path)[:ns]
    X_b = np.load(bkg_path)[:nb]

    r_train, r_val, r_test = 0.7, 0.15, 0.15

    X_s_train = X_s[:int(len(X_s)*r_train)]
    X_s_val = X_s[int(len(X_s)*r_train):int(len(X_s)*(r_train+r_val))]
    X_s_test = X_s[int(len(X_s)*(r_train+r_val)):]

    X_b_train = X_b[:int(len(X_b)*r_train)]
    X_b_val = X_b[int(len(X_b)*r_train):int(len(X_b)*(r_train+r_val))]
    X_b_test = X_b[int(len(X_b)*(r_train+r_val)):]

    X_train = np.concatenate((X_s_train, X_b_train), axis=0)
    X_val = np.concatenate((X_s_val, X_b_val), axis=0)
    X_test = np.concatenate((X_s_test, X_b_test), axis=0)

    y_train = np.concatenate((np.ones(len(X_s_train)), np.zeros(len(X_b_train))), axis=0)
    y_val = np.concatenate((np.ones(len(X_s_val)), np.zeros(len(X_b_val))), axis=0)
    y_test = np.concatenate((np.ones(len(X_s_test)), np.zeros(len(X_b_test))), axis=0)

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


    # Create the model  
    model = DNN()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, verbose=1, patience=patience)
    check_point    = tf.keras.callbacks.ModelCheckpoint(save_model_name, monitor='val_loss', verbose=1, save_best_only=True)

    history = model.fit(train_dataset, validation_data=valid_dataset, epochs=train_epochs,
                        callbacks=[early_stopping,
                                check_point,
                                ]
                        )
    
    # Load model
    loaded_model = tf.keras.models.load_model(save_model_name)
    results = loaded_model.evaluate(test_dataset)
    print(f'Testing Loss = {results[0]:.3}, Testing Accuracy = {results[1]:.3}')

    X, y = X_test, y_test
    y_pred = loaded_model.predict(X)
    auc = roc_auc_score(y, y_pred)
    acc = get_highest_accuracy(y, y_pred)

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