# build and train the PepBD score predictor for PE-binding peptides
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
import keras.backend as K
import os
import sys
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_script_dir, '../../data_loader/'))
from data_loader import load_array


def main():
    # load and process the sequence-score data for PE-binding peptides
    df = pd.read_csv('../../../data/sequence_score_data_pe_processed.csv')

    sequences = df['Short Sequence'].to_list()
    scores = df['NET_E'].to_list()

    amino_acid_alphabet = ['A', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                           'M', 'N', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    encoder = LabelEncoder()
    encoder.fit(amino_acid_alphabet)

    encoded_sequences = [encoder.transform(list(seq)) for seq in sequences]
    X = pad_sequences(encoded_sequences)

    y = np.array(scores)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    dataset_train = load_array(X_train, y_train, batch_size=128, buffer_size=X_train.shape[0])
    dataset_val = load_array(X_val, y_val, batch_size=128, is_train=False, buffer_size=None)

    # define the model
    model = Sequential()
    model.add(Embedding(input_dim=len(amino_acid_alphabet), output_dim=32, input_length=len(X[0])))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1, activation='linear'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    checkpoint_dir = '../ckpts'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True
    )

    # train and save the trained model
    history = model.fit(
        x=dataset_train,
        validation_data=dataset_val,
        callbacks=[checkpoint_callback],
        steps_per_epoch=int(np.round(X_train.shape[0]/128)),
        validation_steps=int(np.round(X_val.shape[0]/128)),
        epochs=200
    )

    model.save('../trained_model/')

if __name__ == '__main__':
    main()
