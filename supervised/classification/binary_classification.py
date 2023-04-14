import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import logging

logging.basicConfig(level=logging.INFO)





f = 5
n = 10
np.random.seed(100)
x = np.random.randint(0, 2, (n, f))
y = np.random.randint(0, 2, n)


model = Sequential()
model.add(Dense(256, activation='relu', input_dim=f))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop',
              metrics=['acc'])


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='model.h5', monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch',
    options=None
)

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
    update_freq='epoch', profile_batch=2, embeddings_freq=0,
    embeddings_metadata=None
)




history = model.fit(x, y, epochs=100, verbose=1, 
                    validation_split=0.2)



predicted = model.predict(x)

y_ = np.where(predicted.flatten() > 0.5, 1, 0)

y == y_

res = pd.DataFrame(history.history)



res.plot(figsize=(10, 6))
plt.show()
