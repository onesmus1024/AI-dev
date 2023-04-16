import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import logging
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


logging.basicConfig(level=logging.INFO)





f = 5
n = 10
np.random.seed(100)
x = np.random.randint(0, 2, (n, f))
y = np.random.randint(0, 2, n)

x_train = x[:int(n*0.8)]
y_train = y[:int(n*0.8)]
x_test = x[int(n*0.8):]
y_test = y[int(n*0.8):]


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




history = model.fit(x_train, y_train, epochs=100, batch_size=1,validation_split=0.2,
                    callbacks=[tensorboard]
)


predicted = model.predict(x_train)

y_ = np.where(predicted.flatten() > 0.5, 1, 0)

# print formated accuracy score

print('Accuracy score for train data: {:.2f}'.format(accuracy_score(y_train, y_)))
# print formated accuracy score for test data

print('Accuracy score for test data: {:.2f}'.format(accuracy_score(y_test, np.where(model.predict(x_test).flatten() > 0.5, 1, 0))))

res = pd.DataFrame(history.history)
res.plot(figsize=(10, 6))
plt.show()
