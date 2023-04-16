import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text as text

# Load the IMDB dataset
# imdb = tf.keras.datasets.imdb
# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
#     num_words=10000)

max_seq_length = 128


# Load the BERT model from TensorFlow Hub
bert_layer = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", trainable=True)

# Define the model architecture
input_word_ids = tf.keras.layers.Input(
    shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
input_mask = tf.keras.layers.Input(
    shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
input_type_ids = tf.keras.layers.Input(
    shape=(max_seq_length,), dtype=tf.int32, name="input_type_ids")

pooled_output, sequence_output = bert_layer(
    [input_word_ids, input_mask, input_type_ids])
dropout = tf.keras.layers.Dropout(0.2)(pooled_output)
output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(dropout)

model = tf.keras.Model(
    inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)

# Compile the model
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(
    lr=2e-5), metrics=["accuracy"])

# Train the model
# model.fit(train_input, train_labels,
#           validation_split=0.2, epochs=3, batch_size=32)

# # Evaluate the model on the test set
# test_loss, test_acc = model.evaluate(test_input, test_labels)

# # Make predictions on new data
# new_data = np.array(["This was a great movie! Highly recommend."])
# new_data_input = convert_examples_to_features(
#     new_data, tokenizer, max_seq_length)
# prediction = model.predict(new_data_input)
