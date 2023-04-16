
import tensorflow as tf
import numpy as np
import tensorflow_text as text
import tensorflow_hub as hub

# Define some sentences to feed into the model
sentences = [
    "Here We Go Then, You And I is a 1999 album by Norwegian pop artist Morten Abel. It was Abel's second CD as a solo artist.",
"The album went straight to number one on the Norwegian album chart, and sold to double platinum.",
"Ceylon spinach is a common name for several plants and may refer to: Basella alba Talinum fruticosum",
"A solar eclipse occurs when the Moon passes between Earth and the Sun, thereby totally or partly obscuring the image of the Sun for a viewer on Earth.",
"A partial solar eclipse occurs in the polar regions of the Earth when the center of the Moon's shadow misses the Earth.",
]

# Load the BERT encoder and preprocessing models
preprocess = hub.load(
    'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
bert = hub.load('https://tfhub.dev/google/experts/bert/wiki_books/2')

# Convert the sentences to bert inputs
bert_inputs = preprocess(sentences)

#Feed the inputs to the model to get the pooled and sequence outputs
bert_outputs = bert(bert_inputs, training=False)
pooled_output = bert_outputs['pooled_output']
sequence_output = bert_outputs['sequence_output']

print('\nSentences:')
print(sentences)
print('\nPooled output:')
print(pooled_output)
print('\nSequence output:')
print(sequence_output)
