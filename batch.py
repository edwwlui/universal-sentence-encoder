import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# Compute a representation for each message, showing various lengths supported.
word = "Elephant"
sentence = "I am a sentence for which I would like to get its embedding."
paragraph = (
    "Universal Sentence Encoder embeddings also support short paragraphs. "
    "There is no hard limit on how long the paragraph is. Roughly, the longer "
    "the more 'diluted' the embedding will be.")
messages = [word,word,word, sentence, paragraph]

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.INFO)

dataset = tf.data.Dataset.from_tensor_slices(np.array(messages))
print(type(dataset))
batched_dataset = dataset.batch(10)
iterator = batched_dataset.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  session.run(iterator.initializer)
  list_message_embeddings=[]
  while True:
    try:
      #print(type(message_embeddings))
      #print(message_embeddings)
      #print(message_embeddings.shape)
      message_embeddings = session.run(embed(next_element))
      list_message_embeddings+=message_embeddings.tolist()
    except tf.errors.OutOfRangeError:
      break
  for i, message_embedding in enumerate(list_message_embeddings):
    print("Message: {}".format(messages[i]))
    print("Embedding size: {}".format(len(message_embedding)))
    
    message_embedding_snippet = ", ".join(
        (str(x) for x in message_embedding[:3]))
    print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
