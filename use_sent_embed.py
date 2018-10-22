import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import gc



def use(messages):
    # with tf.device('/device:GPU:0'):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module(module_url)
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    dataset = tf.data.Dataset.from_tensor_slices(np.array(messages))
    batched_dataset = dataset.batch(65536)

    iterator = batched_dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as session:
        session = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False, ))
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        session.run(iterator.initializer)
        list_message_embeddings = []
        while True:
            try:
                message_embeddings = session.run(embed(next_element))
                list_message_embeddings += message_embeddings.tolist()

            except tf.errors.OutOfRangeError:
                break

    session.close()
    gc.collect()
    return list_message_embeddings


def main():
    SAMPLE_SENTENCES_PATH = '/home/fyp1/language_style_transfer/code/is' \
                            '/InferSent/encoder/samples.txt'
    DATASET_PATH = "~/language_style_transfer/data/dataset/ye/yelp.all"

    # with tf.device('/device:GPU:0'):
    """
    messages = []
    with open(os.path.expanduser(DATASET_PATH), "r") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(line)
    """
    dataset = tf.data.TextLineDataset(os.path.expanduser(DATASET_PATH))
    batched_dataset = dataset.batch(65536)  #modify batch size, 65536 max tested on kao
    iterator = batched_dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        session.run(iterator.initializer)
        while True:
            try:
                matrix = use(session.run(next_element).tolist())
                gc.collect()
                

                """
                #do something with matrix
                for i, message_embedding in enumerate(matrix):
                     print("Message: {}".format(messages[i]))
                    print("Embedding size: {}".format(len(matrix)))
                    matrix_snippet = ", ".join(
                        (str(x) for x in matrix[:3]))
                    print("Embedding: [{}, ...]\n".format(matrix_snippet))
                    if (i == 3):
                        break
                """
            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':
    main()

