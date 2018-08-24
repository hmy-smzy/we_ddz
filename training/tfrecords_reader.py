import tensorflow as tf


def read_data(batch_size, file_path):
    reader = tf.TFRecordReader()
    filename = tf.train.match_filenames_once(file_path)
    filename_queue = tf.train.string_input_producer(filename)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'sample': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'legal': tf.FixedLenFeature([], tf.string),
            'score': tf.FixedLenFeature([], tf.float32)
        })
    input_ = tf.decode_raw(features['sample'], tf.uint8)
    input_ = tf.cast(tf.reshape(input_, shape=[22, 43, 15]), dtype=tf.float32)
    input_ = tf.transpose(input_, perm=[1, 2, 0])
    label_ = tf.cast(features['label'], tf.int32)
    legal_ = tf.decode_raw(features['legal'], tf.uint8)
    legal_ = tf.cast(tf.reshape(legal_, shape=[309]), dtype=tf.float32)
    score_ = tf.reshape(features['score'], shape=[1])

    x, sparse_labels, legal_label, score = tf.train.shuffle_batch(
        [input_, label_, legal_, score_],
        batch_size=batch_size, num_threads=4,
        capacity=5000 + 10 * batch_size,
        min_after_dequeue=5000)
    y = tf.one_hot(sparse_labels, 309)
    return x, y, legal_label, score


if __name__ == '__main__':
    x, y, legal_score, r = read_data(128, 'E:/cnn_data/training')
