
import tensorflow as tf
import numpy as np
import os
import argparse
import sys
import random
import logging

FLAGS = None

np.set_printoptions(edgeitems=12, linewidth=10000, precision=4, suppress=True)

logger = logging.getLogger('tensorflow')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

def example(value):
    record = {
        'value': tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    }

    return tf.train.Example(features=tf.train.Features(feature=record))

def create_records(value_file, threshold, tfrecords_file):

    with tf.io.TFRecordWriter(tfrecords_file.format(os.path.splitext(os.path.basename(value_file))[0])) as writer:

        value_array = np.loadtxt(value_file, delimiter=',', dtype=np.float)

        record_count = 0
        i = 0

        while i < value_array.shape[0]:

            if value_array[i] > threshold:
                tf_example = example(value_array[i]-threshold)
                writer.write(tf_example.SerializeToString())
                record_count = record_count + 1

            i = i + 1

    logger.info ("total {} over threshold {}".format(i, record_count))

def main():
    create_records(FLAGS.files_path, FLAGS.t, FLAGS.tfrecords_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
    parser.add_argument('--files_path', type=str, default=None,
            help='File with values location.')
    parser.add_argument('--t', type=float, default=None,
            help='Initial threshold.')
    parser.add_argument('--tfrecords_file', type=str, default='gs://anomaly_detection/pot/data/train/{}.tfrecords',
            help='tfrecords output file. It will be used as a prefix if split.')

    FLAGS, unparsed = parser.parse_known_args()

    logger.setLevel(FLAGS.logging)

    logger.debug ("Running with parameters: {}".format(FLAGS))

    main()
