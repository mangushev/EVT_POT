
import numpy as np
import os
import argparse
import sys

FLAGS = None


def main():
    value_array = np.loadtxt(FLAGS.value_path, delimiter=',', dtype=np.float)
    print (FLAGS.quantile, np.quantile(value_array, FLAGS.quantile))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--value_path', type=str, default=None,
            help='File with values location.')
    parser.add_argument('--quantile', type=float, default=None,
            help='Quantile from 0 to 1.')

    FLAGS, unparsed = parser.parse_known_args()

    main()
