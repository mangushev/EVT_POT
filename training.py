
import numpy as np
np.set_printoptions(edgeitems=25, linewidth=10000, precision=4, suppress=True)

import collections
import re
import argparse
import sys
import os
import tensorflow as tf

from model import POT, get_shape_list

FLAGS = None


def make_input_fn(filename, is_training, drop_reminder):
  """Returns an `input_fn` for train and eval."""

  def input_fn(params):
    def parser(serialized_example):
      example = tf.io.parse_single_example(
          serialized_example,
          features={
              "value": tf.io.FixedLenFeature((), tf.float32)
          })
      
      return example

    dataset = tf.data.TFRecordDataset(
      filename) #, buffer_size=FLAGS.dataset_reader_buffer_size)
    
    if is_training:
      dataset = dataset.repeat()
      #dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size, reshuffle_each_iteration=True)

    dataset = dataset.map(parser)

    dataset = dataset.batch(params["batch_size"])

    #dataset = dataset.apply(
    #  tf.contrib.data.map_and_batch(
    #    parser, batch_size=params["batch_size"]
    #    num_parallel_batches=8,
    #    drop_remainder=drop_reminder))
    return dataset

  return input_fn

def model_fn_builder(init_checkpoint, learning_rate, num_train_steps, use_tpu):

  def model_fn(features, labels, mode, params):

    exceedances = features["value"]

    is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False

    model = POT(exceedances,
      n=params["n"],
      t=params["t"],
      q=params["q"],
      initializer_range=params["initializer_range"])

    if mode == tf.estimator.ModeKeys.TRAIN:

      tvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

      grads = tf.gradients(model.loss, tvars, name='gradients')

      if (FLAGS.clip_gradients > 0):
        gradients, _ = tf.clip_by_global_norm(grads, FLAGS.clip_gradients)
      else:
        gradients = grads

      #calculated_learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, tf.compat.v1.train.get_global_step()+1, 100, 0.93, staircase=False)

      #effective_learning_rate = tf.Print(calculated_learning_rate, [calculated_learning_rate], "Calculated learning rate")

      #optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=effective_learning_rate)
      optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

      train_op = optimizer.apply_gradients(zip(gradients, tvars), global_step=tf.compat.v1.train.get_global_step())

      training_hooks = None
      if not FLAGS.use_tpu:
        logging_hook = tf.train.LoggingTensorHook({"loss": model.loss, "sigma": model.sigma, "gamma": model.gamma, "threshold": model.threshold, "step": tf.train.get_global_step()}, every_n_iter=1)
        training_hooks = [logging_hook]

      return tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
        mode, predictions=None, loss=model.loss, train_op=train_op, eval_metrics=None,
        export_outputs=None, scaffold_fn=None, host_call=None, training_hooks=training_hooks,
        evaluation_hooks=None, prediction_hooks=None)

#    elif mode == tf.estimator.ModeKeys.EVAL:
#
#      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
#        #predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
#        #accuracy = tf.metrics.accuracy(
#        #    labels=label_ids, predictions=predictions, weights=is_real_example)
#        loss = tf.metrics.mean(values=per_example_loss, weights=None)
#        return {
#            "eval_accuracy": 0,
#            "eval_loss": loss,
#        }
#
#      eval_metrics = (metric_fn,
#                      [per_example_loss, 0, 0, 0])
#      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
#          mode=mode,
#          loss=total_loss,
#          eval_metrics=eval_metrics,
#          scaffold_fn=None)

    else:
      spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions={'sigma': model.sigma,
                     'gamma': model.gamma,
                     'threshold': model.threshold
                  })
      return spec 

  return model_fn   

def main():
  tf.logging.set_verbosity(tf.logging.INFO)

  tpu_cluster_resolver = None

  if FLAGS.use_tpu:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=None,
      job_name='worker',
      coordinator_name=None,
      coordinator_address=None,
      credentials='default', 
      service=None,
      discovery_url=None
    )

  tpu_config = tf.compat.v1.estimator.tpu.TPUConfig(
    iterations_per_loop=FLAGS.iterations_per_loop, 
    num_cores_per_replica=FLAGS.num_tpu_cores,
    per_host_input_for_training=True 
  )

  run_config = tf.compat.v1.estimator.tpu.RunConfig(
    tpu_config=tpu_config,
    evaluation_master=None,
    session_config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True),
    master=None,
    cluster=tpu_cluster_resolver,
    **{
      'save_checkpoints_steps': FLAGS.save_checkpoints_steps,
      'tf_random_seed': FLAGS.random_seed,
      'model_dir': FLAGS.output_dir, 
      'keep_checkpoint_max': FLAGS.keep_checkpoint_max,
      'log_step_count_steps': FLAGS.log_step_count_steps
    }
  )

  estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
    model_fn=model_fn_builder(FLAGS.init_checkpoint, FLAGS.learning_rate, FLAGS.num_train_steps, FLAGS.use_tpu),
    use_tpu=FLAGS.use_tpu,
    train_batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.batch_size,
    predict_batch_size=FLAGS.batch_size,
    config=run_config,
    params={
        "n": FLAGS.n,
        "t": FLAGS.t,
        "q": FLAGS.q,
        "initializer_range": FLAGS.initializer_range
    })

  if FLAGS.action == 'TRAIN':
    estimator.train(input_fn=make_input_fn(FLAGS.train_file, is_training=True, drop_reminder=True), max_steps=FLAGS.num_train_steps)
  
  if FLAGS.action == 'PREDICT':
    predict_drop_remainder = True if FLAGS.use_tpu else False
    results = estimator.predict(input_fn=make_input_fn(FLAGS.train_file, is_training=False, drop_reminder=predict_drop_remainder))

    for prediction in results:
      sigma = prediction["sigma"]
      gamma = prediction["gamma"]
      threshold = prediction["threshold"]
      print ("sigma: ", sigma)
      print ("gamma: ", gamma)
      print ("threshold: ", threshold)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='gs://anomaly_detection/pot/output',
            help='Model directrory in google storage.')
    parser.add_argument('--init_checkpoint', type=str, default=None,
            help='This will be checkpoint from previous training phase.')
    parser.add_argument('--train_file', type=str, default='gs://anomaly_detection/pot/data/train/machine-1-1.tfrecords',
            help='Train file location in google storage.')
    parser.add_argument('--n', type=float, default=None,
            help='Total number of observer values.')
    parser.add_argument('--t', type=float, default=None,
            help='Initial threshold.')
    parser.add_argument('--q', type=float, default=1e-3,
            help='Probability that metrics will not exceed threshold.')
    parser.add_argument('--num_train_steps', type=int, default=1000,
            help='Number of steps to run trainer.')
    parser.add_argument('--iterations_per_loop', type=int, default=1000,
            help='Number of iterations per TPU training loop.')
    parser.add_argument('--save_checkpoints_steps', type=int, default=1000,
            help='Number of tensorflow checkpoint to keep.')
    parser.add_argument('--log_step_count_steps', type=int, default=1000,
            help='Number of step to write logs.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10,
            help='Number of tensorflow checkpoint to keep.')
    parser.add_argument('--batch_size', type=int, default=32,
            help='Batch size.')
    parser.add_argument('--dataset_reader_buffer_size', type=int, default=100,
            help='input pipeline is I/O bottlenecked, consider setting this parameter to a value 1-100 MBs.')
    parser.add_argument('--shuffle_buffer_size', type=int, default=24000,
            help='Items are read from this buffer.')
    parser.add_argument('--use_tpu', default=False, action='store_true',
            help='Train on TPU.')
    parser.add_argument('--tpu', type=str, default='node-1-15-2',
            help='TPU instance name.')
    parser.add_argument('--num_tpu_cores', type=int, default=8,
            help='Number of cores on TPU.')
    parser.add_argument('--tpu_zone', type=str, default='us-central1-c',
            help='TPU instance zone location.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
            help='Optimizer learning rate.')
    parser.add_argument('--clip_gradients', type=float, default=-1.,
            help='Clip gradients to deal with explosive gradients.')
    parser.add_argument('--random_seed', type=int, default=1234,
            help='Random seed to initialize values in a grath. It will produce the same results only if data and grath did not change in any way.')
    parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
    parser.add_argument('--action', default='PREDICT', choices=['TRAIN','EVALUATE','PREDICT'],
            help='An action to execure.')
    parser.add_argument('--initializer_range', type=float, default=0.02,
            help='.')

    FLAGS, unparsed = parser.parse_known_args()

    main()
