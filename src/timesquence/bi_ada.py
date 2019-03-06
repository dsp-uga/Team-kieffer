from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import load_model
import scipy.io as sio
import functools
import numpy as np
import random
import adanet
from adanet.examples import simple_dnn
import tensorflow as tf

# The random seed to use.
RANDOM_SEED = 42
global width, length
width =  100
length = 28

def load_data():
    global width,length
    summay_set = []
   
    for line in open("../env_label/b6b9c741277310c36ac57d971da832689cfbd06bb537fb2aa3e347f44a25e872.csv"):
      Info = line.strip().split("\t")
      single_data = Info[1].split(",")
      

      for i in range(len(single_data)):
        single_data[i] = int(single_data[i])

      single_label = int(Info[0])
      if single_label == 2:
        single_label = 1
      elif single_label == 1:
        single_label = 0
      else:
        single_label = 0
    
      #print single_label,single_data
      summay_set.append([single_data,single_label])
    return summay_set



#load the dataset
mnist=load_data()
x_train,y_train,x_test,y_test = [],[],[],[]
for k in mnist:
    if random.random() < 0.8:
        x_train.append(k[0])
        y_train.append(k[1])
    else:
        x_test.append(k[0])
        y_test.append(k[1])

#print (y_test,y_test)

FEATURES_KEY = "images"


def generator(images, labels):
  """Returns a generator that returns image-label pairs."""

  def _gen():
    for image, label in zip(images, labels):
      yield image, label

  return _gen


def preprocess_image(image, label):
  global width,length 
  """Preprocesses an image for an `Estimator`."""
  # First let's scale the pixel values to be between 0 and 1.
  image = image / 255
  # Next we reshape the image so that we can apply a 2D convolution to it.
  image = tf.reshape(image, [width, 1])
  # Finally the features need to be supplied as a dictionary.
  features = {FEATURES_KEY: image}
  return features, label


def input_fn(partition, training, batch_size):
  """Generate an input_fn for the Estimator."""

  def _input_fn():
    global width ,length
    if partition == "train":
      dataset = tf.data.Dataset.from_generator(
          generator(x_train, y_train), (tf.float32, tf.int32), ((width ), ()))
    else:
      dataset = tf.data.Dataset.from_generator(
          generator(x_test, y_test), (tf.float32, tf.int32), ((width), ()))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    if training:
      dataset = dataset.shuffle(10 * batch_size, seed=RANDOM_SEED).repeat()

    dataset = dataset.map(preprocess_image).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

  return _input_fn

# The number of classes.
NUM_CLASSES = 2

# We will average the losses in each mini-batch when computing gradients.
loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE

# A `Head` instance defines the loss function and metrics for `Estimators`.
head = tf.contrib.estimator.binary_classification_head(
     loss_reduction=loss_reduction)

# Some `Estimators` use feature columns for understanding their input features.
feature_columns = [
    tf.feature_column.numeric_column(FEATURES_KEY, shape=[width, 1])
]

# Estimator configuration.
config = tf.estimator.RunConfig(
    save_checkpoints_steps=50000,
    save_summary_steps=50000,
    tf_random_seed=RANDOM_SEED)




class SimpleCNNBuilder(adanet.subnetwork.Builder):
  """Builds a CNN subnetwork for AdaNet."""

  def __init__(self, learning_rate, max_iteration_steps, seed):
    """Initializes a `SimpleCNNBuilder`.

    Args:
      learning_rate: The float learning rate to use.
      max_iteration_steps: The number of steps per iteration.
      seed: The random seed.

    Returns:
      An instance of `SimpleCNNBuilder`.
    """
    self._learning_rate = learning_rate
    self._max_iteration_steps = max_iteration_steps
    self._seed = seed

  def build_subnetwork(self,
                       features,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
    """See `adanet.subnetwork.Builder`."""
    print(features['images'])
    images = features['images']
    kernel_initializer = tf.keras.initializers.he_normal(seed=self._seed)
    x = tf.layers.conv1d(
        images,
        filters=16,
        kernel_size=20,
        padding="same",
        activation="relu",
        kernel_initializer=kernel_initializer)
    x = tf.layers.max_pooling1d(x, pool_size=1, strides=1)
    x = tf.layers.conv1d(
        images,
        filters=32,
        kernel_size=20,
        padding="same",
        activation="relu",
        kernel_initializer=kernel_initializer)
    x = tf.layers.max_pooling1d(x, pool_size=1, strides=1)
    x = tf.layers.conv1d(
        images,
        filters=64,
        kernel_size=20,
        padding="same",
        activation="relu",
        kernel_initializer=kernel_initializer)
    x = tf.layers.max_pooling1d(x, pool_size=1, strides=1)


    x = tf.layers.flatten(x)
    
    x = tf.layers.dense(
        x, units=128, activation="relu", kernel_initializer=kernel_initializer)

    # The `Head` passed to adanet.Estimator will apply the softmax activation.
    logits = tf.layers.dense(
        x, units=1, activation=None, kernel_initializer=kernel_initializer)

    # Use a constant complexity measure, since all subnetworks have the same
    # architecture and hyperparameters.
    complexity = tf.constant(1)

    return adanet.Subnetwork(
        last_layer=x,
        logits=logits,
        complexity=complexity,
        persisted_tensors={})

  def build_subnetwork_train_op(self, 
                                subnetwork, 
                                loss, 
                                var_list, 
                                labels, 
                                iteration_step,
                                summary, 
                                previous_ensemble=None):
    """See `adanet.subnetwork.Builder`."""

    # Momentum optimizer with cosine learning rate decay works well with CNNs.
    learning_rate = tf.train.cosine_decay(
        learning_rate=self._learning_rate,
        global_step=iteration_step,
        decay_steps=self._max_iteration_steps)
    optimizer = tf.train.MomentumOptimizer(learning_rate, .9)
    # NOTE: The `adanet.Estimator` increments the global step.
    return optimizer.minimize(loss=loss, var_list=var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    """See `adanet.subnetwork.Builder`."""
    return tf.no_op("mixture_weights_train_op")

  @property
  def name(self):
    """See `adanet.subnetwork.Builder`."""
    return "simple_cnn"

class SimpleCNNGenerator(adanet.subnetwork.Generator):
  """Generates a `SimpleCNN` at each iteration.
  """

  def __init__(self, learning_rate, max_iteration_steps, seed=None):
    """Initializes a `Generator` that builds `SimpleCNNs`.

    Args:
      learning_rate: The float learning rate to use.
      max_iteration_steps: The number of steps per iteration.
      seed: The random seed.

    Returns:
      An instance of `Generator`.
    """
    self._seed = seed
    self._dnn_builder_fn = functools.partial(
        SimpleCNNBuilder,
        learning_rate=learning_rate,
        max_iteration_steps=max_iteration_steps)

  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
    """See `adanet.subnetwork.Generator`."""
    seed = self._seed
    # Change the seed according to the iteration so that each subnetwork
    # learns something different.
    if seed is not None:
      seed += iteration_number
    return [self._dnn_builder_fn(seed=seed)]

#@title Parameters
LEARNING_RATE = 0.05  #@param {type:"number"}
TRAIN_STEPS = 10000  #@param {type:"integer"}
BATCH_SIZE = 600  #@param {type:"integer"}
ADANET_ITERATIONS = 3  #@param {type:"integer"}

max_iteration_steps = TRAIN_STEPS // ADANET_ITERATIONS
estimator = adanet.Estimator(
    head=head,
    subnetwork_generator=SimpleCNNGenerator(
        learning_rate=LEARNING_RATE,
        max_iteration_steps=max_iteration_steps,
        seed=RANDOM_SEED),
    max_iteration_steps=max_iteration_steps,
    evaluator=adanet.Evaluator(
        input_fn=input_fn("train", training=False, batch_size=BATCH_SIZE),
        steps=None),
    report_materializer=adanet.ReportMaterializer(
        input_fn=input_fn("train", training=False, batch_size=BATCH_SIZE),
        steps=None),
    adanet_loss_decay=.99,
    config=config)

results, _ = tf.estimator.train_and_evaluate(
    estimator,
    train_spec=tf.estimator.TrainSpec(
        input_fn=input_fn("train", training=True, batch_size=BATCH_SIZE),
        max_steps=TRAIN_STEPS),
    eval_spec=tf.estimator.EvalSpec(
        input_fn=input_fn("test", training=False, batch_size=BATCH_SIZE),
        steps=None))
print("Accuracy:", results["accuracy"])
print("Loss:", results["average_loss"])

#print the result out 
a = []
predictions = estimator.predict(input_fn=input_fn("predict", training=False, batch_size=1))
for i, val in enumerate(predictions):


    predicted_class = val['class_ids'][0]
    a.append(predicted_class)

print(a)
