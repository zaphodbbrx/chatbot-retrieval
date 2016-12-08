import time
import sys
import numpy as np
import tensorflow as tf
import udc_model
import udc_hparams
from models.dual_encoder import dual_encoder_model
import pandas as pd

tf.flags.DEFINE_string("model_dir", None, "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", "./data/vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

# Load vocabulary
vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
  FLAGS.vocab_processor_file)

# Load data for predict
test_df = pd.read_csv("./data/test.csv")
elementId = 0
INPUT_CONTEXT = test_df.Context[elementId]
POTENTIAL_RESPONSES = test_df.iloc[elementId,1:].values

def get_features(context, utterance):
  context_matrix = np.array(list(vp.transform([context])))
  utterance_matrix = np.array(list(vp.transform([utterance])))
  context_len = len(context.split(" "))
  utterance_len = len(utterance.split(" "))
  features = {
    "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
    "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
    "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
    "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
  }
  return features, None

if __name__ == "__main__":
  # tf.logging.set_verbosity(tf.logging.INFO)
  hparams = udc_hparams.create_hparams()
  model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)

  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)

  # Ugly hack, seems to be a bug in Tensorflow
  # estimator.predict doesn't work without this line
  estimator._targets_info = tf.contrib.learn.estimators.tensor_signature.TensorSignature(tf.constant(0, shape=[1,1]))

  results = []

  starttime = time.time()

  for r in POTENTIAL_RESPONSES:
    prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, r),as_iterable=True)
    results.append(next(prob)[0])
    # print("[ ] {}: {:g}".format(r, next(prob)[0]))
    # print("{}: {:g}".format(r, prob[0,0]))

  endtime = time.time()

  results = np.array(results)
  answerId = results.argmax(axis=0)

  print("[Time]", endtime - starttime,"sec")
  print("[Context       ] {}".format(INPUT_CONTEXT))
  print("[Results value ]",results)
  print("[answer        ]", POTENTIAL_RESPONSES[answerId])
  if not answerId==0:
      print("[right responce]", POTENTIAL_RESPONSES[0])
