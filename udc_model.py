import tensorflow as tf
import sys

def get_id_feature(features, key, len_key, max_len):
  ids = features[key]
  ids_len = tf.squeeze(features[len_key], [1])
  ids_len = tf.minimum(ids_len, tf.constant(max_len, dtype=tf.int64))
  return ids, ids_len

def create_train_op(loss, hparams):
  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=hparams.learning_rate,
      clip_gradients=10.0,
      optimizer=hparams.optimizer)
  return train_op


def create_model_fn(hparams, model_impl):

  def model_fn(features, targets, mode):
    context, context_len = get_id_feature(
        features, "context", "context_len", hparams.max_context_len)
    context_token_len_avg, _ = get_id_feature(
        features, "context_token_len_avg", "context_len", hparams.max_context_len)
    context_nums, _ = get_id_feature(
        features, "context_nums", "context_len", hparams.max_context_len)
    
    utterance, utterance_len = get_id_feature(
        features, "utterance", "utterance_len", hparams.max_utterance_len)
    utterance_token_len_avg, _ = get_id_feature(
        features, "utterance_token_len_avg", "context_len", hparams.max_context_len)
    utterance_nums, _ = get_id_feature(
        features, "utterance_nums", "context_len", hparams.max_context_len)
    
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      probs, loss = model_impl(
          hparams,
          mode,
          context,
          context_len,
          context_token_len_avg,
          context_nums,
          utterance,
          utterance_len,
          utterance_token_len_avg,
          utterance_nums,
          targets)
      train_op = create_train_op(loss, hparams)
      return probs, loss, train_op

    if mode == tf.contrib.learn.ModeKeys.INFER:

      all_contexts = [context]
      all_context_lens = [context_len]
      all_context_token_len_avgs = [context_token_len_avg]
      all_context_nums = [context_nums]
      all_utterances = [utterance]
      all_utterance_lens = [utterance_len]
      all_utterance_token_len_avgs  = [utterance_token_len_avg]
      all_utterance_nums = [utterance_nums]

      for i in range(1,features["len"]):
        distractor, distractor_len, distractor_token_len_avg, distractor_nums = get_id_feature(features,
            "utterance_{}".format(i),
            "utterance_{}_len".format(i),
            "utterance_{}_token_len_avg".format(i),
            "utterance_{}_nums".format(i),
            hparams.max_utterance_len)
        all_contexts.append(context)
        all_context_lens.append(context_len)
        all_context_token_len_avgs.append(context_token_len_avg)
        all_context_nums.append(context_nums)
        all_utterances.append(distractor)
        all_utterance_lens.append(distractor_len)
        all_utterance_token_len_avgs.append(distractor_token_len_avg)
        all_utterance_nums.append(distractor_nums)

      probs, loss = model_impl(
          hparams,
          mode,
          tf.concat(0, all_contexts),
          tf.concat(0, all_context_lens),
          tf.concat(0, all_context_token_len_avgs),
          tf.concat(0, all_context_nums),
          tf.concat(0, all_utterances),
          tf.concat(0, all_utterance_lens),
          tf.concat(0, all_utterance_token_len_avgs),
          tf.concat(0, all_utterance_nums),
          None)

      split_probs = tf.split(0, features["len"], probs)
      probs = tf.concat(1, split_probs)

      return probs, 0.0, None

    if mode == tf.contrib.learn.ModeKeys.EVAL:
      batch_size = targets.get_shape().as_list()[0]

      # We have 10 exampels per record, so we accumulate them
      
      all_contexts = [context]
      all_context_lens = [context_len]
      all_context_token_len_avgs = [context_token_len_avg]
      all_context_nums = [context_nums]
      all_utterances = [utterance]
      all_utterance_lens = [utterance_len]
      all_utterance_token_len_avgs  = [utterance_token_len_avg]
      all_utterance_nums = [utterance_nums]
      
      all_targets = [tf.ones([batch_size, 1], dtype=tf.int64)]
      
      

      for i in range(9):       
        distractor, distractor_len, distractor_token_len_avg, distractor_nums = get_id_feature(features,
            "utterance_{}".format(i),
            "utterance_{}_len".format(i),
            "utterance_{}_token_len_avg".format(i),
            "utterance_{}_nums".format(i),
            hparams.max_utterance_len)
        all_contexts.append(context)
        all_context_lens.append(context_len)
        all_context_token_len_avgs.append(context_token_len_avg)
        all_context_nums.append(context_nums)
        all_utterances.append(distractor)
        all_utterance_lens.append(distractor_len)
        all_utterance_token_len_avgs.append(distractor_token_len_avg)
        all_utterance_nums.append(distractor_nums)
        
        all_targets.append(
          tf.zeros([batch_size, 1], dtype=tf.int64)
        )

      probs, loss = model_impl(
          hparams,
          mode,
          tf.concat(0, all_contexts),
          tf.concat(0, all_context_lens),
          tf.concat(0, all_context_token_len_avgs),
          tf.concat(0, all_context_nums),
          tf.concat(0, all_utterances),
          tf.concat(0, all_utterance_lens),
          tf.concat(0, all_utterance_token_len_avgs),
          tf.concat(0, all_utterance_nums))

      split_probs = tf.split(0, 10, probs)
      shaped_probs = tf.concat(1, split_probs)

      # Add summaries
      tf.histogram_summary("eval_correct_probs_hist", split_probs[0])
      tf.scalar_summary("eval_correct_probs_average", tf.reduce_mean(split_probs[0]))
      tf.histogram_summary("eval_incorrect_probs_hist", split_probs[1])
      tf.scalar_summary("eval_incorrect_probs_average", tf.reduce_mean(split_probs[1]))

      return shaped_probs, loss, None

  return model_fn
