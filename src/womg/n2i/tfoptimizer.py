""" Tensorflow alternative implementation of Word2Vec;
    by default we generate non-negative embeddings.
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import math
import random

def generate_batch(walks, batch_size, num_skips, window):
    """ Generator that, given a list of walks (each walk being a list of indexes),
        yields the `batch, labels` pairs needed to train Word2vec.

        Args:
            batch_size: Number of pairs (target, context) in each example.
            num_skips: Number of examples for each target node.
            window: Length of the context.
    """
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * window
    span = 2 * window + 1  # [ window target window ]
    for walk in walks:
        for data_index in range(0, len(walk), span):
            batch = np.ndarray(shape=(batch_size), dtype=np.int32)
            labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
            buffer = walk[data_index:data_index + span]
            for i in range(batch_size // num_skips):
                context_words = [w for w in range(span) if w != window and w < len(buffer)]
                if num_skips > len(context_words):
                    words_to_use = random.choices(context_words, k=num_skips)
                else:
                    words_to_use = random.sample(context_words, num_skips)
                for j, context_word in enumerate(words_to_use):
                    print(len(buffer))
                    print(window)
                    batch[i * num_skips + j] = buffer[window]
                    labels[i * num_skips + j, 0] = buffer[context_word]
            yield batch, labels

EPSILON = 1E-04
def beta_kl_divergence(sample, prior_alpha, prior_beta):
    mu = tf.math.reduce_mean(sample)
    var = tf.reduce_mean(tf.squared_difference(sample, mu)) + EPSILON
    observed_alpha = ((1. - mu) / var - (1. / (mu + EPSILON))) * tf.square(mu)
    observed_beta = observed_alpha * (1. / (mu + EPSILON) - 1)
    return (
        tf.lgamma(prior_alpha + prior_beta)
        - (tf.lgamma(prior_alpha) + tf.lgamma(prior_beta))
        - (tf.lgamma(observed_alpha + observed_beta + EPSILON))
        + (tf.lgamma(observed_alpha + EPSILON) + tf.lgamma(observed_beta + EPSILON))
        + (prior_alpha - observed_alpha) * (tf.digamma(prior_alpha) - tf.digamma(prior_alpha + prior_beta))
        + (prior_beta - observed_beta) * (tf.digamma(prior_beta) - tf.digamma(prior_alpha + prior_beta))
    )

class Word2vec:
    def __init__(self,
            number_of_nodes,
            alpha=0.025,
            embedding_size=128,
            cost_of_negatives=1.,
            batch_size=100,
            num_sampled=5, # Same default as Gensim.
            beta=0,
            prior='half_norm'
    ):
        """
        Build the TensorFlow operation graph.

        Args:
            number_of_nodes: vocabulary size, aka number of nodes in the graph.
            alpha: Learning rate.
            embedding_size: Number of dimensions of the embedding.
            cost_of_negatives: Multiplier for the loss function to avoid negatives. Use 0 to allow them.
            num_sampled: Number of negative examples to sample for NCF.
        """
        batch_size = None # Will use whatever batch size will be passed at runtime.
        self.embedding_size = embedding_size

        self.embeddings = tf.Variable(
            tf.random_uniform([number_of_nodes, embedding_size], 0, 1.0),
            name='embeddings'
        )

        # nce loss
        nce_weights = tf.Variable(
          tf.truncated_normal([number_of_nodes, embedding_size],
                              stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([number_of_nodes]))


        self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size], name='input')
        self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1], name='labels')
        embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)


        self.nce_loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                             biases=nce_biases,
                             labels=self.train_labels,
                             inputs=embed,
                             num_sampled=num_sampled,
                             num_classes=number_of_nodes))
        # non-negative constraint
        if cost_of_negatives > 0:
            self.negatives_loss = tf.nn.softsign(
                tf.reduce_sum(- tf.minimum(0., self.embeddings)))
            self.loss = self.nce_loss + cost_of_negatives * self.negatives_loss
        else:
            self.loss = self.nce_loss


        # beta-vae constraint
        mu = tf.reduce_mean(self.embeddings, axis=0, name='mu')
        log_std = tf.log(tf.math.reduce_std(self.embeddings, axis=0, name='std'))
        #print('mu: ', mu.shape, 'log_std: ', log_std.shape)

        if prior == 'half_norm':
            # kl with half-normal distribution
            kl = 0.5 * (tf.math.square(tf.exp(log_std) - 1)) - log_std
        if prior == 'norm':
            # kl with normal distribution
            kl = 0.5 * (-log_std + tf.square(mu - 10) + tf.exp(log_std) - 1)
        if prior == 'beta':
            prior_alpha = np.array([2. for _ in range(embedding_size)], dtype=np.float32)
            prior_beta = np.array([2. for _ in range(embedding_size)], dtype=np.float32)
            kl = beta_kl_divergence(self.embeddings, prior_alpha, prior_beta)


        kl_penalty = tf.reduce_sum(kl)

        #print('loss_before ', self.loss)
        self.loss += beta * kl_penalty
        #print('loss_after ', self.loss)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(
            self.loss
        )


    def run(self, walks, batch_size=100, iiter=4, num_skips=None, window=5, verbose=True):
        """ Run Word2Vec on the given random walks, and returns the embeddings.

        Args:
            walks: list of random walks generated by `node2vec`, each walk being a list of indexes.
            batch_size: Number of pairs (target, context) in each example.
            iiter: number of epochs.
            num_skips: Number of examples for each target node. If `None`, it uses `2 * window`.
            window: Length of the context.
            verbose: if True, prints a `tqdm` ProgressBar.

        Returns:
            The embeddings, as a number_of_nodes x embedding_size numpy matrix.
        """
        # Preprocess args.
        if num_skips is None:
            num_skips = 2 * window
        span = 2 * window + 1

        # Initialize Training.
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            if verbose:
                examples_per_epoch = sum(math.ceil(len(walk) / span) for walk in walks)
                pbar = tqdm(total=(iiter * examples_per_epoch), desc="Learning embeddings")

            # Training.
            for _epoch in range(iiter):
                for inputs, labels in generate_batch(walks,
                                                     batch_size=batch_size,
                                                     num_skips=num_skips,
                                                     window=window):
                    #print('inputs ', inputs)
                    feed_dict = {self.train_inputs: inputs, self.train_labels: labels}
                    _, cur_loss = session.run([
                        self.optimizer, self.loss
                    ], feed_dict=feed_dict)
                    if verbose:
                        pbar.update(1)
                        pbar.set_postfix(loss=cur_loss)

            if verbose: pbar.close()

            # Returns embeddings.
            return self.embeddings.eval()
