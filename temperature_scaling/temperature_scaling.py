import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp

def find_scaling_temperature(labels, logits, temp_range=(1e-5, 1e5)):
  """Find max likelihood scaling temperature using binary search.
  Args:
    labels: Integer labels (shape=[num_samples]).
    logits: Floating point softmax inputs (shape=[num_samples, num_classes]).
    temp_range: 2-tuple range of temperatures to consider.
  Returns:
    Floating point temperature value.
  """
  if not tf.executing_eagerly():
    raise NotImplementedError(
        'find_scaling_temperature() not implemented for graph-mode TF')
  if len(labels.shape) != 1:
    raise ValueError('Invalid labels shape=%s' % str(labels.shape))
  if len(logits.shape) not in (1, 2):
    raise ValueError('Invalid logits shape=%s' % str(logits.shape))
  if len(labels.shape) != 1 or len(labels) != len(logits):
    raise ValueError('Incompatible shapes for logits (%s) vs labels (%s).' %
                     (logits.shape, labels.shape))

  @tf.function(autograph=False)
  def grad_fn(temperature):
    """Returns gradient of log-likelihood WRT a logits-scaling temperature."""
    temperature *= tf.ones([])
    if len(logits.shape) == 1:
      dist = tfp.distributions.Bernoulli(logits=logits / temperature)
    elif len(logits.shape) == 2:
      dist = tfp.distributions.Categorical(logits=logits / temperature)
    nll = -dist.log_prob(labels)
    nll = tf.reduce_sum(nll, axis=0)
    grad, = tf.gradients(nll, [temperature])
    return grad

  tmin, tmax = temp_range
  return scipy.optimize.bisect(lambda t: grad_fn(t).numpy(), tmin, tmax)


def apply_temperature_scaling(temperature, probs):
  """Apply temperature scaling to an array of probabilities.
  Args:
    temperature: Floating point temperature.
    probs: Array of probabilities with probabilities over axis=-1.
  Returns:
    Temperature-scaled probabilities; same shape as input probs.
  """
  logits_t = inverse_softmax(probs).T / temperature
  return scipy.special.softmax(logits_t.T, axis=-1)


def inverse_softmax(x):
    """Inverse softmax over last dimension."""
    return np.log(x/x[...,:1])
