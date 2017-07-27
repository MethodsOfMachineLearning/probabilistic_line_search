# Probabilistic Line Search

This is a Python implementation of a [Probabilistic Line Search for Stochastic
Optimization][1] plus a TensorFlow interface that allows you to use the line
search to train your TensorFlow model. **Please note: this is a development version with multiple experimental changes compared to the original paper!**

## The Algorithm in a Nutshell
The probabilistic line search is an algorithm for the optimization of a
stochastic objective function F. Being at point x and having fixed a search
direction d, it maintains a Gaussian process model for the one-dimensional
function f(t) = F(x + td). This function and its derivative are evaluated at
(possibly multiple) step sizes t, updating the GP after each observation. This
is repeated until a _probabilistic belief_ over a quality criterion of the step
size, implied by the GP, exceeds a certain threshold.

## Installation

No installation is required, just clone this git repositiory to your machine.

Requirements:
- tensorflow (0.12.0 is known to work)
- numpy (1.11.2 is known to work)
- scipy (0.13.3 is known to work)
- Some of the demo scripts require additional packages, like sys, os, matplotlib
  et cetera.

## Usage

The built-in TensorFlow optimizers are used roughly like this

```python
var_list = ...
losses = ... # A vector of losses, one for each example in the batch

loss = tf.mean(losses)
opt = tf.train.GradientDescentOptimizer(learning_rate)
sgd_step = opt.minimize(loss)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(num_steps):
  ...
  sess.run(sgd_step, feed_dict_if_applicable)
```

Usage is slightly different for the probabilistic line search optimizer, but its only five additional lines of code:

```python
from probls.tensorflow_interface.interface_sgd import ProbLSOptimizerSGDInterface
from probls.line_search import ProbLSOptimizer

var_list = ...
losses = ... # A vector of losses, one for each example in the batch

opt_interface = ProbLSOptimizerSGDInterface()
opt_interface.minimize(losses, var_list) # Note that we pass losses, not an aggregate mean loss
sess = tf.Session()
sess.run(tf.initialize_all_variables())
opt_interface.register_session(sess)
opt_ls = ProbLSOptimizer(opt_interface)
opt_ls.prepare(feed_dict_if_applicable)

for i in range(num_steps):
  ...
  opt_ls.proceed(feed_dict_if_applicable)
```

The effects of these individual commands will become clear in the next section.
See the ``examples/`` folder for working demo scripts.


## Quick Guide to this Implementation

This implementation consists of two major components:
- A line search class (``ProbLSOptimizer``). It performs the line search, i.e. it gathers observations, updates the GP model, decides where to evaluate next, et cetera. The ``ProbLSOptimizer`` takes as argument a ``func`` object that is the "interface" to the objective function. It assumes that this interface has certain methods for evaluating at new points or accepting the current one; see below.
- The TensorFlow interface ``ProbLSOptimizerSGDInterface``. This can be used as the ``func`` argument for a ``ProbLSOptimizer`` and provides the necessary interface to use the line search to train your TensorFlow model.

### Line Search

The ``ProbLSOptimizer`` class is implemented in ``probls.line_search``. It
excepts a ``func`` argument which acts as the interface to the objective function.
It is assumend that ``func`` has three methods:
- ``f, df, fvar, dfvar = func.adv_eval(dt, *args)`` to proceed along the current search
  direction by an increment ``dt``, returning function value ``f``, projected gradient ``df``
  and variance estimates for both (``fvar, dfvar``).
- ``f, df, fvar, dfvar = func.accept()`` to accept the current step size,
  returning function value, projected gradients and an estimate of the variance
  of these two quantities (``df`` and ``dfvar`` with respect to the new search direction).
- ``f, df, fvar, dfvar = func.prepare(*args)`` to prepare the interface returning an
  initial observation.

``*args`` are additional positional arguments, e.g.. an optional feed_dict in the case the TensorFlow interface; see below.
The line search algorithm "communicates" with the objective function exclusively via these three methods.

Other than ``func``, ``ProbLSOptimizer`` has no required arguments, most notably, no learning rate!
The remaining arguments are design parameters of the line search algorithm. See the docstring of ``ProbLSOptimizer`` a description of these parameters.

``opt_ls`` has two methods that are of interest for the end-user.
- ``opt_ls.prepare(*pass_to_func_args)`` has to be called once to initialize the line search.
- ``opt_ls.proceed(*pass_to_func_args)`` proceeds one step in the line search (i.e. one
function evaluation). We call this method for however many steps we want to train the model. This is where
the actual line search happens, so check out its code (and that of the subroutines it calls) to get an idea of what is going on!

The Gaussian process functionality needed in the line search is outsourced to
``probls.gaussian_process``. It implements one-dimensional Gaussian process regression with an integrated
Wiener process kernel that uses observations of both the function value and the
derivative. For details, see the docstring of the ``ProbLSGaussianProcess`` class.

### TensorFlow Interface

The TensorFlow interface ``ProbLSOptimizerSGDInterface`` is implemented in ``probls.tensorflow_interface.interface_sgd``.
It inherits from ``tf.train.Optimizer`` and implements the necessary functionality to serve as the ``func`` argument of the ``ProbLSOptimizer``, providing the
desired interface to the objective function defined by your TensorFlow model.
Its ``minimize(losses, var_list)`` method adds to sets of operations to the TensorFlow graph:
- ``adv_eval_op``
  Advance along the current search direction, compute the loss,
  the gradients and variances of both. Gradient and its variance are stored
  in slot variables. Return the loss ``f``, projected gradient ``df``,
  variance of the loss fvar, and variance of the projected gradient dfvar
- ``accept_op``:
  Accept the current point. Set its gradient as the new search direction.
  Returns f, df fvar and dfvar, where df and dfvar are now with respect to this new search direction.

In order for the ``ProbLSOptimizerSGDInterface`` object to work as a self-contained
interface that can perform function/gradient evaluations, you have to pass it a
TensorFlow session via its ``register_session(sess)`` method. After that, the interface is
ready to go and provides the three aforementioned methods ``adv_eval(dt, optional_feed_dict)``, ``accept()`` and ``prepare(optional_feed_dict)``.

A crucial part of the line search are within-batch estimates of the variance of the function 
value and the gradient, see equations (17) and (18) in the [paper][1]. The variance
of the objective is easily computed given the individual loss values for the examples
in the batch. That is why we pass the vector of ``losses``, instead of a mean ``loss``.
Computing the gradient variance is a little tricky; a detailed explanation can be found in this [note][2].
For the implementation, see ``probls.tensorflow_interface.gradient_moment``.

[1]: https://arxiv.org/abs/1502.02846
[2]: https://drive.google.com/open?id=0B0adgqwcMJK5aDNaQ2Q4ZmhCQzA
