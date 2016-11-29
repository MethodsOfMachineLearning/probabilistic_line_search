#Probabilistic Line Search

This is a Python implementation of a [Probabilistic Line Search for Stochastic
Optimization][1] plus a TensorFlow interface that allows you to use the line
search to train your TensorFlow model.

## The Algorithm in a Nutshell
The probabilistic line search is an algorithm for the optimization of a
stochastic objective function F. Being at point x and having fixed a search
direction d, it maintains a Gaussian process model for the one-dimensional
function f(t) = F(x + td). This function and its derivative are evaluated at
(possibly multiple) step sizes t, updating the GP after each observation. This
is repeated until a _probabilistic belief_ over a quality criterion of the step
size, implied by the GP, exceeds a certain threshold.

## Installation

This an early development version. No installation is required, just clone this
git repositiory to your machine.

Requirements:
- tensorflow (0.10.0 is known to work)
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
opt_interface.minimize(losses, var_list)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
opt_interface.register_session(sess)
opt_ls = ProbLSOptimizer(opt_interface)
opt_ls.prepare(feed_dict_if_applicable)

for i in range(num_steps):
  ...
  opt_ls.proceed(feed_dict_if_applicable)
```

*Why is That?*
This is due to the fact that the implementation actually consists of two components:
- A line search class (``ProbLSOptimizer``). It performs the line search, i.e. it gathers observations, updates the GP model, decides where to evaluate next, et cetera. The ``ProbLSOptimizer`` takes as argument a ``func`` object that is the "interface" to the objective function. It assumes that this interface has certain methods for evaluating at new points or accepting the current one; see the documentation of ``ProbLSOptimizer`` for details.
- The TensorFlow interface ``ProbLSOptimizerSGDInterface``. This can be used as the ``func`` argument for a ``ProbLSOptimizer`` and provides the necessary interface functions to use the line search to train your TensorFlow model.

Hence, to use the line search to train a TensorFlow model, we first import the two aformentioned classes.

```python
from probls.tensorflow_interface.interface_sgd import ProbLSOptimizerSGDInterface
from probls.line_search import ProbLSOptimizer
```

The function interface is set up by

```python
opt_interface = ProbLSOptimizerSGDInterface()
opt_interface.minimize(losses, var_list)
```

(Note that you have to pass the vector of ``losses``, instead of an aggregated
mean loss!) Next, we start a TensorFlow session and pass it to the interface

```python
sess = tf.Session()
opt_interface.register_session(sess)
```

``opt_interface`` now uses this session to perform the evaluations and, thus, can work as a self-contained
interface to the TensorFlow model. We now initialize the line search object, passing ``opt_interface`` as its objective.

```python
opt_ls = ProbLSOptimizer(opt_interface)
```

``opt_ls`` has two methods that are of interest for the end-user.
- ``opt_ls.prepare(*args)`` has to be called once to initialize the line search.
- ``opt_ls.proceed(*args)`` proceeds one step in the line search (i.e. one
function evaluation). We call this method for however many steps we want to train the model.
For both functions, ``*args`` are arguments passed to the objective ``func``, which can be used to pass a ``feed_dict`` to your ``opt_interface`` if you are feeding data via placeholders.


See the ``examples/`` folder for working demo scripts with placeholders (MNIST) and without (CIFAR-10).

## Quick Guide to this Implementation

### Line Search

The ``ProbLSOptimizer`` class is implemented in ``probls.line_search``. A line search object
excepts a ``func`` argument, that acts as the interface to the objective function.
It is assumend that it has three methods:
- ``f, df, fvar, dfvar = func.adv_eval(dt, *pass_to_func_args)`` to proceed along the current search
  direction by an increment ``dt``, returning function value, projected gradient
  and variance estimates for both.
- ``f, df, fvar, dfvar = func.accept()`` to accept the current step size,
  returning function value, projected gradients and an estimate of the variance
  of these two quantities.
- ``f, df, fvar, dfvar = func.prepare(*pass_to_func_args)`` to prepare the interface returning an
  initial observation of function value and gradient, as well as the variances.
If the function interface takes additional arguments (e.g. a feed_dict with a
batch of data in for the TensorFlow interface), those are passed as positional
arguments ``*pass_to_func_args``. The line search algorithm "communicates" with
the objective function exclusively via these three methods.

Other than that, ``ProbLSOptimizer`` has no required arguments (most notably, no learning rate).
The remaining arguments are design parameters of the line search algorithm.

The Gaussian process functionality needed in the line search is implemented in
``probls.gaussian_process``. It implements one-dimensional Gaussian process regression with an integrated
Wiener process kernel that uses observations of both the function value and the
derivative. For details, see the documentation of the ``probls.gaussian_process.ProbLSGaussianProcess`` class.

### TensorFlow Interface

The TensorFlow interface is implemented in ``probls.tensorflow_interface.interface_sgd``.
It can act as the ``func`` argument of the ``ProbLSOptimizer``, providing the
desired interface to the objective function defined by your TensorFlow model.

A crucial part of the line search are within-batch estimated of the function 
value and the gradient, see equations (17) and (18) in [1]. Computing the gradient
variance is a little tricky; details can be found in the following note: (coming soon...)

To be continued...



[1]: https://arxiv.org/abs/1502.02846
