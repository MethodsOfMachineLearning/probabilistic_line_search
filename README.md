#Probabilistic Line Search

This is a Python implementation of a [Probabilistic Line Search for Stochastic
Optimization][1] with a TensorFlow interface.

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
- tensorflow (Version 0.10.0 is known to work)
- numpy (1.11.2 is known to work)
- scipy (0.13.3 is known to work)
- Some of the demo scripts require additional packages, like sys, os, matplotlib
  et cetera.

## Usage

The built-in tensorflow optimizer are used roughly like this

```python
var_list = ...
losses = ... # A vector of losses, one for each example in the batch

loss = tf.mean(losses)
opt = tf.train.GradientDescentOptimizer(learning_rate)
sgd_step = opt.minimize(loss)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(num_steps):
  sess.run(sgd_step)
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
  opt_ls.proceed(feed_dict_if_applicable)
```

This is due to the fact that the implementation actually consists of two components:
- A line search class (``ProbLSOptimizer``). It performs the line search, i.e. it gathers observations, updates the GP model, decides where to evaluate next, et cetera. The ``ProbLSOptimizer`` takes as argument a ``func`` object that is the "interface" to the objective function. It assumes that this interface has certain methods for evaluating at new points or accepting the current one; see the documentation of ``ProbLSOptimizer`` for details.
- The TensorFlow interface ``ProbLSOptimizerSGDInterface``. This can be used as the ``func`` argument for a ``ProbLSOptimizer`` and provides the necessary interface functions to use the line search to train your TensorFlow model.

Let's work our way through the above code snippet. First, we import the two aformentioned classes.

```python
from probls.tensorflow_interface.interface_sgd import ProbLSOptimizerSGDInterface
from probls.line_search import ProbLSOptimizer
```

The function interface is set up by (note that you have to pass the vector of ``losses``, instead of an aggregated
mean loss)

```python
opt_interface = ProbLSOptimizerSGDInterface()
opt_interface.minimize(losses, var_list)
```

You then have to start a TensorFlow session and pass it to the interface

```python
sess = tf.Session()
opt_interface.register_session(sess)
```

``opt_interface`` now takes control of the session and uses it to perform the evaluations in order to work as a self-contained
interface to TensorFlow. You can now initialize the line search object, passing ``opt_interface`` as its objective.

```python
opt_ls = ProbLSOptimizer(opt_interface)
```

``opt_ls`` has two methods that are of interest for the end-user.
- ``opt_ls.prepare(*args)`` has to be called once to initialize the line search.
- ``opt_ls.proceed(*args)`` proceeds one step in the line search (i.e. one
function evaluation).
For both functions, ``*args`` are arguments passed to the ``opt_interface``, which can be used to pass a ``feed_dict`` (if you are feeding data via placeholders).


See the ``examples/`` for working demo scripts.

## Guide to this Implementation

Coming soon...



[1]: https://arxiv.org/abs/1502.02846
