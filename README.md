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

Usage is slightly different for the probabilistic line search optimizer, since
the TensorFlow interface (which is a TensorFlow Optimizer) and the line search
algorithm itself (the "outer loop") are separated in this development version.

Import both

```python
from probls.tensorflow_interface.interface_sgd import ProbLSOptimizerSGDInterface
from probls.line_search import ProbLSOptimizer
```

Set up your model as usually and use the interface as follows:

```python
opt_interface = ProbLSOptimizerSGDInterface()
opt_interface.minimize(losses, var_list)
```

Note that you have to pass the vector of ``losses``, instead of an aggregated
mean loss! You then have to start a TensorFlow session and pass it to the
interface

```python
sess = tf.Session()
opt_interface.register_session(sess)
```

``opt_interface`` now takes control of the session and works a self-contained
interface to TensorFlow. You can now initialize the line search object, passing
it ``opt_interface`` as its "objective".

```python
opt_ls = ProbLSOptimizer(opt_interface)
```

``opt_ls`` has two methods that are of interest for the end-user.
``opt_ls.prepare(*args)`` has to be called once to initialize the line search.
``opt_ls.proceed(*args)`` proceeds one step in the line search (i.e. one
function evaluation). For both functions, ``*args`` are arguments passed to the
``opt_interface``, which can be used to pass a ``feed_dict`` (if you are feeding
data via placeholders).

Overall, it's just five extra lines of code and reads

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

See the ``examples/`` for demo scripts.

## Guide to this Implementation

Coming soon...



[1]: https://arxiv.org/abs/1502.02846
