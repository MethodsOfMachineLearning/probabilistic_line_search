# -*- coding: utf-8 -*-
"""
TensorFlow optimizer that acts as an interface for the probabilistic line
search algorithm.

Created on Tue Jul 26 15:23:49 2016

@author: Lukas Balles [lballes@tuebingen.mpg.de]
"""

import tensorflow as tf
import gradient_moment as gm

class ProbLSOptimizerSGDInterface(tf.train.Optimizer):
  """Optimizer that implements gradient descent with and interface for the
  probabilistic line search algorithm.  
  @@__init__
  """

  def __init__(self, momentum=None, use_locking=False, name="ProbLS"):
    """Construct a new probabilistic line search optimizer.
    
    Args:
    
    :momentum: None or scalar momentum parameter.
    :use_locking: If True use locks for update operations.
    :name: Optional name prefix for the operations created when applying
        gradients. Defaults to "ProbLS".
    """
    super(ProbLSOptimizerSGDInterface, self).__init__(use_locking, name)
    
    assert momentum is None or (isinstance(momentum, float) and 0<=momentum<=1)
    self.momentum = momentum
    
    self._ops_ready = False
    self._prepared = False
    self.sess = None
    
    self.dt = None
    self.adv_eval_op = None
    self.accept_op = None
  
  def _create_slots(self, var_list):
    for v in var_list:
      self._zeros_slot(v, "grad", "grad") # Variables to memorize gradients
      self._zeros_slot(v, "dir", "dir") # Search direction
      self._zeros_slot(v, "gradvar", "gradvar") # Gradient variance
  
  def minimize(self, losses, var_list):
    """Add operations to perform SGD with probabilistic line search. This
    comprises two sets of operations:
    
    1) adv_eval_op:    
       Advance along the current search direction, compute the loss,
       the gradients and both variances. Gradient and its variance are stored
       in slot variables. Return the loss f, projected gradient df,
       variance of the loss fvar, and variance of the projected gradient dfvar
    2) accept_op:
       Accept the current point. Set its gradient as the new search direction.
       Returns df and dfvar with respect to this new search direction.
    
    Inputs:
      :losses: A Tensor of shape (batch_size,) containing the *individual*
          loss for each example in the batch. Do *not* pass a scalar mean loss
          as for the built-in tensorflow optimizers.
      :var_list: List of Variable objects to update to minimize loss."""
    
    assert isinstance(losses, tf.Tensor)
    for var in var_list: assert isinstance(var, tf.Variable)
    assert len(var_list) >= 0
    assert len(var_list) == len(set(var_list)) # Check for duplicates
    
    input_dtype = losses.dtype.base_dtype
    
    # Create and retrieve slot variables
    self._create_slots(var_list)
    mem_grads = [self.get_slot(v, "grad") for v in var_list]
    dirs = [self.get_slot(v, "dir") for v in var_list]
    mem_gradvars = [self.get_slot(v, "gradvar") for v in var_list]
    mem_f = tf.Variable(0.0, input_dtype, name="mem_f")
    mem_fvar = tf.Variable(0.0, input_dtype, name="mem_fvar")
    
    with tf.name_scope("ProbLS"):
      
      ###### adv_eval_op ######################################################      
      # Extract the batch size, i.e. the length of the losses vector
      batch_size = tf.cast(tf.gather(tf.shape(losses), 0), input_dtype,
                           name="batch_size")
      
      # Add a scalar placeholder dt and operations that advance t by dt,
      # i.e., update v += dt*d (v: variable, d: search direction)
      with tf.name_scope("advance_t"):
        self.dt = tf.placeholder(dtype=input_dtype, shape=[], name="delta_t")
        steps = [tf.mul(self.dt, tf.convert_to_tensor(d)) for d in dirs]
        advance_t_updates = [v.assign_add(s) for v, s in zip(var_list, steps)]
      
      # With a dependency on the advance_t update (making sure that a step is
      # taken first), add tensors that compute the loss f, the gradients and
      # the gradient moments
      with tf.control_dependencies(advance_t_updates):
        loss = tf.reduce_mean(losses, name="f")
        grads, moms = gm.grads_and_grad_moms(loss, batch_size, var_list)
      
      # Add variance of the loss
      ssl = tf.reduce_mean(tf.square(losses), name="sum_of_squared_losses")
      fvar = tf.div(ssl-tf.square(loss), batch_size-1., name="fvar")
        
      # Add projected gradient df (w.r.t. the current search direction)
      with tf.name_scope("df"):
        proj_grads = [tf.reduce_sum(tf.mul(g, d), name="proj_grad")
                      for g, d in zip(grads, dirs)]
        df = tf.add_n(proj_grads, name="df")
      
      # Add gradient variances and the variance of df
      gradvars = [tf.div(mom-tf.square(g), batch_size-1.)
                  for mom, g in zip(moms, grads)]
      dfvar = tf.add_n([tf.reduce_sum(gv*tf.square(d))
                        for gv, d in zip(gradvars, dirs)])
      
      # Add operations to memorize stuff in variables. This is because they
      # are needed in the case that this points ends up being accepted (i.e.,
      # if the accept op is called next). Stored quantities are
      # - gradients
      # - gradient moment
      # - f and fvar
      with tf.name_scope("memorize"):
        mem_updates = [v.assign(grad) for v, grad in zip(mem_grads, grads)]
        mem_updates.extend(
            [v.assign(gv) for v, gv in zip(mem_gradvars, gradvars)]
            )
        mem_updates.append(mem_f.assign(loss))
        mem_updates.append(mem_fvar.assign(fvar))
      
      # With a dependency on the memorization, add the adv_eval_op. It is
      # simply the tuple (f, df, fvar, dfvar). All the dependencies make sure
      # that it also does the other stuff
      with tf.control_dependencies(mem_updates):
        self.adv_eval_op = tf.tuple([loss, df, fvar, dfvar], name="results")
      
      ###### accept_op ########################################################      
      # Operation that accepts the current state, i.e.
      # - sets the current gradient as the new search direction
      # - returns a new df, computed w.r.t. to that new search direction 
      with tf.name_scope("accept"):
        # Add operations the set the new search direction
        if self.momentum is None:
          new_dirs = [tf.neg(g) for g in mem_grads]
        else:
          mu = tf.convert_to_tensor(self.momentum, name="momentum_mu")
          new_dirs = [mu*d-g for d, g in zip(dirs, mem_grads)]
        dir_updates = [d.assign(d_new) for d, d_new in zip(dirs, new_dirs)]
        
        # With a dependency on the search direction updates, compute df and
        # dfvar w.r.t. the new search direction, using the memorized gradients
        # and gradient variances
        with tf.control_dependencies(dir_updates):
          proj_grads_new = [tf.reduce_sum(g*d)
                            for g, d in zip(mem_grads, dirs)]
          df_new = tf.add_n(proj_grads_new, name="df_new")
          dfvar_new = tf.add_n([tf.reduce_sum(gv*tf.square(d))
                                for gv, d in zip(mem_gradvars, dirs)])
        self.accept_op = tf.tuple([mem_f, df_new, mem_fvar, dfvar_new],
                                  name="results_after_accept")
    
    # Set internal flag that the operations are now ready
    self._ops_ready = True
  
  def register_session(self, sess):
    """Register the session ``sess`` with this line search interface.
    Computations resulting from calls to ``prepare``, ``adv_eval`` or
    ``accept`` will be executed in this session.
    
    Inputs:
      :sess: A TensorFlow Session."""
    
    if not self._ops_ready:
      raise Warning("You have to call minimize first")
    assert isinstance(sess, tf.Session)
    self.sess = sess
  
  def prepare(self, feed_dict=None):
    """Make a first evaluation to properly initialize all gradients, et cetera.
    Call this function before using ``adv_eval`` or ``accept``."""
    
    if self.sess is None:
      raise Warning("You have to register a session first.")
    
    if feed_dict is None:
      feed_dict = {}
    feed_dict[self.dt] = 0.0
    
    # We need to evaluate and accept once in order to compute initial
    # gradients and accept them as search direction. Only then can we
    # make the first "real" evaluation and return the results
    self.sess.run(self.adv_eval_op, feed_dict)
    self.sess.run(self.accept_op)
    self.sess.run(self.adv_eval_op, feed_dict)
    self._prepared = True
    return self.sess.run(self.accept_op)
    
  def adv_eval(self, dt, feed_dict=None):
    """Advance by an increment ``dt`` along the current search direction and
    evaluate.
    
    Inputs:
      :dt: Float step size increment.
      :feed_dict: Optional feed_dict.
    
    Returns:
      :f: Function value at the new point.
      :df: Gradient at the new point, projected onto the search direction.
      :fvar: Variance of f.
      :dfvar: Variance of df."""
    
    if not self._prepared:
      raise Warning("You have to call prepare first")
    if feed_dict is None:
      feed_dict = {}
    feed_dict[self.dt] = dt
    return self.sess.run(self.adv_eval_op, feed_dict)
  
  def accept(self):
    if not self._prepared:
      raise Warning("You have to call prepare first")
    return self.sess.run(self.accept_op)