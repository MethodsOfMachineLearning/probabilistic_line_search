# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:00:24 2016

@author: lballes
"""

import numpy as np
import gaussian_process
import utils

class ProbLSOptimizer(object):
  """Probabilistic line search optimizer.
  
  @@__init__
  """
  
  def __init__(self, func, c1=0.05, cW=0.3, fpush=1.0, alpha0=0.01,
               target_df=0.5, df_lo=-0.1, df_hi=1.1, max_steps=10, max_expl=6, 
               max_dmu0=0.0, max_change_factor=10.0, expl_policy="linear"):
    """Create a new probabilistic line search object.
    
    Inputs:
      :func: Interface to the objective function. We assume that it has three
          methods.          
             - ``f, df, fvar, dfvar = func.adv_eval(dt)`` to proceed along the current search
               direction by an increment ``dt``, returning function value, 
               projected gradient and variance estimates for both.
             - ``f, df, fvar, dfvar = func.accept()`` to accept the current
               step size, returning function value, projected gradients and an
               estimate of the variance of these two quantities.
             - ``f, df, fvar, dfvar = func.prepare()`` to prepare the
               interface, returning an initial observation of function value and
               gradient, as well as the variances.
          If the function interface takes additional arguments (e.g. a feed
          dict with a batch of data in tensorflow), those are passed as
          positional arguments ``*pass_to_func_args``. 
      :c1: Scalar parameters for the first Wolfe conditions. Default to 0.05.
      :cW: Acceptance threshold for the Wolfe probability. Defaults to 0.3.
      :fpush: Push factor that is multiplied with the accepted step size to get
          the base step size for the next line search.
      :alpha0: Initial step size. Defaults to 0.03.
      :target_df: The desired value for the relative projected gradient. Defaults
          to 0.5.
      :df_lo, df_hi: Lower and higher threshold for the relative projected
          gradient. Default to -0.1 and 1.1.
      :max_steps: Maximum number of steps (function evaluations) per line search.
          Defaults to 5.
      :max_epl: Maximum number of exploration steps per line search. Defaults to
          3.
      :max_dmu0: If the posterior derivative at t=0 exceeds ``max_dmu0``, the
          current line search is aborted. This is a safeguard against bad search
          directions. Defaults to 0.0.
      :max_change_factor: The algorithm usually takes the accepted alpha of the
          current line search as the base ``alpha0`` of the next one (after
          multiplying with ``fpush``. However, if a line search accepts an alpha
          that is more than ``max_change_factor`` times smaller or larger than
          the current ``alpha0``, we instead set the next ``alpha0`` to a running
          average of the accepted alphas (``alpha_stats``). Defaults to 10.0.
      :expl_policy: String indicating the policy used for exploring points *to
          the right* in the line search. If ``k`` is the number of exploration steps
          already made, then the ``"linear"`` exploration policy chooses
          ``2*(k+1)*alpha0`` as the next exploration candidate. The ``"exponential"``
          policy chooses ``2**(k+1)*alpha0``. Defaults to ``"linear"``."""
    
    # Make sure the function_interface is valid and store it
    assert hasattr(func, "adv_eval")
    assert hasattr(func, "accept")
    assert hasattr(func, "prepare")
    self.func = func
    
    # Store the line search parameters
    self.c1 = c1
    self.cW = cW
    self.fpush = fpush
    self.target_df = target_df
    self.df_lo = df_lo
    self.df_hi = df_hi
    self.max_steps = max_steps
    self.max_expl = max_expl
    self.max_dmu0 = max_dmu0
    self.max_change_factor = max_change_factor
    assert expl_policy in ["linear", "exponential"]
    self.expl_policy = expl_policy
    
    # Initialize base step size with given value.
    self.alpha0 = alpha0
    
    # alpha_stats will contain a running average of accepted step sizes
    self.alpha_stats = alpha0
    
    # Raw function values at the origin of the line search
    self.f0 = None
    self.df0 = None
    
    # Counting steps in the current line search and, separately, steps that
    # explore "to the right"
    self.num_steps = 0
    self.num_expl = 0
    
    # Initialize GP object
    self.gp = gaussian_process.ProbLSGaussianProcess()
    
    # Switch to assert that the prepare method will be called first
    self.prepare_called = False
    
    # Internal abort status
    self.abort_status = 0
  
  def scale_obs(self, f_raw, df_raw, fvar_raw, dfvar_raw):
    """Scale an observation of function value and gradient. See section 3.4 of
    [1] for details."""
    
    f = (f_raw-self.f0)/(self.df0*self.alpha0)
    df = df_raw/(self.df0)
    fvar = fvar_raw/((self.alpha0*self.df0)**2)
    dfvar = dfvar_raw/(self.df0**2)
    return f, df, fvar, dfvar
  
  # LEGACY
#  def scale_sigmas(self, sigma_f_raw, sigma_df_raw):
#    """Scale the variance estimates. See section 3.4 of [1] for details."""
#    
#    sigma_f = sigma_f_raw/((self.alpha0*self.df0)**2)
#    sigma_df = sigma_df_raw/(self.df0**2)
#    return sigma_f, sigma_df
  
  def rescale_t(self, t):
    """Rescale a step size used internally by multiplying with the base step
    size."""
    
    return t*self.alpha0
  
  def rescale_obs(self, f, df):
    """Rescale ab observation to real-world scale."""
    
    f_raw = f*self.df0*self.alpha0 + self.f0
    df_raw = df*self.df0
    return f_raw, df_raw
  
  def prepare(self, *pass_to_func_args):
    """Preparation.
    
    *pass_to_func_args are arguments that are passed to the function interface,
    e.g. a feed dict."""
    
    # Call the prepare op of the function interface, reset the observation
    # lists, the sigmas, and f0 and df0
    f_raw, df_raw, fvar_raw, dfvar_raw = self.func.prepare(*pass_to_func_args)
    self.f0 = f_raw
    self.df0 = np.abs(df_raw)
    
    # Add the first observation to the gp
    f, df, fvar, dfvar = self.scale_obs(f_raw, df_raw, fvar_raw, dfvar_raw)
    self.gp.add(0.0, f, df, fvar, dfvar)
    
    # Set flag that the prepare method has been called
    self.prepare_called = True
  
  def accept(self):
    """Accept the most recent step size."""
    
    assert self.abort_status != 1
    assert self.num_steps >= 1
    
    # Rescale to the "real-world" step size alpha
    alpha = self.rescale_t(self.gp.ts[-1])
    
    # If this accept was not due to an abort and the step size did not change
    # *too much*, we use the accepted alpha as the new base step size alpha0
    # (and update a running average alpha_stats). Otherwise, we use said
    # running average as the new base step size.
    f = self.max_change_factor    
    if self.abort_status == 0 and self.alpha0/f < alpha < self.alpha0*f:
      self.alpha_stats = 0.95*self.alpha_stats + 0.05*alpha
      self.alpha0 = self.fpush*alpha
    else:
      self.alpha0 = self.alpha_stats
    
    # Reset abort status and counters
    self.abort_status = 0
    self.num_steps = 0
    self.num_expl = 0
    
    # Run accept op, reset f0 and df0
    f_raw, df_raw, fvar_raw, dfvar_raw = self.func.accept()
    self.f0 = f_raw
    self.df0 = np.abs(df_raw)
    
    # Reset the gp and add the first observation to the gp
    self.gp.reset()
    f, df, fvar, dfvar = self.scale_obs(f_raw, df_raw, fvar_raw, dfvar_raw)
    self.gp.add(0.0, f, df, fvar, dfvar)
  
  def evaluate(self, t, *pass_to_func_args):
    """Evaluate at step size ``t``.
    
    *pass_to_func_args are arguments that are passed to the function interface,
    e.g. a feed dict."""
    
    assert self.prepare_called
    
    self.num_steps += 1
    
    # Call the adv_eval method of the function interface with the increment
    # re-scaled to the "real-world" step size
    dt = t-self.gp.ts[-1]
    dalpha = self.rescale_t(dt)
    f_raw, df_raw, fvar_raw, dfvar_raw = self.func.adv_eval(dalpha,
                                                            *pass_to_func_args)
    
    # Safeguard against inf or nan encounters. Trigerring abort.
    if np.isnan(f_raw) or np.isinf(f_raw):
      f_raw = 100.0
#      self.abort_status = 1 # Previously, we aborted when encountering inf / nan. Now we just put a high function value / gradient and go on. Is this the right thing to do? Which values?
    if np.isnan(df_raw) or np.isinf(df_raw):
      df_raw = 10.0
#      self.abort_status = 1
    
    # Scale the observations, add it to the GP and update the GP
    f, df, fvar, dfvar = self.scale_obs(f_raw, df_raw, fvar_raw, dfvar_raw)
    self.gp.add(t, f, df, fvar, dfvar)
    self.gp.update()
  
  def find_next_t(self):
    """Find the step size for the next evaluation."""
    
    assert self.num_steps >= 1
    
    # Generate candidates: Minima of the piece-wise cubic posterior mean plus
    # one exploration point (2**k for several exploration steps in one search)
    candidates = self.gp.find_dmu_equal(self.target_df)
    if self.expl_policy == "linear":
      candidates.append(2.*(self.num_expl+1))
    elif self.expl_policy == "exponential":
      candidates.append(2.**(self.num_expl+1))
    else:
      raise Exception("Unknown exploration policy")
    print "\t * Computing utilities for candidates", candidates
    
    # Compute p_Wolfe for candidates
    pws = [self.compute_p_wolfe(t) for t in candidates]
    print "\t * p_Wolfe:", pws
    ind_best = np.argmax(pws)
    
    # Memorize when we have chosen the exploration point
    if ind_best == len(candidates) - 1:
        self.num_expl += 1
    
    # Return the candidate t with maximal utility
    print "\t * Best candidate is", candidates[ind_best], "(was candidate", ind_best, "/", len(candidates)-1, ")"
    return candidates[ind_best]
  
  def find_abort_t(self):
    """Find the step size to use for an abort."""
    
    ts = self.gp.ts
    pws = [self.compute_p_wolfe(t) for t in ts]
    if max(pws) > 0.5*self.cW:
      t = ts[np.argmax(pws)]
    else:
      t = 0.0
    offset = 0.01
    
    return t + offset
  
  def compute_p_wolfe(self, t):
    # Already changed dCov and Covd here
    """Computes the probability that step size ``t`` satisfies the adjusted
    Wolfe conditions under the current GP model."""
    
    # Compute mean and covariance matrix of the two Wolfe quantities a and b
    # (equations (11) to (13) in [1]).
    mu0 = self.gp.mu(0.)
    dmu0 = self.gp.dmu(0.)
    mu = self.gp.mu(t)
    dmu = self.gp.dmu(t)    
    V0 = self.gp.V(0.)
    Vd0 = self.gp.Vd(0.)
    dVd0 = self.gp.dVd(0.)    
    dCov0t = self.gp.dCov_0(t)
    Covd0t = self.gp.Covd_0(t)
    
    ma = mu0 - mu + self.c1*t*dmu0
    Vaa = V0 + dVd0*(self.c1*t)**2 + self.gp.V(t) \
          + 2.*self.c1*t*(Vd0 - dCov0t) - 2.*self.gp.Cov_0(t)
    mb = dmu
    Vbb = self.gp.dVd(t)
    
    # Very small variances can cause numerical problems. Safeguard against
    # this with a deterministic evaluation of the Wolfe conditions.
    if Vaa < 1e-9 or Vbb < 1e-9:
      return 1. if ma>=0. and mb>=0. else 0.
    
    Vab = Covd0t + self.c1*t*self.gp.dCovd_0(t) - self.gp.Vd(t)
    
    # Compute correlation factor and integration bounds for adjusted p_Wolfe
    # and return the result of the bivariate normal integral.
    rho = Vab/np.sqrt(Vaa*Vbb)
    al = -ma/np.sqrt(Vaa)
    bl = (self.df_lo - mb)/np.sqrt(Vbb)
    bu = (self.df_hi - mb)/np.sqrt(Vbb)
    return utils.bounded_bivariate_normal_integral(rho, al, np.inf, bl, bu)
  
  def check_for_acceptance(self):
    """Checks whether the most recent point should be accepted."""
    
    # Return False when no evaluations t>0 have been made yet
    if self.num_steps == 0:
      return False
    
    # If an abort has been triggered, return True
    if self.abort_status == 2:
      return True
    
    # Check Wolfe probability
    pW = self.compute_p_wolfe(self.gp.ts[-1])
    if pW >= self.cW:
      return True
    else:
      return False
  
  def proceed(self, *pass_to_func_args):
    """Make one step (function evaluation) in the line search.
    
    *pass_to_func_args are arguments that are passed to the function interface,
    e.g. a feed dict."""
    
    assert self.prepare_called
    
    # Check for acceptance and accept the previous point as the case may be
    if self.check_for_acceptance():
      print "-> ACCEPT"
      print "\t * alpha = ", self.rescale_t(self.gp.ts[-1]), "[alpha0 was", self.alpha0, "]"
      self.accept()
      print "\t * f = ", self.f0
    
    # In the first call to proceed in a new line search, evaluate at t=1.
    if self.num_steps == 0:
      print "************************************"
      print "NEW LINE SEARCH [alpha0 is", self.alpha0, "]"
      print "-> First step, evaluating at t = 1.0"
      self.evaluate(1., *pass_to_func_args)
    
    # Abort with a very small, safe step size if 
    # - Abort triggered in an other method, e.g. evaluate() encountered inf or
    #   nan. (self.abort_status==1)
    # - the maximum number of steps per line search is exceeded
    # - the maximum number of exploration steps is exceeded
    # - the posterior derivative at t=0. is too large (bad search direction)
    elif (self.abort_status == 1
          or self.num_steps >= self.max_steps
          or self.num_expl >= self.max_expl
          or self.gp.dmu(0.) >= self.max_dmu0):
      t_new = self.find_abort_t()
      print "-> Aborting with t = ", t_new
      self.evaluate(t_new, *pass_to_func_args)
      self.abort_status = 2
    
    # This is an "ordinary" evaluation. Find the best candidate for the next
    # evaluation and evaluate there.
    else:
      print "-> Ordinary step", self.num_steps, ", searching for new t"
      t_new = self.find_next_t()
      print "\t * Evaluating at t =", t_new
      self.evaluate(t_new, *pass_to_func_args)
    
    # Return the real-world function value
    f, _ = self.rescale_obs(self.gp.fs[-1], self.gp.dfs[-1])    
    return f
  
  def proceed_constant_step(self, alpha, *pass_to_func_args):
    """Make one step (function evaluation) in the line search.
    
    *pass_to_func_args are arguments that are passed to the function interface,
    e.g. a feed dict."""
    
    assert self.prepare_called
    
    if self.num_steps >= 1:
      self.accept()
    
    print "************************************"
    print "CONSTANT STEP with alpha =", alpha, "[alpha0 is", self.alpha0, "]"
    t = alpha/self.alpha0
    print "-> Evaluating at t =", t
    self.evaluate(t, *pass_to_func_args)
    
    f, _ = self.rescale_obs(self.gp.fs[-1], self.gp.dfs[-1])
    return f
  
  # ToDo: Commenting
  def visualize_ei_pw(self, ax):
    """Visualize the current state of the line search: expected improvement
    and p_Wolfe.
    
    ``ax`` is a matplotlib axis."""
    a, b = min(self.gp.ts), max(self.gp.ts)
    lo = a - .05*(b-a)
    up = b + (b-a) #.05*(b-a)   
    tt = np.linspace(lo, up, num=1000)
    ei = [self.gp.expected_improvement(t) for t in tt]
    pw = [self.compute_p_wolfe(t) for t in tt]
    prod = [e*p for e, p in zip(ei, pw)]
    ax.hold(True)
    ax.plot(tt, ei, label="EI")
    ax.plot(tt, pw, label="pW")
    ax.plot(tt, prod, label="EI*pW")
    ax.plot([lo, up], [self.cW, self.cW], color="grey")
    ax.text(lo, self.cW, "Acceptance threshold", fontsize=8)
    ax.set_xlim(lo, up)
    ax.legend(fontsize=10)

## LEGACY VERSION OF p_Wolfe #################################################
# Changed dCov and Covd here already!
#  def compute_p_wolfe_original(self, t):
#    """Computes the probability that step size ``t`` satisfies the Wolfe
#    conditions under the current GP model."""
#    
#    # Compute mean and covariance matrix of the two Wolfe quantities a and b
#    # (equations (11) to (13) in [1]).
#    mu0 = self.gp.mu(0.)
#    dmu0 = self.gp.dmu(0.)
#    mu = self.gp.mu(t)
#    dmu = self.gp.dmu(t)    
#    V0 = self.gp.V(0.)
#    Vd0 = self.gp.Vd(0.)
#    dVd0 = self.gp.dVd(0.)    
#    ma = mu0 - mu + self.c1*t*dmu0
#    Vaa = V0 + dVd0*(self.c1*t)**2 + self.gp.V(t) \
#          + 2.*self.c1*t*(Vd0 - self.gp.dCov_0(t)) - 2.*self.gp.Cov_0(t)
#    mb = dmu - self.c2*dmu0
#    Vbb = dVd0*self.c2**2 - 2.*self.c2*self.gp.dCovd_0(t) + self.gp.dVd(t)
#    
#    # Very small variances can cause numerical problems. Safeguard against
#    # this with a deterministic evaluation of the Wolfe conditions.
#    if Vaa < 1e-9 or Vbb < 1e-9:
#      return 1. if ma>=0. and mb>=0. else 0.
#    
#    Vab = -self.c2*(Vd0 + self.c1*t*dVd0) + self.c2*self.gp.dCov_0(t) \
#          + self.gp.Covd_0(t) + self.c1*t*self.gp.dCovd_0(t) - self.gp.Vd(t)
#    
#    # Compute rho and integration bounds for p_Wolfe and return the result of
#    # the bivariate normal integral. Upper limit for b is used when strong
#    # Wolfe conditions are requested (cf. equations (14) to (16)in [1]).
#    rho = Vab/np.sqrt(Vaa*Vbb)
#    al = -ma/np.sqrt(Vaa)
#    bl = -mb/np.sqrt(Vbb)
#    if self.strong_wolfe:
#      bbar = 2.*self.c2*(np.abs(dmu0) + 2.*np.sqrt(dVd0))
#      bu = (bbar - mb)/np.sqrt(Vbb)
#      return utils.bounded_bivariate_normal_integral(rho, al, np.inf, bl, bu)
#    else:
#      return utils.unbounded_bivariate_normal_integral(rho, al, bl)
###############################################################################