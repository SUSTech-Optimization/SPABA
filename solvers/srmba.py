from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from numba.experimental import jitclass

    from benchmark_utils import constants
    from benchmark_utils.get_memory import get_memory
    from benchmark_utils.minibatch_sampler import init_sampler
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.minibatch_sampler import MinibatchSampler
    from benchmark_utils.minibatch_sampler import spec as mbs_spec
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler
    from benchmark_utils.learning_rate_scheduler import spec as sched_spec
    from benchmark_utils.oracles import MultiLogRegOracle, DataCleaningOracle
    from benchmark_utils.learning_rate_scheduler import LearningRateScheduler

    import jax
    import jax.numpy as jnp
    from functools import partial


class Solver(BaseSolver):
    """Stochastic Bilevel Algorithm (SOBA).

    M. Dagréou, P. Ablin, S. Vaiter and T. Moreau, "A framework for bilevel
    optimization that enables stochastic and global variance reduction
    algorithms", NeurIPS 2022."""
    name = 'SOBA-STORM'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'batch_size': [64],
        'eta': [1],
        'eval_freq': [128],
        'random_state': [1],
        'framework': ["jax"],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def skip(self, f_train, f_val, **kwargs):
        if self.framework == 'numba':
            if self.batch_size == 'full':
                return True, "Numba is not useful for full bach resolution."
            elif isinstance(f_train(),
                            (MultiLogRegOracle, DataCleaningOracle)):
                return True, "Numba implementation not available for " \
                      "this oracle."
            elif isinstance(f_val(), (MultiLogRegOracle, DataCleaningOracle)):
                return True, "Numba implementation not available for" \
                      "this oracle."
        elif self.framework not in ['jax', 'none', 'numba']:
            return True, f"Framework {self.framework} not supported."

        try:
            f_train(framework=self.framework)
        except NotImplementedError:
            return (
                True,
                f"Framework {self.framework} not compatible with "
                f"oracle {f_train()}"
            )
        return False, None

    def set_objective(self, f_train, f_val, n_inner_samples, n_outer_samples,
                      inner_var0, outer_var0):

        self.f_inner = f_train(framework=self.framework)
        self.f_outer = f_val(framework=self.framework)
        self.n_inner_samples = n_inner_samples
        self.n_outer_samples = n_outer_samples

        if self.batch_size == "full":
            self.batch_size_inner = n_inner_samples
            self.batch_size_outer = n_outer_samples
        else:
            self.batch_size_inner = self.batch_size
            self.batch_size_outer = self.batch_size

        if self.framework == 'numba':
            # JIT necessary functions and classes
            self.soba = njit(soba)
            self.MinibatchSampler = jitclass(MinibatchSampler, mbs_spec)
            self.LearningRateScheduler = jitclass(
                LearningRateScheduler, sched_spec
            )
        elif self.framework == "none":
            self.soba = soba
            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = LearningRateScheduler
        elif self.framework == 'jax':
            self.f_inner = jax.jit(
                partial(self.f_inner, batch_size=self.batch_size_inner)
            )
            self.f_outer = jax.jit(
                partial(self.f_outer, batch_size=self.batch_size_outer)
            )
            inner_sampler, self.state_inner_sampler \
                = init_sampler(n_samples=n_inner_samples,
                               batch_size=self.batch_size_inner)
            outer_sampler, self.state_outer_sampler \
                = init_sampler(n_samples=n_outer_samples,
                               batch_size=self.batch_size_outer)
            self.soba = partial(
                soba_jax,
                inner_sampler=inner_sampler,
                outer_sampler=outer_sampler
            )
        else:
            raise ValueError(f"Framework {self.framework} not supported.")

        self.inner_var = inner_var0
        self.outer_var = outer_var0
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        self.memory = 0

    def warm_up(self):
        if self.framework in ['numba', 'jax']:
            self.run_once(2)
            self.inner_var = self.inner_var0
            self.outer_var = self.outer_var0

    def run(self, callback):
        eval_freq = self.eval_freq
        memory_start = get_memory()

        # Init variables
        inner_var = self.inner_var.copy()
        outer_var = self.outer_var.copy()
        if self.framework == "jax":
            v =  jnp.zeros_like(inner_var)
            v1 = jnp.zeros_like(inner_var)
            v2 = jnp.zeros_like(inner_var)
            inner_var1 = jnp.zeros_like(inner_var)
            inner_var2 = jnp.zeros_like(inner_var)
            outer_var1 = jnp.zeros_like(outer_var)
            outer_var2 = jnp.zeros_like(outer_var)
            # Init lr scheduler
            step_sizes = jnp.array(
                [self.step_size, self.step_size / self.outer_ratio, self.eta]
            )
            exponents = jnp.array(
                [1/3, 1/3, 2/3]
            )
            state_lr = init_lr_scheduler(step_sizes, exponents)
            carry = dict(
                state_lr=state_lr,
                state_inner_sampler=self.state_inner_sampler,
                state_outer_sampler=self.state_outer_sampler,
            )
        else:
            rng = np.random.RandomState(self.random_state)
            v = np.zeros_like(inner_var)
            # Init lr scheduler
            step_sizes = np.array(
                [self.step_size, self.step_size / self.outer_ratio, self.eta]
            )
            exponents = exponents = jnp.zeros(3)

            lr_scheduler = self.LearningRateScheduler(
                np.array(step_sizes, dtype=float), exponents
            )
            inner_sampler = self.MinibatchSampler(self.n_inner_samples,
                                                  self.batch_size_inner)
            outer_sampler = self.MinibatchSampler(self.n_outer_samples,
                                                  self.batch_size_outer)

        # Start algorithm
        while callback():
            if self.framework == 'jax':
                inner_var, outer_var, v, v1, v2, inner_var1, inner_var2, outer_var1, outer_var2, carry = self.soba(
                    self.f_inner, self.f_outer,
                    inner_var, outer_var, v, v1, v2, inner_var1, inner_var2, outer_var1, outer_var2, max_iter=eval_freq, **carry
                )
            else:
                inner_var, outer_var, v, v1, v2, inner_var1, inner_var2, outer_var1, outer_var2 = self.soba(
                    self.f_inner, self.f_outer,
                    inner_var, outer_var, v, v1, v2, inner_var1, inner_var2, outer_var1, outer_var2,
                    inner_sampler=inner_sampler,
                    outer_sampler=outer_sampler,
                    lr_scheduler=lr_scheduler, max_iter=eval_freq,
                    seed=rng.randint(constants.MAX_SEED)
                )
            memory_end = get_memory()
            self.inner_var = inner_var
            self.outer_var = outer_var
            self.memory = memory_end - memory_start
            self.memory /= 1e6

    def get_result(self):
        return dict(inner_var=self.inner_var, outer_var=self.outer_var,
                    memory=self.memory)
                    
    


def soba(inner_oracle, outer_oracle, inner_var, outer_var, v, v1, v2, inner_var1, inner_var2, outer_var1, outer_var2, 
         inner_sampler=None, outer_sampler=None, lr_scheduler=None, max_iter=1,
         seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_iter):
        inner_step_size, outer_step_size, eta = lr_scheduler.get_lr()

        # Step.1 - get all gradients and compute the implicit gradient.
        slice_inner, _ = inner_sampler.get_batch()
        _, grad_inner_var_old, hvp_old, cross_v_old = inner_oracle.oracles(
            inner_var1, outer_var1, v1, slice_inner, inverse='id'
        )
        grad_inner_var, hvp, cross_v = inner_oracle.oracles(
            inner_var, outer_var, v, slice_inner, inverse='id'
        )

        slice_outer, _ = outer_sampler.get_batch()
        grad_in_outer_old, grad_out_outer_old = outer_oracle.grad(
            inner_var1, outer_var1, slice_outer
        )
        grad_in_outer, grad_out_outer = outer_oracle.grad(
            inner_var, outer_var, slice_outer
        )

        # Step.2 - update the variables d
        d_z1 =  hvp_old - grad_in_outer_old
        d_x1 = -cross_v_old + grad_out_outer_old
        d_y1 = grad_inner_var_old

        d_z2 =  hvp - grad_in_outer
        d_x2 = -cross_v + grad_out_outer
        d_y2 = grad_inner_var

        # Step.3 - update the variables v
        v2 = (1-eta)*v2 + (1-eta) * (d_z2 - d_z1) + eta * d_z2
        inner_var2 = (1-eta)*inner_var2 + (1-eta) * (d_y2 - d_y1) + eta * d_y2
        outer_var2 = (1-eta)*outer_var2 + (1-eta) * (d_x2 - d_x1) + eta * d_x2


        # Step.4 - Save the current variables
        inner_var1 = inner_var
        outer_var1 = outer_var
        v1 = v
        # Step.5 - update the variables
        inner_var -= inner_step_size * inner_var2
        outer_var -= outer_step_size * outer_var2
        v -= inner_step_size * v2

    return inner_var, outer_var, v,  v1, v2, inner_var1, inner_var2, outer_var1, outer_var2

@partial(jax.jit, static_argnums=(0, 1),
         static_argnames=('inner_sampler', 'outer_sampler', 'max_iter'))
def soba_jax(f_inner, f_outer, inner_var, outer_var, v,  v1, v2, inner_var1, inner_var2, outer_var1, outer_var2,
             state_inner_sampler=None, state_outer_sampler=None, state_lr=None,
             inner_sampler=None, outer_sampler=None, max_iter=1):

    grad_inner = jax.grad(f_inner, argnums=0)
    grad_outer = jax.grad(f_outer, argnums=(0, 1))

    def soba_one_iter(carry, _):

        (inner_step_size, outer_step_size, eta), carry['state_lr'] = update_lr(
            carry['state_lr']
        )

        # Step.1 - get all gradients and compute the implicit gradient.
        start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
            carry['state_inner_sampler']
        )
        grad_inner_var, vjp_train = jax.vjp(
            lambda z, x: grad_inner(z, x, start_inner), carry['inner_var'],
            carry['outer_var']
        )
        hvp, cross_v = vjp_train(carry['v'])

        grad_inner_var_old, vjp_train_old = jax.vjp(
            lambda z, x: grad_inner(z, x, start_inner), carry['inner_var1'],
            carry['outer_var1']
        )
        hvp_old, cross_v_old = vjp_train_old(carry['v1'])

        start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
            carry['state_outer_sampler']
        )
        grad_in_outer, grad_out_outer = grad_outer(
            carry['inner_var'], carry['outer_var'], start_outer
        )

        grad_in_outer_old, grad_out_outer_old = grad_outer(
            carry['inner_var1'], carry['outer_var1'], start_outer
        )

        # Step.2 - update the variables d
        d_z1 =  hvp_old - grad_in_outer_old
        d_x1 = -cross_v_old + grad_out_outer_old
        d_y1 = grad_inner_var_old

        d_z2 =  hvp - grad_in_outer
        d_x2 =  -cross_v + grad_out_outer
        d_y2 = grad_inner_var

        # Step.3 - update the variables v
        carry['v2'] = (1-3*eta) * carry['v2']+(1-3*eta) * (d_z2 - d_z1) + 3*eta * d_z2
        carry['inner_var2'] = (1-15*eta) * carry['inner_var2'] + (1-15*eta) * (d_y2 - d_y1) + 15*eta * d_y2
        carry['outer_var2'] = (1-eta) * carry['outer_var2'] + (1-eta) * (d_x2 - d_x1) + eta * d_x2

        # Step.4 - Save the current variables
        carry['v1'] = carry['v']
        carry['inner_var1'] = carry['inner_var']
        carry['outer_var1'] = carry['outer_var']
        
        # Step.5 - update the variables
        carry['inner_var'] -= inner_step_size * carry['inner_var2']
        carry['v'] -= inner_step_size * carry['v2']
        carry['outer_var'] -= outer_step_size * carry['outer_var2']
       
        # #Use prox to make sure we do not diverge
        # inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

        return carry, _

    init = dict(
        inner_var=inner_var, outer_var=outer_var, v=v, v1=v1, v2=v2, inner_var1=inner_var1, inner_var2=inner_var2, outer_var1=outer_var1, 
        outer_var2=outer_var2, state_lr=state_lr,
        state_inner_sampler=state_inner_sampler,
        state_outer_sampler=state_outer_sampler
    )
    carry, _ = jax.lax.scan(
        soba_one_iter,
        init=init,
        xs=None,
        length=max_iter,
    )

    return (
        carry['inner_var'], carry['outer_var'], carry['v'],
        carry['v1'], carry['v2'], carry['inner_var1'],
        carry['inner_var2'], carry['outer_var2'], carry['outer_var2'],
        {k: v for k, v in carry.items()
         if k not in ['inner_var', 'outer_var', 'v', 'v1', 'v2', 'inner_var1', 'inner_var2', 'outer_var1', 'outer_var2']}
    )
