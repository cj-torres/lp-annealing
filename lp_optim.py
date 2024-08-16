import torch
import numpy as np
import math
from scipy.special import lambertw
from typing import Optional, Tuple

_EPS = 1e-6
# Constant for - 1 / e.  This is the lowest 'z' for which principal / non-principal W
# is real valued (W(-1/e) = -1).  For any z < -1 / exp(1), W(z) = NA.
_EXP_INV = np.exp(-1)
_M_EXP_INV = -1 * _EXP_INV

_MAX_ITER = 100
def _fritsch_iteration(
    w: torch.Tensor, z: torch.Tensor, tol: float, iteration_count: int
) -> Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Root finding iteration for W(z) using Fritsch iteration.

    Args:
        w (torch.Tensor): Current value of w.
        z (torch.Tensor): Value for which the root is being found.
        tol (float): Tolerance for convergence.
        iteration_count (int): Current iteration count.

    Returns:
        Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, int]:
            - bool: Whether the iteration should stop.
            - torch.Tensor: Updated value of w.
            - torch.Tensor: Input value z.
            - torch.Tensor: Delta value.
            - int: Updated iteration count.
    """
    zn = torch.log(torch.abs(z)) - torch.log(torch.abs(w)) - w
    wp1 = w + 1.0
    q = 2.0 * wp1 * (wp1 + 2.0 / 3.0 * zn)
    q_minus_2zn = q - 2.0 * zn
    error = zn / wp1 * (1.0 + zn / q_minus_2zn)
    delta = torch.abs(error * w)
    converged = delta <= tol
    converged = converged | torch.isnan(wp1)
    should_stop_next = torch.all(converged) or (iteration_count >= _MAX_ITER)
    return should_stop_next, w * (1.0 + error), z, delta, iteration_count + 1


def _lambertw_nonprincipal_branch_nonna(z: torch.Tensor) -> torch.Tensor:
    """Computes the non-principal branch of z; only defined for z in [-1/exp(1), 0)."""
    # See eq (4.19) of Corless et al. (1996).
    # Taken from https://github.com/gmgeorg/torchlambertw/blob/main/torchlambertw/special.py
    L1 = torch.log(-z)
    L1_sq = L1 * L1

    L2 = torch.log(-L1)
    L2_sq = L2 * L2

    L3 = L2 / L1
    w = (
        L1
        - L2
        + L3
        + L3 * (-2.0 + L2) / (2.0 * L1)
        + (L3 * ((6.0 - 9.0 * L2 + 2 * L2_sq) / 6.0 * L1_sq))
    )

    stop_condition = False
    counter = 0
    while not stop_condition:
        counter += 1
        stop_condition, w, z, _, _ = _fritsch_iteration(w, z, _EPS, counter)
    # if z = _M_EXP_INV, return exactly -1.
    return torch.where(torch.abs(z - _M_EXP_INV) < _EPS, -1 * torch.ones_like(z), w)


def lp_calc(decay_rates: torch.tensor):
    return -2 / _lambertw_nonprincipal_branch_nonna(-2*decay_rates/math.e)


class LPAnnealingAGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, decay_rate=1e-3, start_lp=1.0, end_lp=None, gamma=.999):
        if end_lp is None:
            # default annealing target is point of maximum sparsity, proof will be in paper
            end_lp = -2 / lambertw(-2*decay_rate/math.e, k=-1).real
        defaults = dict(lr=lr, decay_rate=decay_rate, start_lp=start_lp, gamma=gamma, end_lp=end_lp)
        super(LPAnnealingAGD, self).__init__(params, defaults)
        for group in self.param_groups:
            group['lp'] = group['start_lp']

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lp = group['lp']
            gamma = group['gamma']
            decay_rate = group['decay_rate']
            lr = group['lr']
            end_lp = group['end_lp']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = -p.grad.data
                d_p.mul_(lr)
                p.data.add_(d_p)
                decay = self._lp_decay(p, lp, decay_rate)
                p.data.add_(decay)
            group['lp'] -= (1-gamma)*(lp-end_lp)

        return loss

    @staticmethod
    def _lp_decay(param, lp, decay_rate):
        with torch.no_grad():
            intermediate = torch.sign(param)*decay_rate*lp*torch.pow(torch.abs(param), lp-1)
            step_sz = torch.nan_to_num(intermediate, nan=0, posinf=0, neginf=0)
            overstep = torch.abs(step_sz) > torch.abs(param)
            return -(overstep*param + torch.logical_not(overstep)*step_sz)


class LPAnnealingAdam(torch.optim.Optimizer):
    def __init__(self, params, beta_1=.9, beta_2=.999, alpha=1e-3, epsilon=1e-8,
                 decay_rate=1e-3, start_lp=1.0, end_lp=None, gamma=.999):
        if end_lp is None:
            # default annealing target is point of maximum sparsity, proof will be in paper
            end_lp = -2 / lambertw(-2*decay_rate/math.e, k=-1).real
        defaults = dict(beta_1=beta_1, beta_2=beta_2, alpha=alpha, epsilon=epsilon,
                        decay_rate=decay_rate, start_lp=start_lp, end_lp=end_lp, gamma=gamma)
        super(LPAnnealingAdam, self).__init__(params, defaults)
        for group in self.param_groups:
            group['m'] = [torch.zeros_like(p) for p in group['params']]
            group['v'] = [torch.zeros_like(p) for p in group['params']]
            group['lp'] = group['start_lp']
            group['beta_1_t'] = group['beta_1']
            group['beta_2_t'] = group['beta_2']

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # Adam (hopefully)
            lp = group['lp']
            end_lp = group['end_lp']
            gamma = group['gamma']
            decay_rate = group['decay_rate']
            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            beta_1_t = group['beta_1_t']
            beta_2_t = group['beta_2_t']
            alpha = group['alpha']
            epsilon = group['epsilon']
            m = group['m']
            v = group['v']
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                m[i] = (beta_1*m[i] + (1-beta_1)*p.grad)
                v[i] = (beta_2*v[i] + (1-beta_2)*torch.pow(p.grad, 2))
                m_i_hat = m[i] / (1 - beta_1_t)
                v_i_hat = v[i] / (1 - beta_2_t)
                d_p = -m_i_hat/(torch.sqrt(v_i_hat) + epsilon)
                d_p.mul_(alpha)
                p.data.add_(d_p)
                decay = self._lp_decay(p, lp, decay_rate)
                p.data.add_(decay)
            group['lp'] -= (1-gamma)*(lp-end_lp)
            group['beta_1_t'] *= beta_1
            group['beta_2_t'] *= beta_2

        return loss

    @staticmethod
    def _lp_decay(param, lp, decay_rate):
        with torch.no_grad():
            intermediate = torch.sign(param)*decay_rate*lp*torch.pow(torch.abs(param), lp-1)
            step_sz = torch.nan_to_num(intermediate, nan=0, posinf=0, neginf=0)
            overstep = torch.abs(step_sz) > torch.abs(param)
            return -(overstep*param + torch.logical_not(overstep)*step_sz)


class AdamAS(torch.optim.Optimizer):
    # Adam with adaptive sparsity
    def __init__(self, params, beta_1=.9, beta_2=.999, alpha=1e-3, epsilon=1e-8,
                 decay_rate=1e-3, gamma=.999):

        defaults = dict(beta_1=beta_1, beta_2=beta_2, alpha=alpha, epsilon=epsilon, gamma=gamma,
                        decay_rate=decay_rate)
        super(LPAnnealingAdam, self).__init__(params, defaults)
        for group in self.param_groups:
            group['m'] = [torch.zeros_like(p) for p in group['params']]
            group['v'] = [torch.zeros_like(p) for p in group['params']]
            #group['adjusted_decay_rate'] = [torch.full_like(p, decay_rate) for p in group['params']]
            #group['lp'] = [lp_calc(dr) for dr in group['adjusted_decay_rate']]
            group['beta_1_t'] = group['beta_1']
            group['beta_2_t'] = group['beta_2']

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # Adam (hopefully)
            gamma = group['gamma']
            decay_rate = group['decay_rate']
            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            beta_1_t = group['beta_1_t']
            beta_2_t = group['beta_2_t']
            alpha = group['alpha']
            epsilon = group['epsilon']
            m = group['m']
            v = group['v']
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                m[i] = (beta_1 * m[i] + (1 - beta_1) * p.grad)
                v[i] = (beta_2 * v[i] + (1 - beta_2) * torch.pow(p.grad, 2))
                m_i_hat = m[i] / (1 - beta_1_t)
                v_i_hat = v[i] / (1 - beta_2_t)
                d_p = -m_i_hat / (torch.sqrt(v_i_hat) + epsilon)
                d_p.mul_(alpha)

                adjusted_decay_rate = (torch.sqrt(v_i_hat) + epsilon) * decay_rate
                lp = lp_calc(adjusted_decay_rate)
                
                p.data.add_(d_p)
                decay = self._lp_decay(p, lp, adjusted_decay_rate)
                p.data.add_(decay)
            group['beta_1_t'] *= beta_1
            group['beta_2_t'] *= beta_2

        return loss

    @staticmethod
    def _lp_decay(param, lp, decay_rate):
        with torch.no_grad():
            intermediate = torch.sign(param) * decay_rate * lp * torch.pow(torch.abs(param), lp - 1)
            step_sz = torch.nan_to_num(intermediate, nan=0, posinf=0, neginf=0)
            overstep = torch.abs(step_sz) > torch.abs(param)
            return -(overstep * param + torch.logical_not(overstep) * step_sz)


""" MODEL TESTS """


# 1 - element test
def test_1(n):

    model = torch.nn.Linear(n, n)
    opt = AnnealingOptim(model.parameters())
    model.train()

    num_epochs = 15000

    for i in range(num_epochs):
        in_tensors = torch.randn(500, n)
        target = torch.zeros_like(in_tensors)
        target[:, 0] = in_tensors[:, 0]

        opt.zero_grad()
        loss = ((model(in_tensors) - target).pow(2)).mean()

        loss.backward()
        opt.step()
        if i % 10 == 0:
            print(loss.item())

    model.eval()
    in_tensors = torch.randn(500, n)
    target = torch.zeros_like(in_tensors)
    print((model(in_tensors) - target).pow(2).mean().item())
    print(list(model.parameters()))


# all-element test (n^2)
def test_2(n):

    model = torch.nn.Linear(n, n)
    model.train()
    opt = AnnealingOptim(model.parameters())

    num_epochs = 15000

    for i in range(num_epochs):
        in_tensors = torch.randn(500, n)
        target = in_tensors.sum(dim=1).unsqueeze(dim=1).expand(-1, n)

        opt.zero_grad()
        loss = ((model(in_tensors) - target).pow(2)).mean()

        loss.backward()
        opt.step()
        if i % 100 == 0:
            print(loss.item())

    model.eval()
    in_tensors = torch.randn(500, n)
    target = in_tensors.sum(dim=1).unsqueeze(dim=1).expand(-1, n)
    print((model(in_tensors) - target).pow(2).mean().item())
    print(list(model.parameters()))


# identity test (n)
def test_3(n):

    model = torch.nn.Linear(n, n)
    opt = AnnealingOptim(model.parameters())

    model.train()

    num_epochs = 15000

    for i in range(num_epochs):
        in_tensors = torch.randn(500, n)
        target = in_tensors

        opt.zero_grad()
        loss = ((model(in_tensors) - target).pow(2)).mean()

        loss.backward()
        opt.step()
        if i % 100 == 0:
            print(loss.item())

    model.eval()
    in_tensors = torch.randn(500, n)
    target = in_tensors
    print((model(in_tensors) - target).pow(2).mean().item())
    print(list(model.parameters()))
