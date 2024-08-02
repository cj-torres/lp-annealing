import torch
import math
from scipy.special import lambertw


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
    def __init__(self, params, beta_1=.9, beta_2=.999, alpha=10e-3, epsilon=10e-8,
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
