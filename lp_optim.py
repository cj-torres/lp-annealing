import torch


class AnnealingOptim(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, decay_rate=1e-3, start_lp=1.0, gamma=.999):
        defaults = dict(lr=lr, decay_rate=decay_rate, start_lp=start_lp, gamma=gamma)
        super(AnnealingOptim, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if 'lp' not in group:
                group['lp'] = group['start_lp']
            lp = group['lp']
            gamma = group['gamma']
            decay_rate = group['decay_rate']
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = -p.grad.data
                d_p.mul_(lr)
                p.data.add_(d_p)
                decay = self._lp_decay(p, lp, decay_rate)
                p.data.add_(decay)
                group['lp'] *= gamma

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
