import torch
from torch.optim.optimizer import Optimizer, required

class gRDA(Optimizer):
    def __init__(self, params, lr=0.01, c=0.0, mu=0.7):
        defaults = dict(lr=lr, c=c, mu=mu)
        super(gRDA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(gRDA, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            c = group['c']
            mu = group['mu']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                param_state = self.state[p]
                
                if 'iter_num' not in param_state:
                    iter_num = param_state['iter_num'] = torch.zeros(1)
                    accumulator = param_state['accumulator'] = torch.FloatTensor(p.shape).uniform_(-0.1, 0.1).to(p.device)
                else:
                    iter_num = param_state['iter_num']
                    accumulator = param_state['accumulator']
                iter_num.add_(1)
                accumulator.data.add_(-group['lr'], d_p)

                l1 = c * torch.pow(torch.tensor(lr), 0.5 + mu) * torch.pow(iter_num, mu)
                new_a_l1 = torch.abs(accumulator.data) - l1.to(p.device)
                p.data = torch.sign(accumulator.data) * new_a_l1.clamp(min=0)
                    
        return loss
