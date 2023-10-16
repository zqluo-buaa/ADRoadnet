from torch.optim.lr_scheduler import _LRScheduler, StepLR
import math

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_iters, power=0.9, min_lr=1e-8):
        self.base_lr = base_lr
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        self.optimizer = optimizer

    def update(self, c_iters=0):
        self.c_iters = c_iters

        self.optimizer.param_groups[0]['lr'] = \
            max(self.base_lr * (1 - self.c_iters / self.max_iters) ** self.power, self.min_lr)

class ConsineAnnWithWarmup(_LRScheduler):
    def __init__(self, optimizer, loader_len, lr_max, lr_min=0,  epo_tot = 60, epo_mult=1.5, epo_cur=0, warm_steps=1, warm_lr=0, warm_prefix=False, cycle_decay=0.5):

        self.optimizer = optimizer
        self.loader_len = loader_len
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.epo_cur = epo_cur
        self.epo_tot = epo_tot
        self.epo_mult = epo_mult
        self.warm_steps = warm_steps
        self.warm_lr = warm_lr
        self.cycle_decay = cycle_decay
        self.run_i = 1
        self.warm_prefix = warm_prefix

        super(ConsineAnnWithWarmup, self).__init__(optimizer)  # 初始化就会step一次，调用一次get_lr

    def get_lr(self):

        self.epo_cur += 1/self.loader_len
        param_num = len(self.optimizer.param_groups)

        if self.warm_prefix:
            if self.epo_cur <= self.warm_steps:
                return [self.epo_cur/self.warm_steps* self.lr_max]*param_num
                pass
            else:
                self.epo_cur = 1/self.loader_len  # finish warm up
                self.warm_prefix = False

        elif self.epo_cur > self.epo_tot:
            '''restart'''
            self.run_i += 1
            self.lr_restart(run_i=self.run_i)

        return [self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * self.epo_cur/self.epo_tot))]*param_num

    def lr_restart(self, run_i):
        self.lr_max = self.lr_max * self.cycle_decay
        self.lr_min = self.lr_min * self.cycle_decay
        self.epo_tot = self.epo_tot * self.epo_mult
        self.warm_prefix = True
        self.epo_cur = 1/self.loader_len



