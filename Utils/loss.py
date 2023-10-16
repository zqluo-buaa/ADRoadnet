import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import logging
from Utils.metrics import *

class FocalLoss(nn.Module):
    def init(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # 其中ignore_index表示忽略某个类别的损失,none表示输出每个元素的损失
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha*(1-pt)**self.gamma*ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class l2_reg_loss(nn.Module):
    def __init__(self, weight_decay=0.0005):
        super(l2_reg_loss, self).__init__()

        self.weight_decay = weight_decay
        self.reg_loss = 0

    def forward(self, model):
        self.reg_loss = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                self.reg_loss = self.reg_loss + torch.sum(abs(param)**2)/2.

        return self.reg_loss * self.weight_decay  # 对权重做惩罚，与样本数无关

    def reset(self):
        self.reg_loss = 0

class BCEDiceLoss(nn.Module):
    def __init__(self, gamma=0.5, weight=1):
        super(BCEDiceLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        # BCE loss
        # bce_loss = Balanced_CE_loss()(input, target).double()
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.weight]).cuda())(input, target).double()

        pred = torch.sigmoid(input).view(-1)  # 压平
        truth = target.view(-1)

        smooth = 1.
        # Dice Loss
        dice_coef = 2.0 * ((pred * truth).double().sum() + smooth) / (
            pred.double().sum() + truth.double().sum() + smooth
        )  # .double()转换为double类型

        return (1-self.gamma)*bce_loss + (1 - dice_coef)*self.gamma

class MSEDiceLoss(nn.Module):
    def __init__(self, gamma=0.5, weight=1):
        super(MSEDiceLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        # BCE loss
        # bce_loss = Balanced_CE_loss()(input, target).double()
        # bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.weight]).cuda())(input, target).double()

        pred = torch.sigmoid(input).view(-1)  # 压平
        truth = target.view(-1)

        mse_loss = nn.MSELoss()(pred, truth)

        smooth = 1e-8
        # Dice Loss
        dice_coef = 2.0 * ((pred * truth).double().sum() + smooth) / (
            pred.double().sum() + truth.double().sum() + smooth
        )  # .double()转换为double类型

        return (1 - self.gamma) * mse_loss + (1 - dice_coef) *self.gamma


class Overlap_MSEDiceLoss(nn.Module):
    def __init__(self, gamma=0.5, weight=1):
        super(Overlap_MSEDiceLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        # BCE loss
        # bce_loss = Balanced_CE_loss()(input, target).double()

        # bce_loss = nn.BCEWithLogitsLoss(torch.tensor([self.weight]).cuda())(input, target).double()

        pred = (torch.sigmoid(input) * target.data).view(-1)  # 压平
        truth = target.view(-1)

        mse_loss = nn.MSELoss()(pred, truth)

        smooth = 1e-8
        # Dice Loss
        dice_coef = 2.0 * ((pred * truth).double().sum() + smooth) / (
            pred.double().sum() + truth.double().sum() + smooth
        )  # .double()转换为double类型

        return self.gamma*mse_loss + (1 - dice_coef)*(1-self.gamma)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):

        pred = torch.sigmoid(input).view(-1)  # 压平
        truth = target.view(-1)

        smooth = 1
        # Dice Loss
        dice_coef = 2.0 * ((pred * truth).double().sum() + smooth) / (
            pred.double().sum() + truth.double().sum() + smooth
        )  # .double()转换为double类型

        return (1 - dice_coef)

class BoundaryBCELoss(nn.Module):
    def __init__(self, weight=1):
        super(BoundaryBCELoss, self).__init__()
        self.weight = weight

    def forward(self, input, target, boundary):
        # BCE loss
        # bce_loss = Balanced_CE_loss()(input, target).double()
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.weight]).cuda(), reduction='none')(input, target).double()
        b_bceloss = torch.mean(bce_loss + bce_loss * boundary, dtype=torch.float32)

        return b_bceloss


class BoundaryCELoss(nn.Module):
    def __init__(self, weight=1):
        super(BoundaryCELoss, self).__init__()
        self.weight = weight

    def forward(self, input, target, boundary):
        # BCE loss
        # bce_loss = Balanced_CE_loss()(input, target).double()
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target).double()
        b_celoss = torch.mean(ce_loss + ce_loss * boundary, dtype=torch.float32)

        return b_celoss

class MultiTaskWrapper(nn.Module):
    def __init__(self, task_num=3):
        super(MultiTaskWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, losses):

        precision1 = torch.exp(-self.log_vars[0])
        loss1 = torch.sum(precision1 * losses[0] +self.log_vars[0], -1)

        precision2 = torch.exp(-self.log_vars[1])
        loss2 = torch.sum(precision2 * losses[1] +self.log_vars[1], -1)

        precision3 = torch.exp(-self.log_vars[2])
        loss3 = torch.sum(precision3 * losses[2] + self.log_vars[2], -1)

        return loss1, loss2, loss3


from typing import Tuple
alpha_to_threshold = (
    lambda alpha: round(1.0 / (1.0 - alpha)) if alpha != 1.0 else float("inf")
)
class RunningStats(nn.Module):
    """
    Utility class to compute running estimates of mean/stdev of torch.Tensor.
    """

    def __init__(
        self,
        compute_stdev: bool = False,
        shape: Tuple[int, ...] = None,
        condense_dims: Tuple[int, ...] = (),
        cap_sample_size: bool = False,
        ema_alpha: float = 0.999,
    ) -> None:
        """
        Init function for RunningStats. The running mean will always be computed, and a
        running standard deviation is also computed if `compute_stdev = True`.
        Arguments
        ---------
        compute_stdev : bool
            Whether or not to compute a standard deviation along with a mean.
        shape : Tuple[int, ...]
            The shape of the tensors that we will be computing stats over.
        condense_dims : Tuple[int, ...]
            The indices of dimensions to condense. For example, if `shape=(2,3)` and
            `condense=(1,)`, then a tensor `val` with shape `(2, 3)` will be treated as
            3 samples of a random variable with shape `(2,)`.
        cap_sample_size : bool
            Whether or not to stop increasing the sample size when we switch to EMA.
            This may be helpful because an EMA weights recent samples more than older
            samples, which can increase variance. To offset this, we can leave the
            sample size at a fixed value so that the sample size reflects the level of
            variance.
        ema_alpha : float
            Coefficient used to compute exponential moving average. We compute an
            arithmetic mean for the first `ema_threshold` steps (as computed below),
            then switch to EMA. If `ema_alpha == 1.0`, then we will never switch to EMA.
        """

        super(RunningStats, self).__init__()

        self.compute_stdev = compute_stdev
        self.condense_dims = condense_dims
        self.cap_sample_size = cap_sample_size
        self.ema_alpha = ema_alpha
        self.ema_threshold = alpha_to_threshold(ema_alpha)

        self.shape = shape
        self.condensed_shape = tuple(
            [shape[i] for i in range(len(shape)) if i not in condense_dims]
        )
        self.register_buffer("mean", torch.zeros(self.condensed_shape))
        if self.compute_stdev:
            self.register_buffer("square_mean", torch.zeros(self.condensed_shape))
            self.register_buffer("var", torch.zeros(self.condensed_shape))
            self.register_buffer("stdev", torch.zeros(self.condensed_shape))

        # Used to keep track of number of updates and effective sample size, which may
        # stop decreasing when we switch to using exponential moving averages.
        self.register_buffer("num_steps", torch.zeros(self.condensed_shape))
        self.register_buffer("sample_size", torch.zeros(self.condensed_shape))

    def update(self, val: torch.Tensor, flags: torch.Tensor = None) -> None:
        """
        Update running stats with new value.
        Arguments
        ---------
        val : torch.Tensor
            Tensor with shape `self.shape` representing a new sample to update running
            statistics.
        flags : torch.Tensor
            Tensor with shape `self.condensed_shape` representing whether or not to
            update the stats at each element of the stats tensors (0/False for don't
            update and 1/True for update). This allows us to only update a subset of the
            means/stdevs in the case that we receive a sample for some of the elements,
            but not all of them.
        """

        if flags is None:
            flags = torch.ones(self.condensed_shape, device=self.sample_size.device)

        # Update `self.num_steps` and `self.sample_size`.
        self.num_steps += flags
        if self.cap_sample_size:
            below = self.sample_size + flags < self.ema_threshold
            above = torch.logical_not(below)
            self.sample_size = (
                self.sample_size + flags
            ) * below + self.ema_threshold * above
        else:
            self.sample_size += flags

        # Condense dimensions of sample if necessary.
        if len(self.condense_dims) > 0:
            new_val = torch.mean(val, dim=self.condense_dims)
            if self.compute_stdev:
                new_square_val = torch.mean(val ** 2, dim=self.condense_dims)
        else:
            new_val = val
            if self.compute_stdev:
                new_square_val = val ** 2

        # Update stats.
        self.mean = self.single_update(self.mean, new_val, flags)
        if self.compute_stdev:
            self.square_mean = self.single_update(
                self.square_mean, new_square_val, flags
            )
            self.var = self.square_mean - self.mean ** 2
            self.stdev = torch.sqrt(self.var)

    def single_update(
        self, m: torch.Tensor, v: torch.Tensor, flags: torch.Tensor
    ) -> torch.Tensor:
        """
        Update a mean, either through computing the arithmetic mean or an exponential
        moving average.
        """

        below = self.num_steps <= self.ema_threshold
        above = torch.logical_not(below)
        if torch.all(below):
            new_m = (m * (self.num_steps - 1) + v) / self.num_steps
        elif torch.all(above):
            new_m = m * self.ema_alpha + v * (1.0 - self.ema_alpha)
        else:
            arithmetic = (m * (self.num_steps - 1) + v) / self.num_steps
            ema = m * self.ema_alpha + v * (1.0 - self.ema_alpha)
            new_m = arithmetic * below + ema * above

        # Trick to set nans to zero (in case self.num_steps = 0 for any elements), since
        # these values can only be overwritten in the return statement if set to zero.
        nan_indices = self.num_steps == 0
        if torch.any(nan_indices):
            new_m[nan_indices] = 0

        return new_m * flags + m * torch.logical_not(flags)

    def forward(self, x: torch.Tensor) -> None:
        """
        Forward function for RunningStats. This should never be used. It's super hacky,
        but we made RunningStats a subclass of Module so that we could enjoy the
        benefits like having the device set automatically when RunningStats is a member
        of a Module for which to() is called.
        """
        raise NotImplementedError

from typing import List, Dict, Any
class LossWeighter(nn.Module):
    """ Compute task loss weights for multi-task learning. """

    def __init__(self, num_tasks: int, loss_weights: List[float]) -> None:
        """ Init function for LossWeighter. """

        super(LossWeighter, self).__init__()

        # Set state.
        self.num_tasks = num_tasks
        if loss_weights is not None:
            assert len(loss_weights) == self.num_tasks
            loss_weights = torch.Tensor(loss_weights)
        else:
            loss_weights = torch.ones((self.num_tasks,))
        self.register_buffer("loss_weights", loss_weights)
        self.register_buffer("initial_loss_weights", torch.clone(self.loss_weights))
        self.total_weight = float(torch.sum(self.loss_weights))
        self.loss_history = []
        self.steps = 0
        self.MAX_HISTORY_LEN = 2

    def update(self, loss_vals: torch.Tensor, **kwargs: Dict[str, Any]) -> None:
        """
        Compute new loss weights using most recent values of task losses. Extra
        arguments are passed to `self._update_weights()`.
        """
        self.loss_history.append(loss_vals.detach())
        self.loss_history = self.loss_history[-self.MAX_HISTORY_LEN :]
        # if (
        #     isinstance(self, GradNorm)
        #     or isinstance(self, SLW)
        #     or isinstance(self, SLAWTester)
        # ):
        #     kwargs["loss_vals"] = loss_vals
        self._update_weights(**kwargs)
        self.steps += 1

    def _update_weights(self) -> None:
        """ Update loss weights. Should be implemented in subclasses. """
        raise NotImplementedError

EPSILON = 1e-5
class SLAW(LossWeighter):
    """
    Compute task loss weights with Centered Loss Approximated Weighting. Here we keep a
    running std of each task's loss, and set each task's loss weight equal to the
    inverse of the std of the task loss.
    """

    def __init__(self, ema_alpha: float = 0.99, **kwargs: Dict[str, Any]) -> None:
        """ Init function for SLAW. """
        super(SLAW, self).__init__(**kwargs)

        self.loss_stats = RunningStats(
            compute_stdev=True, shape=(self.num_tasks,), ema_alpha=ema_alpha,
        )

    def _update_weights(self) -> None:
        """ Compute new loss weights with SLAW. """

        # Update stats.
        self.loss_stats.update(self.loss_history[-1])

        # Set loss weights equal to inverse of loss stdev, then normalize the weights so
        # they sum to the initial total weight. Note that we don't update the weights
        # until after the first step, since at that point each stdev is undefined.
        if self.steps > 0 and not any(torch.isnan(self.loss_stats.stdev)):
            threshold_stdev = torch.max(
                self.loss_stats.stdev, EPSILON * torch.ones_like(self.loss_stats.stdev)
            )
            self.loss_weights = 1.0 / threshold_stdev
            self.loss_weights /= torch.sum(self.loss_weights)
            self.loss_weights *= self.total_weight


