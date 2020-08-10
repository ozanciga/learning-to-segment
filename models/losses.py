import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import preprocessing
from myargs import args

import numpy as np  # todo: find alternative to np rounding (see weighted mse)

def lossfn(lossname, params=None):

    params = params if params is not None else {
        'reduction': 'mean',
        'ratio': 0.5,
        'scale_factor': 1/16,
        'gamma': 2,
        'ignore_index': -1,
        'xent_ignore': -1,
        'weights': torch.ones(args.num_classes)
    }

    params = preprocessing.DotDict(params)

    losses = {
        'xent': lambda x: nn.CrossEntropyLoss(reduction=params.reduction, weight=params.weights, ignore_index=params.xent_ignore),
        'bce': lambda x: nn.BCELoss(reduction=params.reduction),
        'focal': lambda x: FocalLoss2d(gamma=params.gamma, weights=params.weights, reduction=params.reduction),
        'ohem': lambda x: OHEM(ratio=params.ratio, scale_factor=params.scale_factor),
        'cent': lambda x: ConditionalEntropyLoss(weights=params.weights, reduction=params.reduction),
        'dice': lambda x: DiceLoss(weights=params.weights, ignore_index=params.ignore_index),
        'jaccard': lambda x: JaccardLoss(),
        'tversky': lambda x: TverskyLoss(),
        'zeroloss': lambda x: ZeroLoss(),
        'mse': lambda x: nn.MSELoss(reduction=params.reduction),
        'wmse': lambda x: WeightedMseLoss(params.weights),
        'l1': lambda x: nn.L1Loss(reduction=params.reduction),
        'logcosh': lambda x: LogCoshLoss(),
        'xtanh': lambda x: XTanhLoss(),
        'xsigmoid': lambda x: XSigmoidLoss(),
        'rmse': lambda x: RMSELoss(),
    }
    return losses[lossname](0)



'''
regression losses
'''

class WeightedMseLoss(torch.nn.Module):
    def __init__(self, weights):
        super(WeightedMseLoss, self).__init__()
        self.weights = weights

    def forward(self, x, y):

        out = (x - y) ** 2

        # todo: mapping here is extremely inefficient
        mapped_weights = list(map(lambda u: self.weights[np.round(u * 10 ** 2) / 10 ** 2], y.tolist()))
        mapped_weights = torch.tensor(mapped_weights, device='cuda')

        out = out * mapped_weights

        loss = out.mean(0)

        return loss


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.criterion = torch.nn.MSELoss()

    def forward(self, x, y):
        loss = torch.sqrt(self.criterion(x, y))
        return loss


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * torch.tanh(ey_t))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)


'''
classification/discrete losses
'''

####################################################
##### This is focal loss class for multi class #####
##### University of Tokyo Doi Kento            #####
####################################################
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py
class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, weights=None, reduction=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.reduction = reduction
        self.weights = weights

        if isinstance(weights, (float, int)): self.weights = torch.Tensor([weights, 1 - weights])
        if isinstance(weights, list): self.weights = torch.Tensor(weights)

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.weights is not None:
            if self.weights.type() != input.data.type():
                self.weights = self.weights.type_as(input.data)
            at = self.weights.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -((1-pt)**self.gamma) * logpt

        if self.reduction == 'mean':
            return loss.mean()

        return loss.view(-1, input.size(0))


class OHEM(torch.nn.NLLLoss):
    """ Online hard example mining."""

    def __init__(self, ratio, scale_factor=0.125):
        super(OHEM, self).__init__(None, True)
        self.ratio = ratio
        self.scale_factor = scale_factor

    def forward(self, x, y, ratio=None):
        if ratio is not None:
            self.ratio = ratio

        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        y = torch.nn.functional.interpolate(y.unsqueeze_(0).float(), mode='nearest', scale_factor=self.scale_factor).long().squeeze_(0)

        num_inst = x.size(0)

        num_hns = int(self.ratio * num_inst)
        x_ = x.clone()
        inst_losses = torch.autograd.Variable(torch.zeros(num_inst)).cuda()
        for idx, label in enumerate(y.data):
            inst_losses[idx] = -x_.data[idx, label].mean()
        _, idxs = inst_losses.topk(num_hns)
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)
        if x_hn.size(0) == 0:
            return torch.nn.functional.cross_entropy(x, y) * 0
        return torch.nn.functional.cross_entropy(x_hn, y_hn)


class ConditionalEntropyLoss(torch.nn.Module):
    """ conditional entropy + cross entropy combined ."""

    def __init__(self, weights, reduction):
        super(ConditionalEntropyLoss, self).__init__()
        self.weights = weights
        self.reduction = reduction
        self.weights = self.weights.cuda()

    def forward(self, x, y):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        loss = -1.0 * b + F.cross_entropy(x, y, reduction='none', weight=self.weights)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss

class ZeroLoss(torch.nn.Module):
    def __init__(self):
        super(ZeroLoss, self).__init__()

    def forward(self, x, y):
        return torch.tensor(0.)



class TverskyLoss(torch.nn.Module):
    'tversky loss (weighted dice)'
    '''
    It is noteworthy that in the case of α = β = 0.5 the Tversky index 
    simplifies to be the same as the Dice coefficient, which is also 
    equal to the F1 score. With α = β = 1, Equation 2 produces Tanimoto 
    coefficient, and setting α + β = 1 produces the set of Fβ scores. 
    Larger βs weigh recall higher than precision (by placing more emphasis
     on false negatives). We hypothesize that using higher βs in our 
     generalized loss function in training will lead to higher generalization 
     and improved performance for imbalanced data; and 
     effectively helps us shift the emphasis to lower FNs and boost recall
    '''
    def __init__(self, weights=1, beta=1):
        super(TverskyLoss, self).__init__()
        self.weights = weights  # penalty for false positives
        self.beta = beta   # penalty for false negatives
        self.eps = 1e-6

    def forward(self, x, y):

        x = F.softmax(x, 1)

        y_1h = torch.zeros_like(x).cuda()
        y_1h.scatter_(1, y.unsqueeze(dim=1), 1)

        dims = (0, 2, 3)
        intersection = torch.sum(x * y_1h, dims) + self.eps
        fps = torch.sum((x * (1-y_1h)), dims)
        fns = torch.sum((1-x) * y_1h, dims)

        denominator = intersection + self.weights * fps + self.beta * fns
        tversky_loss = intersection / denominator

        return (1-tversky_loss).mean()


class DiceLoss(torch.nn.Module):
    def __init__(self, weights=None, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.eps = 0.0001
        self.weights = weights if weights is not None else torch.ones(args.num_classes)
        self.ignore_index = ignore_index
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, output, target):

        output = F.softmax(output, 1)
        encoded_target = output.detach() * 0
        if self.ignore_index is not None:
            mask = target == self.ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if self.ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + self.eps
        loss_per_channel = self.weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)


class JaccardLoss(torch.nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()
        self.eps = 1

    def forward(self, x, y):

        x = F.softmax(x, 1)

        y_1h = torch.eye(args.num_classes)[y]
        y_1h = y_1h.type(y.type())
        y_1h = y_1h.permute(0, 3, 1, 2).float()

        dims = (0,) + tuple(range(2, y.ndimension()))

        intersection = torch.sum(x * y_1h, dims)
        cardinality = torch.sum(x, dims) + torch.sum(x, dims)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + self.eps))

        return (1 - jacc_loss)

