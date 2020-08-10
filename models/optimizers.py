from torch import optim
from myargs import args
from adabound import AdaBound


def optimfn(lossname, *model):

    m_list = []
    for m in model:
        m_list = list(m.parameters()) + m_list

    losses = {
        'adam': optim.Adam(m_list,
                           lr=args.lr,
                           betas=(args.beta1, args.beta2),
                           weight_decay=args.weight_decay),
        'sgd': optim.SGD(m_list,
                         lr=args.lr,
                         momentum=args.beta1,
                         weight_decay=args.weight_decay),
        'adabound': AdaBound(m_list,
                             lr=args.lr, final_lr=0.1),
    }

    return losses[lossname]
