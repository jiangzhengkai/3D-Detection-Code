import torch
from torch import nn
from lib.solver.fastai_optim import OptimWrapper
from functools import partial


def children(m: nn.Module):
    "Get children of `m`."
    return list(m.children())


def num_children(m: nn.Module) -> int:
    "Get number of children modules in `m`."
    return len(children(m))

flatten_model = lambda m: sum(map(flatten_model,m.children()),[]) if num_children(m) else [m]

get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]




def build_optimizer(config, net):
    optimizer_type = config.optimizer.type
    optimizer_config = config.optimizer.value
    
    if optimizer_type == 'rms_prop_optimizer':
        optimizer_func = partial(
            torch.optim.RMSprop,
            alpha=config.decay,
            momentum=config.momentum_optimizer_value,
            eps=config.epsilon)
    elif optimizer_type == 'momentum_optimizer':
        optimizer_func = partial(
            torch.optim.SGD,
            momentum=config.momentum_optimizer_value,
            eps=config.epsilon)
    elif optimizer_type == 'adam':
        if config.optimizer.fixed_wd:
            optimizer_func = partial(
                torch.optim.Adam, betas=(0.9, 0.99), amsgrad=optimizer_config.amsgrad)
        else:
            # regular adam
             optimizer_func = partial(
                     torch.optim.Adam, amsgrad=optimizer_config.amsgrad)

    optimizer = OptimWrapper.create(
        optimizer_func,
        3e-3,
        get_layer_groups(net),
        wd=optimizer_config.wd,
        true_wd=config.optimizer.fixed_wd,
        bn_wd=True)

    if optimizer is None:
        raise ValueError('Optimizer %s not supported.' % optimizer_type)



    return optimizer

