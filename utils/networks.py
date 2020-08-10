import torch


def continue_train(model, opt, model_path, load_weights):
    start_epoch = 1
    if load_weights:
        state = torch.load(model_path)
        opt.load_state_dict(state['optimizer'])
        model.load_state_dict(state['state_dict'])
        start_epoch = 1 + int(state['epoch'])

    return model, opt, start_epoch
