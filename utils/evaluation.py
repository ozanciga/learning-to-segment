import torch
from sklearn import metrics
import numpy as np
from tqdm import tqdm


def eval_model(model, loader, name, epoch, writer, args):

    with torch.no_grad():

        model.eval()

        preds, gts = [], []

        for batch_it, batch in enumerate(tqdm(loader, disable=1)):

            feature_maps = model(batch[0].cuda())
            _, pred_src = model.c1(feature_maps)

            pred = torch.argmax(pred_src, 1)

            preds.extend(pred.cpu().numpy())

            gts.extend(batch[2].numpy())

        preds, gts = np.asarray(preds).flatten(), np.asarray(gts).flatten()

        my_metrics = {}

        my_metrics['val_acc'] = metrics.accuracy_score(gts, preds)
        if args.num_classes == 2:
            my_metrics['val_f1'] = metrics.f1_score(gts, preds, average='binary')
        else:
            my_metrics['val_f1'] = 0

        my_metrics['val_f1_micro'] = metrics.f1_score(gts, preds, average='micro')
        my_metrics['val_f1_macro'] = metrics.f1_score(gts, preds, average='macro')

        # save epoch to tensorboard
        writer.add_scalar(f'Metrics/{name}/accuracy', my_metrics['val_acc'], epoch)
        writer.add_scalar(f'Metrics/{name}/f1', my_metrics['val_f1'], epoch)
        writer.add_scalar(f'Metrics/{name}/f1_micro', my_metrics['val_f1_micro'], epoch)
        writer.add_scalar(f'Metrics/{name}/f1_macro', my_metrics['val_f1_macro'], epoch)

    model.train()

    return my_metrics


def eval_model_cls(model, loader, name, epoch, writer, args):

    with torch.no_grad():

        model.eval()

        preds, gts = [], []

        for batch_it, batch in enumerate(tqdm(loader, disable=1)):

            feature_maps_l, _ = model.c1(model(batch[0].cuda()))
            y1 = model.fc(feature_maps_l)

            pred = torch.argmax(y1, 1)

            preds.extend(pred.cpu().numpy())

            gts.extend(batch[1].numpy())

        preds, gts = np.asarray(preds).flatten(), np.asarray(gts).flatten()

        my_metrics = {}

        my_metrics['val_acc'] = metrics.accuracy_score(gts, preds)
        if args.num_classes == 2:
            my_metrics['val_f1'] = metrics.f1_score(gts, preds, average='binary')
        else:
            my_metrics['val_f1'] = 0

        my_metrics['val_f1_micro'] = metrics.f1_score(gts, preds, average='micro')
        my_metrics['val_f1_macro'] = metrics.f1_score(gts, preds, average='macro')

        my_metrics['confusion_matrix'] = metrics.confusion_matrix(gts, preds)

        # save epoch to tensorboard
        writer.add_scalar(f'Metrics/{name}/accuracy', my_metrics['val_acc'], epoch)
        writer.add_scalar(f'Metrics/{name}/f1', my_metrics['val_f1'], epoch)
        writer.add_scalar(f'Metrics/{name}/f1_micro', my_metrics['val_f1_micro'], epoch)
        writer.add_scalar(f'Metrics/{name}/f1_macro', my_metrics['val_f1_macro'], epoch)
        writer.add_scalar(f'Metrics/{name}/confusion_matrix', my_metrics['confusion_matrix'], epoch)

        print(my_metrics['confusion_matrix'])

    model.train()

    return my_metrics