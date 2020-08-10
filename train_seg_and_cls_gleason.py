import time
import torch
import utils.dataset as ds
from myargs import args
from tqdm import tqdm
import os
import utils.eval as val
import utils.networks as networktools
import models.losses as losses
from models import optimizers
from models import models
from utils import preprocessing
import utils.filesystem as ufs
import resnet
from datetime import datetime
import json
from utils.evaluation import eval_model
from utils.dataset import Dataset_onlyseg
from torch.utils.tensorboard import SummaryWriter


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

args.use_seg_ratio, args.duplicate_dataset, args.use_cls, args.exp_iter = 0.99, 1, 0, 1

params = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': args.workers,
    'pin_memory': True,
}
iterator_val = torch.utils.data.DataLoader(
    Dataset_onlyseg(
    args.val_image_pth,
    use_seg_ratio=args.use_seg_ratio,
    use_cls=args.use_cls,
    eval=1,
), **params)


def train(seg_and_cls, use_seg_ratio, use_cls, exp_iter):

    args.dataset_name = 'digestpath2019'

    args.seg_and_cls = seg_and_cls
    args.use_seg_ratio = use_seg_ratio

    args.exp_iter = exp_iter  # for seeded randomization

    args.seg_features = 256 * 1

    now = datetime.now()
    args.experiment_time = now.strftime('%m_%d_%Y_%H_%M_%S')

    ' model setup '
    model = resnet.resnet18(pretrained=True, progress=False)
    model.fc = models.domain_classifier(args.seg_features * 4, args.num_classes)

    model.c1 = models.C1(args.num_classes, args.seg_features, (args.tile_h, args.tile_w)).cuda()

    optimizer = optimizers.optimfn(args.optim, model)

    model, optimizer, start_epoch = networktools.continue_train(
        model,
        optimizer,
        args.train_model_pth,
        args.continue_train
    )

    ' datasets '
    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.workers,
        'pin_memory': True,
    }
    iterator_train = torch.utils.data.DataLoader(
        Dataset_onlyseg(
            args.train_image_pth,
            use_seg_ratio=use_seg_ratio,
            use_cls=use_cls,
            eval=0,
        ), **params)
    args.cls_weights_cls, args.cls_weights_seg = list(range(args.num_classes)), list(range(args.num_classes)) #preprocessing.cls_weights(
        #iterator_train, ignore_cls=not args.seg_and_cls)

    ' losses '
    params = {
        'reduction': 'mean',
        'weights': torch.Tensor(args.cls_weights_cls),
        'xent_ignore': -1,
    }
    lossfn_cls = losses.lossfn('xent', params).cuda()

    params = {
        'reduction': 'mean',
        'weights': torch.Tensor(args.cls_weights_seg),
        'xent_ignore': -1,
    }
    lossfn_seg = losses.lossfn('xent', params).cuda()

    ufs.make_folder(args.val_save_pth, True)

    model = model.cuda()

    ' current run train parameters '
    #print(args)

    args.experiment_save_pth = f'{args.experiments_pth}/{args.dataset_name}/{args.experiment_time}_{exp_iter}_{args.gpu_ids}'
    ufs.make_folder(args.experiment_save_pth)
    exp_settings = open('{}/config.txt'.format(args.experiment_save_pth), "w+")
    exp_settings.write(str(args))
    exp_settings.close()

    writer = SummaryWriter(f'{args.experiment_save_pth}/{args.dataset_name}')

    datalist = iterator_train.dataset.datalist
    exp_imagelist = open('{}/image_list.txt'.format(args.experiment_save_pth), "w+")
    exp_imagelist.write('Cls images: ' + str(len([item for item in datalist if item['mask'] is None])))
    exp_imagelist.write('\nSeg images: ' + str(len([item for item in datalist if item['mask'] is not None])))
    for item in datalist:
        exp_imagelist.write('\n')
        exp_imagelist.write(json.dumps(item))
    exp_imagelist.close()

    for epoch in tqdm(range(start_epoch, 1+args.num_epoch), disable=1):

        sum_loss_cls, sum_loss_reg, sum_loss_seg = 0, 0, 0

        progress_bar = tqdm(iterator_train, disable=0)

        for batch_it, (image, label, mask, is_cls) in enumerate(progress_bar):

            image = image.cuda()
            label = label.cuda()
            mask = mask.cuda()
            is_cls = is_cls.type(torch.bool).cuda()

            cls = torch.nonzero(is_cls).size(0) > 0
            seg = torch.nonzero(~is_cls).size(0) > 0

            loss_seg, loss_cls = 0, 0

            fs = model(image)
            feature_maps_l, feature_maps_h = model.c1(fs)

            ''' classification '''
            if args.seg_and_cls:
                if cls or seg:
                    y1 = model.fc(feature_maps_l)
                    loss_cls = (lossfn_cls(y1, label))
                    sum_loss_cls += loss_cls.item()

            ''' segmentation '''
            if seg or cls:
                loss_seg = lossfn_seg(feature_maps_h, mask)
                sum_loss_seg += loss_seg.item()

            optimizer.zero_grad()
            (loss_cls+loss_seg).backward()
            optimizer.step()

            progress_bar.set_description('ep. {},'
                                         ' losses; cls: {:.2f},'
                                         ' seg: {:.2f}'.format(
                epoch,
                sum_loss_cls/(batch_it+1),
                sum_loss_seg/(batch_it+1)
            ))

        ' test model accuracy '
        if epoch >= 1 and args.validate_model > 0 and epoch % args.validate_model == 0:
            0
            #eval_model(model, iterator_val, 'val', epoch, writer, args)
            # val.predict_wsis(model, iterator_val, epoch)

        if args.save_models > 0 and epoch % args.save_models == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': args
            }
            torch.save(
                state,
                f'{args.model_save_pth}/model_{args.dataset_name}_{args.seg_and_cls}_{args.use_seg_ratio}.pt'
            )


if __name__ == "__main__":
    for use_seg_ratio in tqdm([0.25, 0.3, 0.4, 0.5, 0.75, 1.0]):  #tqdm([0.0, 0.01, 0.025, 0.05, 0.075, 0.1]):  #, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]):
        for exp_iter in [1]:#[1, 2, 3, 4, 5]:
            for seg_and_cls in [0, 1, 2]:
                train(seg_and_cls, use_seg_ratio, seg_and_cls, exp_iter)
