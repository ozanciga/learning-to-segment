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


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

validation_params = {
    'ph': args.tile_h * args.scan_resize,  # patch height (y)
    'pw': args.tile_w * args.scan_resize,  # patch width (x)
    'sh': args.tile_stride_h,  # slide step (dy)
    'sw': args.tile_stride_w,  # slide step (dx)
}

iterator_val = ds.Dataset_wsis(args.val_image_pth, validation_params)


def train(seg_and_cls, use_seg_ratio, exp_iter):

    args.seg_and_cls = seg_and_cls
    args.use_seg_ratio = use_seg_ratio
    args.duplicate_trainset = 1

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
    iterator_train = ds.GenerateIterator(
        args.train_image_pth,
        use_seg_ratio=args.use_seg_ratio,
        duplicate_dataset=args.duplicate_trainset,
        use_cls=args.seg_and_cls
    )
    args.cls_weights_cls, args.cls_weights_seg = preprocessing.cls_weights(
        iterator_train, ignore_cls=not args.seg_and_cls)

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
    print(args)

    args.experiment_save_pth = '{}/{}'.format(args.experiments_pth, args.experiment_time)
    ufs.make_folder(args.experiment_save_pth)
    exp_settings = open('{}/config.txt'.format(args.experiment_save_pth), "w+")
    exp_settings.write(str(args))
    exp_settings.close()

    datalist = iterator_train.dataset.datalist[::args.duplicate_trainset]
    exp_imagelist = open('{}/image_list.txt'.format(args.experiment_save_pth), "w+")
    exp_imagelist.write('Cls images: ' + str(len([item for item in datalist if item['mask'] is None])))
    exp_imagelist.write('\nSeg images: ' + str(len([item for item in datalist if item['mask'] is not None])))
    for item in datalist:
        exp_imagelist.write('\n')
        exp_imagelist.write(json.dumps(item))
    exp_imagelist.close()


    for epoch in range(start_epoch, 1+args.num_epoch):

        sum_loss_cls, sum_loss_reg, sum_loss_seg = 0, 0, 0

        progress_bar = tqdm(iterator_train, disable=True)

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
            val.predict_wsis(model, iterator_val, epoch)

        if args.save_models > 0 and epoch % args.save_models == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': args
            }
            torch.save(state, '{}/model_{}_{}_{}.pt'.format(args.model_save_pth, args.arch_encoder, epoch, args.experiment_time))


if __name__ == "__main__":
    for use_seg_ratio in tqdm([0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1]):
        for exp_iter in [1, 2, 3, 4, 5]:
            for seg_and_cls in [0, 1, 2]:
                train(seg_and_cls, use_seg_ratio, exp_iter)
