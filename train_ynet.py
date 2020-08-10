import torch
import utils.dataset as ds
from myargs import args
from tqdm import tqdm
import os
import utils.eval as val
import utils.networks as networktools
import models.losses as losses
from models import optimizers
from models.models import C1
from utils import preprocessing
import resnet

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

seg_and_cls = True

def train():

    ' model setup '
    model = resnet.resnet18(pretrained=True, progress=False)

    pool_size = 8
    nf = args.num_classes * pool_size * pool_size
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(nf, nf // 4),
        torch.nn.ReLU(True),
        torch.nn.Linear(nf // 4, args.num_classes)
    )

    c1 = C1(args.num_classes, 512, (args.tile_h, args.tile_w))

    optimizer = optimizers.optimfn(args.optim, model, c1)

    model, optimizer, start_epoch = networktools.continue_train(
        model,
        optimizer,
        args.train_model_pth,
        args.continue_train
    )

    ' losses '
    cls_weights_cls, cls_weights_seg, seg_cls_multiplier = preprocessing.cls_weights(args.train_image_pth)
    print(cls_weights_cls, cls_weights_seg)

    params = {
        'reduction': 'mean',
        'weights': torch.Tensor(cls_weights_cls),
        'xent_ignore': -1,
        'gamma': 2,
    }
    lossfn_cls = losses.lossfn('xent', params).cuda()

    params = {
        'reduction': 'mean',
        'weights': torch.Tensor(cls_weights_seg),
        'xent_ignore': -1,
    }
    lossfn_seg = losses.lossfn('xent', params).cuda()

    ' datasets '
    iterator_train = ds.GenerateIterator(args.train_image_pth, duplicate_dataset=5)

    validation_params = {
        'ph': args.tile_h * args.scan_resize,  # patch height (y)
        'pw': args.tile_w * args.scan_resize,  # patch width (x)
        'sh': args.tile_stride_h,     # slide step (dy)
        'sw': args.tile_stride_w,     # slide step (dx)
    }
    iterator_val = ds.Dataset_wsis('data/bach/val', validation_params)
    iterator_test = ds.Dataset_wsis('data/bach/test', validation_params)

    model = model.cuda()
    c1 = c1.cuda()

    ' current run train parameters '
    print(args)

    for epoch in range(start_epoch, 1+args.num_epoch):

        sum_loss_cls, sum_loss_reg, sum_loss_seg = 0, 0, 0

        progress_bar = tqdm(iterator_train, disable=False)

        for batch_it, (image, label, mask, is_cls) in enumerate(progress_bar):

            image = image.cuda()
            label = label.cuda()
            mask = mask.cuda()
            is_cls = is_cls.type(torch.bool).cuda()

            cls = torch.nonzero(is_cls).size(0) > 0
            seg = torch.nonzero(~is_cls).size(0) > 0

            ''' segmentation '''
            if seg_and_cls:
                if seg:
                    feature_maps = model(image[~is_cls])  # does batching with segmentation+cls images help???
                    pred_seg = c1(feature_maps)
                    loss_seg = lossfn_seg(pred_seg, mask[~is_cls])

                    sum_loss_seg += loss_seg.item()
                    optimizer.zero_grad()
                    loss_seg.backward()
                    optimizer.step()

            else:
                if seg:
                    feature_maps = model(image[~is_cls])
                    pred_seg = c1(feature_maps)
                    loss_seg = lossfn_seg(pred_seg, mask[~is_cls])

                    sum_loss_seg += loss_seg.item()
                    optimizer.zero_grad()
                    loss_seg.backward()
                    optimizer.step()

            ''' classification '''
            if seg_and_cls:
                if cls:
                    feature_maps = model(image[is_cls])
                    feature_maps_cls = c1(feature_maps)
                    feature_maps_cls = torch.nn.functional.adaptive_max_pool2d(feature_maps_cls, (pool_size, pool_size))
                    feature_maps_cls = torch.flatten(feature_maps_cls, 1)
                    pred_cls = model.fc(feature_maps_cls)
                    loss_cls = lossfn_cls(pred_cls, label[is_cls])
                    sum_loss_cls += loss_cls.item()

                    optimizer.zero_grad()
                    loss_cls.backward()
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
            val.predict_wsis(model, c1, iterator_val, epoch)
            val.predict_wsis(model, c1, iterator_test, epoch)

        if args.save_models > 0 and epoch % args.save_models == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': args
            }
            torch.save(state, '{}/model_{}_{}.pt'.format(args.model_save_pth, args.arch_encoder, epoch))


if __name__ == "__main__":
    train()
