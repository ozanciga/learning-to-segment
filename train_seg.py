import torch
import utils.dataset as ds
from myargs import args
import os
import utils.networks as networktools
import models.losses as losses
from models import optimizers
from models import models
from utils import preprocessing
import resnet
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

args.seg_features = 256 * 1
args.use_seg_ratio = 1
args.exp_iter = 0
args.duplicate_dataset = 1
args.seg_and_cls = False

def train():

    ' model setup '
    model = resnet.resnet18(pretrained=False, progress=False)
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
        duplicate_dataset=args.duplicate_dataset,
        use_cls=args.seg_and_cls,
    )
    iterator_val = ds.GenerateIterator(
        args.val_image_pth,
        use_seg_ratio=args.use_seg_ratio,
        duplicate_dataset=args.duplicate_dataset,
        use_cls=args.seg_and_cls,
        eval=True
    )

    args.cls_weights_cls, args.cls_weights_seg = preprocessing.cls_weights(
        iterator_train, ignore_cls=not args.seg_and_cls)

    args.cls_weights_seg = [1, 1, 1, 1]

    ' losses '
    params = {
        'reduction': 'mean',
        'weights': torch.Tensor(args.cls_weights_seg),
        'xent_ignore': -1,
    }
    lossfn_seg = losses.lossfn('xent', params).cuda()

    model = model.cuda()

    ' current run train parameters '
    print(args)

    train_loss, val_loss = [], []
    val_epoch = []

    for epoch in tqdm(range(start_epoch, 1+args.num_epoch)):

        sum_loss_seg = 0

        for batch_it, (image, label, mask, is_cls) in enumerate(iterator_train):

            image = image.cuda()
            mask = mask.cuda()

            ''' segmentation '''
            out = model.c1(model(image))[1]
            loss_seg = lossfn_seg(out, mask)

            loss_seg_ = torch.argmax(out, 1)
            loss_seg_ = 1 - torch.mean((loss_seg_ == mask).float())

            sum_loss_seg += loss_seg_.item()

            optimizer.zero_grad()
            loss_seg.backward()
            optimizer.step()

        train_loss.append(sum_loss_seg/len(iterator_train))

        ' test model accuracy '
        if args.validate_model > 0 and epoch % args.validate_model == 0:

            model.eval()

            with torch.no_grad():

                sum_loss_seg = 0
                for batch_it, (image, label, mask, is_cls) in enumerate(iterator_val):
                    image = image.cuda()
                    mask = mask.cuda()
                    ''' segmentation '''
                    #loss_seg = lossfn_seg(model.c1(model(image))[1], mask)
                    loss_seg = torch.argmax(model.c1(model(image))[1], 1)
                    loss_seg = 1-torch.mean((loss_seg == mask).float())

                    sum_loss_seg += loss_seg.item()

                val_epoch.append(epoch)
                val_loss.append(sum_loss_seg / len(iterator_val))

                model.train()

                if epoch % 50 == 0:
                    plt.clf()
                    plt.plot(np.arange(1, len(train_loss) + 1), train_loss, 'r', lw=3)
                    plt.plot(val_epoch, val_loss, 'b', lw=3)
                    plt.grid('on')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend(['Train', 'Val'])
                    plt.show()

    print(train_loss)
    print(val_loss)


if __name__ == "__main__":
    train()
