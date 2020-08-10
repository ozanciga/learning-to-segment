import torch
import utils.dataset as ds
from myargs import args
from tqdm import tqdm
import os
import utils.eval as val
import utils.networks as networktools
import segmentation_models_pytorch as smp
import models.losses as losses
from models import optimizers
from utils import preprocessing
from models.models import Classifier

#from apex import amp
#amp_handle = amp.init(enabled=True)



os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids


def train():

    args.val_save_pth = 'data/val/out2'

    ' model setup '
    def activation(x):
        x
    model = eval('smp.'+args.model_name)(
        args.arch_encoder,
        encoder_weights='imagenet',
        classes=args.num_classes,
        activation=activation,
    )
    model.classifier = Classifier(model.encoder.out_shapes[0], args.num_classes)
    optimizer = optimizers.optimfn(args.optim, model)

    model, optimizer, start_epoch = networktools.continue_train(model, optimizer,
                                                                args.train_model_pth, args.continue_train)
    ' losses '
    cls_weights_cls, cls_weights_seg = preprocessing.cls_weights(args.train_image_pth)

    params = {
        'reduction': 'mean',
        'alpha': torch.Tensor(cls_weights_cls),
        'xent_ignore': -1,
    }
    lossfn_cls = losses.lossfn(args.loss, params).cuda()

    params = {
        'reduction': 'mean',
        'alpha': torch.Tensor(cls_weights_seg),
        'xent_ignore': -1,
    }
    lossfn_seg = losses.lossfn(args.loss, params).cuda()

    ' datasets '
    validation_params = {
        'ph': args.tile_h * args.scan_resize,  # patch height (y)
        'pw': args.tile_w * args.scan_resize,  # patch width (x)
        'sh': args.tile_stride_h,     # slide step (dy)
        'sw': args.tile_stride_w,     # slide step (dx)
    }
    iterator_train = ds.GenerateIterator(args.train_image_pth, duplicate_dataset=1)
    iterator_val = ds.Dataset_wsis(args.raw_val_pth, validation_params)

    model = model.cuda()

    ' current run train parameters '
    print(args)

    for epoch in range(start_epoch, 1+args.num_epoch):

        sum_loss = 0
        progress_bar = tqdm(iterator_train, disable=False)

        for batch_it, (image, label, is_cls, cls_code) in enumerate(progress_bar):

            image = image.cuda()
            label = label.cuda()
            is_cls = is_cls.type(torch.bool).cuda()
            cls_code = cls_code.cuda()

            # pass images through the network (cls)
            encoding = model.encoder(image)

            loss = 0

            if torch.nonzero(is_cls).size(0) > 0:
                pred_cls = model.classifier(encoding[0][is_cls, ...])
                loss = loss + lossfn_cls(pred_cls, cls_code[is_cls])

            if torch.nonzero(~is_cls).size(0) > 0:
                pred_seg = model.decoder([x[~is_cls, ...] for x in encoding])
                loss = loss + lossfn_seg(pred_seg, label[~is_cls])

            sum_loss = sum_loss + loss.item()

            optimizer.zero_grad()
            loss.backward()
            #with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            optimizer.step()

            progress_bar.set_description('ep. {}, cls loss: {:.3f}'.format(epoch, sum_loss/(batch_it+args.epsilon)))

        ' test model accuracy '
        if  args.validate_model > 0 and epoch % args.validate_model == 0:
            val.predict_wsis(model, iterator_val, epoch)

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
