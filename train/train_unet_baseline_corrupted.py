import cv2
import os

import copy
import json
import time
import torch
import click
import numpy as np

from skimage import io
from pathlib import Path
from collections import defaultdict
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split

import nn_tools.utils.config as cfg

from nn_tools.models import UnetSm
from nn_tools.utils import init_determenistic
from nn_tools.metrics.segmentation import recall_score, precision_score, f1_score
from nn_tools.process.pre import ( apply_image_augmentations, apply_only_image_augmentations )
from nn_tools.losses.segmentation import ( cross_entropy_with_logits_loss, 
                                           focal_with_logits_loss,
                                           dice_with_logits_loss )

config = dict()

debugging_info = { 'epoch_loss_trace': list(),
                   'epoch_score_trace': list(),
                   'epoch_additional_score_trace': list(),
                   'epoch_times': list(),
                   'max_memory_consumption': 0.,
                 }

def init_precode():
    import albumentations as A

    height = config['IMAGE_SIZE']
    width = int(config['IMAGE_SIZE'] / 0.7586206896551724)

    if width % 2 == 1:
        width += 1

    # CLAHE (clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5)
    # Equalize (mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.5)
    # GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5)
    # HueSaturationValue (hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5)
    # ISONoise (color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5)
    # Posterize (num_bits=4, always_apply=False, p=0.5)
    # RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5)
    # RandomGridShuffle (grid=(3, 3), always_apply=False, p=0.5)
    # RandomToneCurve (scale=0.1, always_apply=False, p=0.5)
    # Superpixels (p_replace=0.1, n_segments=100, max_size=128, interpolation=1, always_apply=False, p=0.5)

    # FAILED

    # A.Emboss( alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.16 )
    # A.ChannelDropout( channel_drop_range=(1, 1), fill_value=0, p=0.16 ),
    # A.RGBShift( r_shift_limit=20, g_shift_limit=20,  b_shift_limit=20, p=0.16 )
    # A.ToSepia (always_apply=False, p=0.5)

    from augs import augs
    
    config['TRAIN_AUG'] = augs[config['AUG']]

#     config['TRAIN_AUG'] = A.Compose([
#         # A.CLAHE( clip_limit=4.0,
#         #          tile_grid_size=(8, 8), 
#         #          p=0.33 ),
#         A.Flip(p=0.75),
#         A.OneOf(
#             [ A.Sharpen( p=0.5 ),
# #               A.Blur ( blur_limit=5,
# #                        p=0.16 ),
#               A.GaussianBlur( blur_limit=(3, 7),
#                               sigma_limit=0,
#                               p=0.5 ),
#             #   A.GlassBlur( sigma=0.7,
#             #                max_delta=4,
#             #                iterations=1,
#             #                mode='fast',
#             #                p=0.16),
#             #   A.MedianBlur( blur_limit=7,
#             #                 p=0.16 ),
#             #   A.MotionBlur( blur_limit=7,
#             #                 p=0.16 ) # ? 
#             ], p=0.32),
#         A.ShiftScaleRotate( shift_limit=0.03,
#                             scale_limit=0.0,
#                             rotate_limit=10,
#                             interpolation=cv2.INTER_CUBIC,
#                             border_mode=cv2.BORDER_CONSTANT,
#                             value=0,
#                             mask_value=0,
#                             p=0.5 ),
# #         A.HueSaturationValue( hue_shift_limit=20,
# #                               sat_shift_limit=0,
# #                               val_shift_limit=0, 
# #                               p=0.5 ),
#         A.Resize( height=height, 
#                   width=width, 
#                   p=1. ),
#         A.ChannelShuffle(p=0.83),
#         A.Solarize( threshold=128, p=0.16 ),
#         A.InvertImg(p=0.16),
#         A.ToGray(p=0.16),
#         A.OneOf(
#             [ A.Cutout( num_holes=8,
#                         max_h_size=int(height*0.015),
#                         max_w_size=int(width*0.015),
#                         fill_value=0, 
#                         p=0.5 ),
#               A.PixelDropout( dropout_prob=0.01,
#                               per_channel=False,
#                               drop_value=0,
#                               mask_drop_value=0,
#                               p=0.5)
#             ], p=0.32),
#     ])

    config['EVAL_AUG'] = A.Compose([
        A.Resize( height=height, 
                  width=width, 
                  p=1. ),
    ])

def init_global_config(**kwargs):
    cfg.init_timestamp(config)
    cfg.init_run_command(config)
    cfg.init_kwargs(config, kwargs)
    cfg.init_logging(config, __name__, config['LOGGER_TYPE'], filename=config['PREFIX']+'logger_name.txt')
    cfg.init_device(config)
    cfg.init_verboser(config, logger=config['LOGGER'])
    cfg.init_options(config)

class EyeDataset(Dataset):
    def __init__(self, datadir, names, aug = None):
        self.datadir = datadir
        self.aug = aug
        self._imgpathes = [ datadir / (name + '.png') for name in names ]
        self._maskpathes = [ datadir / (name + '.geojson') for name in names ]

    @staticmethod
    def parse_polygon(coordinates: dict, image_size: tuple) -> np.ndarray:
        mask = np.zeros(image_size, dtype=np.float32)
        if len(coordinates) == 1:
            points = [np.int32(coordinates)]
            cv2.fillPoly(mask, points, 1)
        else:
            for polygon in coordinates:
                points = [np.int32([polygon])]
                cv2.fillPoly(mask, points, 1)
        return mask

    @staticmethod
    def parse_mask(shape, image_size):
        mask = np.zeros(image_size, dtype=np.float32)
        coordinates = shape['coordinates']
        if shape['type'] == 'MultiPolygon':
            for polygon in coordinates:
                mask += EyeDataset.parse_polygon(polygon, image_size)
        else:
            mask += EyeDataset.parse_polygon(coordinates, image_size)

        return mask

    def read_layout(self, maskpath, image_size):
        with open(maskpath, 'r', encoding='cp1251') as f:
            json_contents = json.load(f)

        mask = np.zeros(image_size, dtype=np.float32)

        if type(json_contents) == dict and json_contents['type'] == 'FeatureCollection':
            features = json_contents['features']
        elif type(json_contents) == list:
            features = json_contents
        else:
            features = [json_contents]

        for shape in features:
            mask += self.parse_mask(shape['geometry'], image_size)

        return mask.clip(max=1)
        
    def __getitem__(self, idx):
        imgpath = self._imgpathes[idx]
        img = io.imread(imgpath)

        maskpath = self._maskpathes[idx]
        mask = self.read_layout(maskpath, img.shape[:-1])

        if self.aug is not None:
            auged = self.aug(image=img, mask=mask)
            img, mask = auged['image'], auged['mask']

        img = np.moveaxis(img, -1, 0)
        
        return img, mask

    def __len__(self):
        return len(self._imgpathes)

def collate_fn(batch):
    imgs, masks = zip(*batch)

    imgs = torch_float(imgs, torch.device('cpu'))
    masks = torch_long(masks, torch.device('cpu'))

    return imgs, masks

def load_data_with_labels(data):
    datapath = Path(os.getenv('DATAPATH'))
    splitpath = Path(config['SPLITPATH'])
    
    with open(splitpath) as f:
        split = json.load(f)

    data['train'] = EyeDataset(
        datapath,
        split['train'],
        config['TRAIN_AUG']
    )

    data['val'] = EyeDataset(
        datapath,
        split['val'],
        config['EVAL_AUG']
    )

def load_data():
    data = { }

    load_data_with_labels(data)

    return data

def torch_long(data, device):
    return Variable(torch.LongTensor(data)).to(device)

def torch_float(data, device):
    return Variable(torch.FloatTensor(data), requires_grad=True).to(device)

def create_model(num_classes):
    model = UnetSm( in_channels=3,
                    out_channels=num_classes,
                    encoder_name=config['BACKBONE'],
                    encoder_weights='imagenet',
                    decoder_attention_type='scse' )

    return model

def inner_supervised(model, imgs_batch, masks_batch):
    imgs_batch = imgs_batch.to(config['DEVICE'])
    masks_batch = masks_batch.to(config['DEVICE'])

    logits_batch = model(imgs_batch)

    loss = dice_with_logits_loss(masks_batch, logits_batch, average='macro', averaged_classes=np.arange(2)) + \
           cross_entropy_with_logits_loss(masks_batch, logits_batch)

    return loss

def inner_train_loop(model, opt, dataset):
    model.train()

    batch_losses = list()

    dataloader = DataLoader( dataset,
                             batch_size=config['BATCH_SIZE'],
                             shuffle=True,
                            #  sampler=RandomSampler(dataset),
                             collate_fn=collate_fn,
                             num_workers=config['NJOBS'],
                             pin_memory=False,
                             prefetch_factor=2 )

    for step_idx, (imgs_batch, masks_batch) in config['VERBOSER'](enumerate(dataloader), total=len(dataloader)):
        loss_sup = inner_supervised(model, imgs_batch, masks_batch)

        loss = loss_sup

        loss.backward()

        if (step_idx + 1) % config['ACCUMULATION_STEP'] == 0:
            opt.step()
            model.zero_grad()

        batch_losses.append(loss.item())

    return np.mean(batch_losses)

def inner_val_loop(model, dataset):
    model.eval()

    metrics = defaultdict(list)
    
    dataloader = DataLoader( dataset,
                             batch_size=1,
                             shuffle=False,
                            #  sampler=RandomSampler(dataset),
                             collate_fn=collate_fn,
                             num_workers=config['NJOBS'],
                             pin_memory=False,
                             prefetch_factor=2 )

    for imgs_batch, masks_batch in config['VERBOSER'](dataloader):
        with torch.no_grad():
            imgs_batch = imgs_batch.to(config['DEVICE'])
            masks_batch = masks_batch.to(config['DEVICE'])

            logits_batch = model(imgs_batch)

            pred_masks_batch = logits_batch.argmax(axis=1)

            masks_batch = masks_batch.cpu().data.numpy().astype(np.uint8)
            pred_masks_batch = pred_masks_batch.cpu().data.numpy().astype(np.uint8)

            for mask, pred_mask in zip(masks_batch, pred_masks_batch):
                for metric, average in [ ('recall_score', 'binary'),
                                         ('precision_score', 'binary'),
                                         ('f1_score', 'binary') ]:
                    score = globals()[metric](mask, pred_mask, average=average)

                    if average == 'none':
                        for idx, scorei in zip(np.unique(mask), score):
                            metrics[metric + '_' + average + '_' + str(idx)].append(scorei)
                    else:
                        metrics[metric + '_' + average].append(score)

    additional_scores = {}

    for key in metrics:
        mean = float(np.mean(metrics[key]))
        additional_scores[f'{key}'] = mean

    score = additional_scores['f1_score_binary']

    return score, additional_scores

def fit(model, data):
    train_losses = list()
    val_scores = list()

    model.to(config['DEVICE'])

    opt = torch.optim.AdamW(model.parameters(), lr=config['LEARNING_RATE'], eps=1e-8)

    epochs_without_going_up = 0
    best_score = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(config['EPOCHS']):
        start_time = time.perf_counter()

        loss = inner_train_loop( model,
                                 opt,
                                 data['train'] )

        config['LOGGER'].info(f'epoch - {epoch+1} loss - {loss:.6f}')
        train_losses.append(loss)

        score, additional = inner_val_loop( model,
                                            data['val'] )

        val_scores.append(score)
        config['LOGGER'].info(f'epoch - {epoch+1} score - {100 * score:.2f}%')

        for key in additional:
            config['LOGGER'].info(f'epoch - {epoch+1} {key} - {100 * additional[key]:.2f}%')

        if best_score < score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_going_up = 0

            store(model)
        else:
            epochs_without_going_up += 1

        if epochs_without_going_up == config['STOP_EPOCHS']:
            break

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        config['LOGGER'].info(f'elapsed time {elapsed_time:.2f} s')
        config['LOGGER'].info(f'epoch without improve {epochs_without_going_up}')

        if config['DEBUG']:
            debugging_info['epoch_loss_trace'].append(round(loss, 3))
            debugging_info['epoch_score_trace'].append(round(100*score, 3))
            debugging_info['epoch_times'].append(round(elapsed_time, 3))

            for key in additional:
                additional[key] = round(100*additional[key], 3)

            debugging_info['epoch_additional_score_trace'].append(additional)

    model.load_state_dict(best_state)

def load(model):
    state = torch.load(config['IMODELNAME'], map_location=config['DEVICE'])

    model.load_state_dict(state)

def store(model):
    state = model.state_dict()
    path = config['MODELNAME']

    torch.save(state, path)

def store_debug():
    if not config['DEBUG']:
        return

    if torch.cuda.is_available():
        debugging_info['max_memory_consumption'] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
    else:
        pass

    with open(config['PREFIX'] + 'debug_name.json', 'w') as f:
        json.dump(debugging_info, f)

@click.command()
@click.option('--aug', '-a', type=str, default='')
@click.option('--image_size', '-is', type=int, default=224)
@click.option('--splitpath', '-sp', type=str, default=None)
@click.option('--learning_rate', '-lr', type=float, default=1e-3)
@click.option('--batch_size', '-bs', type=int, default=16)
@click.option('--epochs', '-e', type=int, default=30, help='The number of epoch per train loop')
@click.option('--accumulation_step', '-as', type=int, default=1, help='The number of iteration to accumulate gradients')
@click.option('--stop_epochs', '-se', type=int, default=5)
@click.option('--modelname', '-mn', type=str, default='model_unet_baseline.pth')
@click.option('--imodelname', '-imn', type=str, default='')
@click.option('--backbone', '-bone', type=str, default='resnext50_32x4d')
@click.option('--njobs', type=int, default=1, help='The number of jobs to run in parallel.')
@click.option('--logger_type', '-lt', type=click.Choice(['stream', 'file'], case_sensitive=False), default='stream')
@click.option('--verbose', is_flag=True, help='Whether progress bars are showed')
@click.option('--debug', is_flag=True, help='Whether debug info is stored')
@click.option('--debugname', '-dn', type=str, default='debug_name.json')
def main(**kwargs):
    init_determenistic()

    init_global_config(**kwargs)
    init_precode()

    config['N_CLASSES'] = 2  ## proper init

    for key in config:
        if key != 'LOGGER':
            config['LOGGER'].info(f'{key} {config[key]}')
            debugging_info[key.lower()] = str(config[key])

    data = load_data()

    config['LOGGER'].info(f'create model')
    model = create_model(num_classes=config['N_CLASSES'])

    if config['IMODELNAME']:
        load(model)

    config['LOGGER'].info(f'fit model')
    fit(model, data)

    config['LOGGER'].info(f'store model')
    store(model)

    store_debug()

if __name__ == '__main__':
    main()
