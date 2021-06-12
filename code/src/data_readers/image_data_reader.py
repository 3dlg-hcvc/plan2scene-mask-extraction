
# Code adapted from https://github.com/CSAILVision/semantic-segmentation-pytorch

import torch
import numpy as np
from torchvision import transforms
import logging
from PIL import Image
import os


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


class BaseImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, opt, **kwargs):
        # parse options
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # parse the input list
        self.parse_input_list(dataset_path, **kwargs)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def parse_input_list(self, dataset_path, max_sample=-1, start_idx=-1, end_idx=-1):
        self.dataset_path = dataset_path

        self.image_paths = []

        for r, d, f in os.walk(dataset_path):
            for file in f:
                if file.endswith(".jpg"):
                    self.image_paths.append(os.path.join(r,file))

        if max_sample > 0:
            self.image_paths = self.image_paths[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:  # divide file list
            self.image_paths = self.image_paths[start_idx:end_idx]

        self.num_sample = len(self.image_paths)
        assert self.num_sample > 0
        logging.info('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class ImageDataset(BaseImageDataset):
    def __init__(self, dataset_path, opt, **kwargs):
        super().__init__(dataset_path, opt, **kwargs)

    def __getitem__(self, index):
        # load image
        image_path = self.image_paths[index]
        img = Image.open(image_path).convert('RGB')
        img = imresize(img, (256, 192), interp='bilinear')
        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['info'] = image_path
        return output

    def __len__(self):
        return self.num_sample