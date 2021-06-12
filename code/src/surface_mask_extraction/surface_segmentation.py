# Code adapted from https://github.com/CSAILVision/semantic-segmentation-pytorch

# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import cv2
import csv
# Our libs
from dataset import TestDataset, BaseDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode, find_recursive, setup_logger
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image, ImageFont
from tqdm import tqdm
from config import cfg
from torchvision import transforms


class SurfaceSegmenter:
    def __init__(self, python_semantic_segmentation_path, surface_encoder_checkpoint, surface_decoder_checkpoint,
                 dataset_cfg,
                 model_arch_encoder="hrnetv2", fc_dim=720, model_arch_decoder="c1", dataset_num_classes=150,
                 gpu = 0):
        self.python_semantic_segmentation_path = python_semantic_segmentation_path
        colors = loadmat(python_semantic_segmentation_path + '/data/color150.mat')['colors']
        colors[27] = np.array([255, 192, 203])
        names = {}
        with open(python_semantic_segmentation_path + '/data/object150_info.csv') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                names[int(row[0])] = row[5].split(";")[0]

        colors_map = {}
        for i in range(colors.shape[0]):
            colors_map[tuple(colors[i])] = names[i + 1]

        weights_encoder = surface_encoder_checkpoint

        # Network Builders
        net_encoder = ModelBuilder.build_encoder(
            arch=model_arch_encoder,
            fc_dim=fc_dim,
            weights=weights_encoder)

        weights_decoder = surface_decoder_checkpoint

        net_decoder = ModelBuilder.build_decoder(
            arch=model_arch_decoder,
            fc_dim=fc_dim,
            num_class=dataset_num_classes,
            weights=weights_decoder,
            use_softmax=True)

        crit = nn.NLLLoss(ignore_index=-1)

        segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
        segmentation_module = segmentation_module.cuda()

        segmentation_module.eval()

        self.dataset_num_classes = dataset_num_classes
        self.names = names
        self.colors = colors
        self.colors_map = colors_map
        self.segmentation_module = segmentation_module
        self.dataset_cfg = dataset_cfg
        self.gpu = gpu

    def visualize_result(self, data, pred):
        (img, info) = data

        # print predictions in descending order
        pred = np.int32(pred)
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)

        for idx in np.argsort(counts)[::-1]:
            name = self.names[uniques[idx] + 1]
            ratio = counts[idx] / pixs * 100

        # colorize prediction
        pred_color = colorEncode(pred, self.colors).astype(np.uint8)

        unique_colours = set([])

        for y in range(pred_color.shape[0]):
            for x in range(pred_color.shape[1]):
                unique_colours.add(tuple(pred_color[y, x]))

        import PIL
        im = PIL.Image.new(mode="RGB", size=(200, pred_color.shape[0]))

        from PIL import ImageDraw
        draw = ImageDraw.Draw(im)

        for i, colour in enumerate(unique_colours):
            draw.rectangle((0, i * 25, 200, i * 25 + 20), fill=(colour))
            draw.text((5, i * 25 + 5), self.colors_map[colour])
        im_numpy = np.asarray(im)

        # aggregate images and save
        im_vis = np.concatenate((img, pred_color, im_numpy), axis=1)
        return Image.fromarray(im_vis)

    def predict_surfaces(self, selected_img):
        segmentation_module = self.segmentation_module
        segmentation_module.eval()
        img_resized_list = selected_img['img_data']
        segSize = (selected_img['img_ori'].shape[0],
                   selected_img['img_ori'].shape[1])

        with torch.no_grad():
            scores = torch.zeros(1, self.dataset_num_classes, segSize[0], segSize[1])
            scores = async_copy_to(scores, self.gpu)

            for img in img_resized_list:
                feed_dict = {}
                feed_dict['img_data'] = img
                feed_dict = async_copy_to(feed_dict, self.gpu)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(self.dataset_cfg["imgSizes"])

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        img = self.visualize_result(
            (selected_img['img_ori'], selected_img['info']),
            pred
        )
        return img, pred