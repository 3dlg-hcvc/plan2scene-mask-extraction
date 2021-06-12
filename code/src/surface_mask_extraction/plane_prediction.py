# Code adapted from https://github.com/svip-lab/PlanarReconstruction/blob/master/predict.py

import os
import cv2
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as tf

from models.baseline_same import Baseline as UNet
from utils.disp import tensor_to_image
from utils.disp import colors_256 as colors
from bin_mean_shift import Bin_Mean_Shift
from modules import get_coordinate_map
from utils.loss import Q_loss
from instance_parameter_loss import InstanceParameterLoss

class PlanePredictor:
    def __init__(self, cfg):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # build network
        network = UNet(cfg.model)

        model_dict = torch.load(cfg.resume_dir, map_location=lambda storage, loc: storage)
        network.load_state_dict(model_dict)

        # load nets into gpu
        if cfg.num_gpus > 1 and torch.cuda.is_available():
            network = torch.nn.DataParallel(network)
        network.to(device)
        network.eval()

        transforms = tf.Compose([
            tf.ToTensor(),
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        bin_mean_shift = Bin_Mean_Shift(device=device)
        k_inv_dot_xy1 = get_coordinate_map(device)
        instance_parameter_loss = InstanceParameterLoss(k_inv_dot_xy1)

        self.device = device
        self.network = network
        self.transforms = transforms
        self.bin_mean_shift = bin_mean_shift
        self.k_inv_dot_xy1 = k_inv_dot_xy1
        self.instance_parameter_loss = instance_parameter_loss

    def predict_planes(self, image_path):
        cfg = self.cfg
        device = self.device
        network = self.network
        transforms = self.transforms
        bin_mean_shift = self.bin_mean_shift
        k_inv_dot_xy1 = self.k_inv_dot_xy1
        instance_parameter_loss = self.instance_parameter_loss

        h, w = 192, 256

        with torch.no_grad():
            image = cv2.imread(image_path)
            # the network is trained with 192*256 and the intrinsic parameter is set as ScanNet
            image = cv2.resize(image, (w, h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = transforms(image)
            image = image.to(device).unsqueeze(0)
            # forward pass
            logit, embedding, _, _, param = network(image)

            prob = torch.sigmoid(logit[0])

            # infer per pixel depth using per pixel plane parameter, currently Q_loss need a dummy gt_depth as input
            _, _, per_pixel_depth = Q_loss(param, k_inv_dot_xy1, torch.ones_like(logit))

            # fast mean shift
            segmentation, sampled_segmentation, sample_param = bin_mean_shift.test_forward(
                prob, embedding[0], param, mask_threshold=0.1)

            # since GT plane segmentation is somewhat noise, the boundary of plane in GT is not well aligned,
            # we thus use avg_pool_2d to smooth the segmentation results
            b = segmentation.t().view(1, -1, h, w)
            pooling_b = torch.nn.functional.avg_pool2d(b, (7, 7), stride=1, padding=(3, 3))
            b = pooling_b.view(-1, h * w).t()
            segmentation = b

            # infer instance depth
            instance_loss, instance_depth, instance_abs_disntace, instance_parameter = instance_parameter_loss(
                segmentation, sampled_segmentation, sample_param, torch.ones_like(logit), torch.ones_like(logit), False)

            # return cluster results
            predict_segmentation = segmentation.cpu().numpy().argmax(axis=1)

            # mask out non planar region
            predict_segmentation[prob.cpu().numpy().reshape(-1) <= 0.1] = 20
            predict_segmentation = predict_segmentation.reshape(h, w)

            # visualization and evaluation
            image = tensor_to_image(image.cpu()[0])
            mask = (prob > 0.1).float().cpu().numpy().reshape(h, w)
            depth = instance_depth.cpu().numpy()[0, 0].reshape(h, w)
            per_pixel_depth = per_pixel_depth.cpu().numpy()[0, 0].reshape(h, w)

            # use per pixel depth for non planar region
            depth = depth * (predict_segmentation != 20) + per_pixel_depth * (predict_segmentation == 20)

            return depth, image, instance_parameter.cpu(), predict_segmentation, param.cpu(), mask