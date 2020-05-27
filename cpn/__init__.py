#!/usr/bin/env python
# coding: utf-8

# Load CPN models
# By Shashi

import numpy as np
import cv2
from models import cpn as modellib
from models.configs.e2e_CPN_ResNet50_FPN_cfg import Config

def load_model():
    model = modellib.CPN(cfg=Config)
    model.load_weights("weights/cpn_resnet50_cpn_0065.h5", by_name=True)
    return model.keras_model

def image_preprocessing(inputs, config):
    img = inputs.astype(np.float32)
    if config.PIXEL_MEANS_VARS:
        img = img - config.PIXEL_MEANS
        if config.PIXEL_NORM:
            img = img / 255.
    return img
