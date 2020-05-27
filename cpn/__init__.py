#!/usr/bin/env python
# coding: utf-8

# Load CPN model
# By Shashi

import numpy as np
import cv2
from .models import cpn as modellib
from .models.configs.e2e_CPN_ResNet50_FPN_cfg import Config

def load_model(filepath):
    model = modellib.CPN(cfg=Config)
    model.load_weights(filepath, by_name=True)
    return model.keras_model

def image_preprocessing(inputs):
    img = inputs.astype(np.float32)
    if Config.PIXEL_MEANS_VARS:
        img = img - Config.PIXEL_MEANS
        if Config.PIXEL_NORM:
            img = img / 255.
    return img
