#!/usr/bin/env python
# coding: utf-8

# 3D Pose Estimation from Multi Camera
# By Shashi

import os
import os.path as osp
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import plotly.graph_objects as go
from cpn import load_model

cpn_weights = "weights/cpn_resnet50_cpn_0065.h5"
model = load_model(cpn_weights)

idx2kpt = ['nose', 'r_eye', 'l_eye', 'r_ear', 'l_ear', 'r_shoulder', 'l_shoulder',
                'r_elbow', 'l_elbow', 'r_hand', 'l_hand', 'r_hip', 'l_hip',
                'r_knee', 'l_knee', 'r_foot', 'l_foot']

kpt2idx = {}
for i in range(17):
    kpt2idx[idx2kpt[i]] = i

class Pose3DCPN:
    def __init__(self, R1, T1, R2, T2):
        self.R1 = R1
        self.T1 = T1
        self.R2 = R2
        self.T2 = T2

    def gen_heatmaps(self, img, oriImg, plot_fig=False):
        input_img = oriImg[np.newaxis,...]
        output_blobs = model.predict(input_img)
        heatmap = output_blobs[0]

        if plot_fig:
            figure = plt.figure(figsize=(15, 15))
            for i in range(17):
                plt.subplot(4, 5, i+1, title='heatmap: ' + idx2kpt[i])
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(img[:,:,[2,1,0]])
                h = cv2.resize(heatmap[:,:,i], (0,0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                plt.imshow(h, alpha=.5)

            plt.subplots_adjust(bottom=0, right=0.95, top=0.6)

        return heatmap

    def extract_kpt_2d(self, heatmap, bbox, crop_img):
        kpt = {}
        for i in range(17):
            h = cv2.resize(heatmap[:,:,i], (crop_img.shape[1], crop_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            x, y = np.unravel_index(np.argmax(h), h.shape)
            kpt[idx2kpt[i]] = [bbox[0] + x, bbox[1] + y]

        x = kpt['r_shoulder'][0] + kpt['l_shoulder'][0] + kpt['nose'][0]
        y = kpt['r_shoulder'][1] + kpt['l_shoulder'][1] + kpt['nose'][1]
        x, y = int(x/3), int(y/3)

        kpt['neck'] = [x, y]

        return kpt

    def draw_skl(self, img, kpt):
        face = ['l_ear', 'l_eye', 'nose', 'neck', 'nose', 'r_eye', 'r_ear']
        arms = ['l_hand', 'l_elbow', 'l_shoulder', 'neck', 'r_shoulder', 'r_elbow', 'r_hand']
        belly = ['l_shoulder', 'r_hip', 'l_hip', 'r_shoulder']
        legs = ['l_foot', 'l_knee', 'l_hip', 'r_hip', 'r_knee', 'r_foot']

        body_parts = [face, arms, belly, legs]

        skl_img = np.zeros(img.shape, dtype = "uint8")

        for part in body_parts:
            for i in range(len(part)-1):
                cv2.line(skl_img, (kpt[part[i]][1], kpt[part[i]][0]), (kpt[part[i+1]][1], kpt[part[i+1]][0]), (0, 255, 0), 3)
                cv2.circle(skl_img,(kpt[part[i]][1], kpt[part[i]][0]), 6, (0,0,255), -1)

            cv2.circle(skl_img,(kpt[part[-1]][1], kpt[part[-1]][0]), 6, (0,0,255), -1)

        skl_img[skl_img==0] = img[skl_img==0]
        return skl_img

    def calc_3D_kpt(self, kpt_1, kpt_2):
        R1_inv = np.linalg.inv(self.R1)
        R2_inv = np.linalg.inv(self.R2)

        keys = list(kpt_1.keys())

        kpt_3D = {}
        kpt_3D_list = []

        for key in keys:
            C1 = copy.deepcopy(kpt_1[key])
            C2 = copy.deepcopy(kpt_2[key])
            y1, x1 = C1[0], C1[1]
            y2, x2 = C2[0], C2[1]

            y1_d, x1_d = y1-self.T1[1], x1-self.T1[0]
            y2_d, x2_d = y2-self.T2[1], x2-self.T2[0]

            R_d = np.array([R1_inv[:,0], R1_inv[:,1], R2_inv[:,0], R2_inv[:,1]]).T
            Y = np.dot(R_d, np.array([[-1*x1_d[0], -1*y1_d[0], x2_d[0], y2_d[0]]]).T)
            X = np.array([R1_inv[:,2], R2_inv[:,2]]).T
            W = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
            z1_d = W[0, 0]

            kpt_W = np.dot(R1_inv, np.array([[x1_d, y1_d, z1_d]]).T)
            kpt_3D[key] = [kpt_W[0, 0][0], kpt_W[1, 0][0], kpt_W[2, 0][0]]

        return kpt_3D

    def get_3D_fig(self, fig, kpt_3D):
        face = ['nose', 'neck']
        arms = ['l_hand', 'l_elbow', 'l_shoulder', 'neck', 'r_shoulder', 'r_elbow', 'r_hand']
        belly = ['l_shoulder', 'r_hip', 'l_hip', 'r_shoulder']
        legs = ['l_foot', 'l_knee', 'l_hip', 'r_hip', 'r_knee', 'r_foot']

        body_parts = [face, arms, belly, legs]

        for part in body_parts:
            face_pt = []
            for i in range(len(part)):
                face_pt.append(kpt_3D[part[i]])

            face_pt = np.array(face_pt)
            x, y, z = face_pt[:, 0], face_pt[:, 1], face_pt[:, 2]
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                       mode='markers+lines',
                                       marker=dict(size=3))
                         )

        return fig
