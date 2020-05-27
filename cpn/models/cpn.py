"""
CPN
The main CPN model implemenetation.

Modified By Longqi-S

Further modified by Shashi for application side of use
"""
import numpy as np
import scipy.misc

import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM

from models.subnet import create_global_net, create_refine_net
import models.resnet_backbone as backbone

class CPN():
    def __init__(self, cfg):
        """
        cfg: A Sub-class of the Config class
        """
        self.cfg = cfg
        self.keras_model = self.build()

    def build(self):
        """Build CPN architecture.
        """

        # Inputs
        input_image = KL.Input(shape=[self.cfg.DATA_SHAPE[0], self.cfg.DATA_SHAPE[1], 3], name="input_image")

        _, C2, C3, C4, C5 = backbone.resnet_graph(input_image, self.cfg.BACKBONE, stage5=True)
        backbone_blocks = [C2, C3, C4, C5]
        global_fms, global_outs = create_global_net(backbone_blocks, self.cfg)
        refine_out = create_refine_net(global_fms, self.cfg)

        model = KM.Model([input_image],
                         [refine_out],
                         name='cpn_model')

        return model

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import saving as topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()
