import numpy as np
import tensorflow as tf
import os
import pathlib

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name,
    origin=base_url + model_file,
    untar=True)
  model_dir = pathlib.Path(model_dir)/"saved_model"
  model = tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']
  return model

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
model = load_model(model_name)

def crop_human(image):
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]

  output_dict = model(input_tensor)
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy()
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  bbox = output_dict['detection_boxes'][output_dict['detection_classes'] == 1][0]
  x, y = image.shape[:2]

  x1, x2 = max(0, int(x*(bbox[0]))-20), min(x, int(x*(bbox[2]))+20)
  y1, y2 = max(0, int(y*(bbox[1]))-20), min(y, int(y*(bbox[3]))+20)

  h, w = x2-x1, y2-y1

  if h/w > 1.34:
    dy = int((h/1.34 - w)/2)
    y1, y2 = max(0, y1-dy), min(y, y2+dy)
  else:
    dx = int((1.34*w - h)/2)
    x1, x2 = max(0, x1-dx), min(x, x2+dx)

  crop_img = image[x1:x2, y1:y2, :]
  crop_bbox = (x1, y1, x2, y2)

  return crop_img, crop_bbox
