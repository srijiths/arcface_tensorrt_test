from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
import face_preprocess


def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  
  #model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  #model.set_params(arg_params, aux_params)

  os.environ['MXNET_USE_TENSORRT'] = '1'
  batch_shape = (1, 3, 112, 112)
  arg_params.update(aux_params)
  all_params = dict([(k, v.as_in_context(mx.gpu(0))) for k, v in arg_params.items()])
  executor = mx.contrib.tensorrt.tensorrt_bind(sym, ctx=mx.gpu(0), all_params=all_params,
                                             data=batch_shape, grad_req='null', force_rebind=True)

  #return model, executor
  return executor

class FaceModel:
  def __init__(self, args):
    self.args = args
    ctx = mx.gpu(args.gpu)
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.model = None
    self.ga_model = None
    if self.args.mode == 'recognize':
        self.face_filter_size = args.face_filter_size.split(',')

    if len(args.model)>0:
      self.model = get_model(ctx, image_size, args.model, 'fc1')

    self.threshold = args.threshold
    self.det_minsize = 50
    self.det_threshold = [0.7,0.8,0.9]
    #self.det_factor = 0.9
    self.image_size = image_size
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    if args.det==0:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
    else:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
    self.detector = detector


  def get_input(self, face_img):
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    if ret is None:
      return None, None
    bboxs, points = ret
    if bboxs.shape[0]==0:
      return None, None

    faces = []
    corpbboxs = []
    for i in range(bboxs.shape[0]):
      bbox = bboxs[i,0:4]
      point = points[i,:].reshape((2,5)).T
      face_crop = face_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
      if self.args.mode == 'recognize':
          if face_crop.shape[0] <= int(self.face_filter_size[0]) or face_crop.shape[1] <= int(self.face_filter_size[1]):
              print('Cropped face less than filter threshold', face_crop.shape[0], self.face_filter_size[0], 
                      self.face_filter_size[1])
              continue

      corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
      corpbboxs.append(corpbbox)
      nimg = face_preprocess.preprocess(face_img, bbox, point, image_size='112,112')
      nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
      aligned = np.transpose(nimg, (2,0,1))
      faces.append(aligned)
    return faces, corpbboxs

  def get_feature(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    print('input blob shape :', input_blob.shape)
    data = mx.nd.array(input_blob)
    #db = mx.io.DataBatch(data=(data,))
    #self.model.forward(db, is_train=False)
    #embedding = self.model.get_outputs()[0].asnumpy()

    y_gen = self.model.forward(is_train=False, data=data)
    #y_gen[0].wait_to_read()
    print('embedding :', y_gen)
    #embedding = embedding[0].wait_to_read()

    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding
