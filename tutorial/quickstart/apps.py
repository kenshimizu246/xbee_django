from django.apps import AppConfig
from django.conf import settings

import logging


from PIL import Image
import cv2
import torch
import numpy as np
import time
from torch import nn
from torchvision import models, transforms
from time import sleep
import os
import datetime
import base64

logger = logging.getLogger(__name__)

class QuickstartConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'tutorial.quickstart'
    # tensorflow_model = "models/mk_model_cpu-mobilenet.model"
    my_models = {}

    def ready(self):
        logger.info("ready()")
        logger.info("settings:{}".format(settings.MY_APP['line_token']))

#        model = models.mobilenet_v2(pretrained=True)

#        model.classifier = nn.Sequential(
#                nn.Dropout(0.2),
#                nn.Linear(model.last_channel, 2),
#            )

#        model.load_state_dict(torch.load(settings.MY_APP['model_file']))

#        model.cpu()

#        model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            ])
#        self.model = model

        trays = settings.MY_APP['trays']
        for tray in trays:
            logger.info("ready(): creating model {}:{}:{}".format(tray['name'], tray['id'], tray['model']))
            self._create_model(tray['id'], tray['model'])

        self._create_yolov3()

    def _create_model(self, id, file):
        logger.info("_create_model():{}:{}".format(id, file))
        model = models.mobilenet_v2(pretrained=True)

        model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(model.last_channel, 2),
            )

        model.load_state_dict(torch.load(file))

        model.cpu()

        model.eval()

        self.my_models[id] = model

    def do_model(self, id, im):
        logger.info("do_model():{}:{}".format(id, str(im)))
        logger.info("do_model().keys:{}:{}".format(id, self.my_models.keys()))
        if(id not in self.my_models.keys()):
            id = list(self.my_models.keys())[0]
            logger.info("do_model.id:{}".format(id))
        logger.info("do_model.my_models:{}".format(str(self.my_models[id])))
        logger.info("do_model.my_models2:{}".format(type(self.my_models[id])))
        ret = self.my_models[id](im)
        logger.info("do_model.my_models.ret:{}".format(str(ret)))
        return ret

    def do_transform(self, im):
        return self.transform(im)

    def _create_yolov3(self):
        logger.info("start _create_yolov3()")
        coco_names_fn = settings.MY_APP['coco_names']
        config_fn = settings.MY_APP['yolov3_config']
        weights_fn = settings.MY_APP['yolov3_weights']

        # Load coco.names and allocate color respectively.
        classes = open(coco_names_fn).read().strip().split('\n')
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
        self.coco_classes = classes
        self.coco_colors = colors

        # Give the configuration and weight files for the model and load the network.
        # net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        net = cv2.dnn.readNetFromDarknet(config_fn, weights_fn)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolov3_net = net

        ln = net.getLayerNames()
        ll = [x[0] for x in net.getUnconnectedOutLayers()]
        ln = [ln[i - 1] for i in ll]
        self.yolov3_ln = ln
        logger.info("end _create_yolov3()")

    def set_yolov3_input(self, blob):
        self.yolov3_net.setInput(blob)

    def do_yolov3_forward(self):
        return self.yolov3_net.forward(self.yolov3_ln)

    def get_coco_class(self, classID):
        return self.coco_classes[classID];

    def get_coco_color(self, classID):
        return self.coco_colors[classID];

