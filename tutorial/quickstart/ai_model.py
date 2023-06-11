import logging


from PIL import Image
import cv2
import torch
from torch import nn
from torchvision import models, transforms
from time import sleep
import os
import datetime
import base64

logger = logging.getLogger(__name__)

class Singleton(object):
    _lock = threading.Lock()

    def __new__(cls, *args, **kargs):
        with cls._lock:
            if not hasattr(cls, "_instance"):
                print("Singleton: new instance {}".format(cls))
                cls._instance = super(Singleton, cls).__new__(cls)
        print("Singleton: return {}".format(cls))
        return cls._instance

class AI(Singleton):
    _name = 'tutorial.quickstart'
    # tensorflow_model = "models/mk_model_cpu-mobilenet.model"
    my_models = {}

    def __init__(self, input):
        print("MyClass: __init__ {}".format(input))
        print("MyClass: hasattr {}".format(hasattr(self, "_instance")))
        self.input = input

    def _create_model(self, id, file):
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
        if(id not in self.my_models.keys()):
            id = list(self.my_models.keys())[0]
        return self.my_models[id](im)

    def do_transform(self, im):
        return self.transform(im)

