from django.shortcuts import render

from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from rest_framework import permissions
from rest_framework.response import Response
from tutorial.quickstart.serializers import UserSerializer, GroupSerializer, UploadSerializer, PeopleDetectSerializer, FaceDetectSerializer, YoloV3Serializer
from django.conf import settings

from PIL import Image
import cv2
import torch
from torch import nn
from torchvision import models, transforms
from time import sleep
import os
import datetime
import base64
import numpy as np

import requests

import logging
from django.apps import apps as my_apps

logger = logging.getLogger(__name__)


class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = [permissions.IsAuthenticated]

class UploadViewSet(viewsets.ViewSet):
    serializer_class = UploadSerializer

    def list(self, request):
        return Response("GET API")

    def create(self, request):
        logger.info("start create for upload")
        now = datetime.datetime.now()
        file_uploaded = request.FILES.get('file_uploaded')
        logger.info("file uploaded:{}".format(file_uploaded))
        content_type = file_uploaded.content_type
        logger.info("content type:{}".format(content_type))
        id = file_uploaded.name
        if("." in id):
            id = id[0:id.rindex(".")]
        filename = file_uploaded.name
        logger.info("filename:{}".format(filename))
        if("." in filename):
            nm = filename[0:filename.rindex(".")]
            sf = filename[filename.rindex("."):]
            filename = "{}_{:%Y%m%d_%H%M%S}{}".format(nm, now, sf)
        else:
            filename = "{}_{:%Y%m%d_%H%M%S}".format(filename, now)
        logger.info("filename:{}".format(filename))
        path = os.path.join("data", filename)
        buf = bytearray()
        # fp = open(path, 'wb')
        for chunk in file_uploaded.chunks():
            # fp.write(chunk)
            buf += bytearray(chunk)
        # fp.close()
        logger.info("buf.len:{}".format(len(buf)))

        # Tray_Watcher : 1yZ9BSn4LD0ihlH4NyvLi8qbCY51xKiTV05M242GUBA
        # reffer to https://stackoverflow.com/questions/17170752/python-opencv-load-image-from-byte-string
        jpg_as_np = np.frombuffer(bytes(buf), dtype=np.uint8)
        img_org = cv2.imdecode(jpg_as_np, flags=1)
        # img = cv2.imread(path)
        img = cv2.resize(img_org, (224,224), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        image = my_apps.get_app_config('quickstart').do_transform(im)
        logger.info("p3.1:{}".format(str(image)))
        image = image.clone().detach()
        logger.info("p3.2:{}".format(str(image)))
        image = image.reshape((1,3,224,224))
        output = my_apps.get_app_config('quickstart').do_model(id, image)
        logger.info("p3.output:{}".format(str(output)))
        output = output.detach().cpu().numpy()
        pred = output[0][1] - output[0][0]
        logger.info("pred:{}".format(pred))

        r = ""
        if(pred > 3.0):
            # LINE
            my_first_line_test_token = settings.MY_APP['line_token']
            url = settings.MY_APP['line_url']

            cv2.imwrite(path, img)
            logger.info("wirte image:{}".format(path))

            access_token = my_first_line_test_token
            headers = {'Authorization': 'Bearer ' + access_token}

            message = "Please replenish lacking red pepper!\n唐辛子を補充してください。\n({})".format(pred)
            payload = {'message': message}

            files = {'imageFile': open(path, 'rb')}
            r = requests.post(url,headers=headers, params=payload, files=files,)
        else:
            path = os.path.join("data0", filename)
            cv2.imwrite(path, img)
            logger.info("wirte image:{}".format(path))
 
        response = "POST API and you have uploaded a {} file {} {} {}".format(content_type, filename, pred, r)

        logger.info("response: {}".format(response))

        return Response(response)

class PeopleDetectViewSet(viewsets.ViewSet):
    serializer_class = PeopleDetectSerializer

    def list(self, request):
        return Response("GET API")

    def inside(self, r, q):
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q
        return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


    def draw_detections(self, img, rects, thickness = 1):
        for x, y, w, h in rects:
            # the HOG detector returns slightly larger rectangles than the real objects.
            # so we slightly shrink the rectangles to get a nicer output.
            pad_w, pad_h = int(0.15*w), int(0.05*h)
            cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

    def create(self, request):
        logger.info("start create for people detect")
        now = datetime.datetime.now()
        file_uploaded = request.FILES.get('file_uploaded')
        content_type = file_uploaded.content_type
        id = file_uploaded.name
        filename = file_uploaded.name
        if("." in filename):
            nm = filename[0:filename.rindex(".")]
            sf = filename[filename.rindex("."):]
            filename = "{}_{:%Y%m%d_%H%M%S}{}".format(nm, now, sf)
        else:
            filename = "{}_{:%Y%m%d_%H%M%S}".format(filename, now)
        path = os.path.join("data", filename)
        buf = bytearray()
        for chunk in file_uploaded.chunks():
            buf += bytearray(chunk)

        jpg_as_np = np.frombuffer(bytes(buf), dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=1)

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
        
        found, prob = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
        if(len(found)):
            found_filtered = []
            for ri, r in enumerate(found):
                for qi, q in enumerate(found):
                    if ri != qi and inside(r, q):
                        break
                else:
                    found_filtered.append(r)
            self.draw_detections(img, found)
            self.draw_detections(img, found_filtered, 3)

            # LINE
            my_first_line_test_token = settings.MY_APP['line_token']
            url = settings.MY_APP['line_url']

            cv2.imwrite(path, img)

            access_token = my_first_line_test_token
            headers = {'Authorization': 'Bearer ' + access_token}

            message = "Found people!"
            payload = {'message': message}

            files = {'imageFile': open(path, 'rb')}
            r = requests.post(url,headers=headers, params=payload, files=files,)
 
            response = "POST API and you have uploaded a {} file {} {}".format(content_type, filename, r)
        else:
            response = "POST API and no people detected"
        logger.info("response: {}".format(response))

        return Response(response)

class FaceDetectViewSet(viewsets.ViewSet):
    serializer_class = FaceDetectSerializer

    def list(self, request):
        return Response("GET API")

    def detect(self, img, cascade):
        rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:,2:] += rects[:,:2]
        return rects

    def draw_rects(self, img, rects, color):
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    def create(self, request):
        logger.info("start create face detect")
        now = datetime.datetime.now()
        file_uploaded = request.FILES.get('file_uploaded')
        content_type = file_uploaded.content_type
        id = file_uploaded.name
        filename = file_uploaded.name
        if("." in filename):
            nm = filename[0:filename.rindex(".")]
            sf = filename[filename.rindex("."):]
            filename = "{}_{:%Y%m%d_%H%M%S}{}".format(nm, now, sf)
        else:
            filename = "{}_{:%Y%m%d_%H%M%S}".format(filename, now)
        logger.info("filename: {}".format(filename))
        path = os.path.join("data", filename)
        buf = bytearray()
        for chunk in file_uploaded.chunks():
            buf += bytearray(chunk)

        cascade_fn = "model_data/haarcascades/haarcascade_frontalface_alt.xml"
        nested_fn = "model_data/haarcascades/haarcascade_eye.xml"

        cascade = cv2.CascadeClassifier(cascade_fn)
        nested = cv2.CascadeClassifier(nested_fn)

        jpg_as_np = np.frombuffer(bytes(buf), dtype=np.uint8)
        img_org = cv2.imdecode(jpg_as_np, flags=1)
        img_small = cv2.resize(img_org, (224,224), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        # cv2.imwrite("a.jpg", gray)

        rects = self.detect(gray, cascade)
        logger.info("len(rects): {}".format(len(rects)))
        if len(rects) > 0:
            vis = img_org.copy()
            self.draw_rects(vis, rects, (0, 255, 0))
            if not nested.empty():
                for x1, y1, x2, y2 in rects:
                    roi = gray[y1:y2, x1:x2]
                    vis_roi = vis[y1:y2, x1:x2]
                    subrects = self.detect(roi.copy(), nested)
                    self.draw_rects(vis_roi, subrects, (255, 0, 0))

            url = settings.MY_APP['line_url']

            #cv2.imwrite(path, img_small)
            cv2.imwrite(path, vis)

            my_first_line_test_token = settings.MY_APP['line_token']
            access_token = my_first_line_test_token
            headers = {'Authorization': 'Bearer ' + access_token}

            message = "Find face!\n"
            payload = {'message': message}

            files = {'imageFile': open(path, 'rb')}
            r = requests.post(url,headers=headers, params=payload, files=files,)
        else:
            path = os.path.join("data0", filename)
            cv2.imwrite(path, img_org)
            r = "no face"
 
        response = "POST API and you have uploaded a {} file {} {}".format(content_type, filename, r)

        logger.info("response: {}".format(response))

        return Response(response)


class YoloV3ViewSet(viewsets.ViewSet):
    serializer_class = YoloV3Serializer

    def list(self, request):
        return Response("GET API")

    def create(self, request):
        logger.info("start create Yolo V3 detect")
        now = datetime.datetime.now()
        file_uploaded = request.FILES.get('file_uploaded')
        content_type = file_uploaded.content_type
        id = file_uploaded.name
        filename = file_uploaded.name
        if("." in filename):
            nm = filename[0:filename.rindex(".")]
            sf = filename[filename.rindex("."):]
            filename = "{}_{:%Y%m%d_%H%M%S}{}".format(nm, now, sf)
        else:
            filename = "{}_{:%Y%m%d_%H%M%S}".format(filename, now)
        logger.info("filename: {}".format(filename))
        path = os.path.join("data", filename)
        buf = bytearray()
        for chunk in file_uploaded.chunks():
            buf += bytearray(chunk)

        conf = 0.5

        jpg_as_np = np.frombuffer(bytes(buf), dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=1)

        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        r = blob[0, 0, :, :]

        # set blob
        outputs = my_apps.get_app_config('quickstart').set_yolov3_input(blob)
        t = datetime.datetime.now()
        logger.info(t)
        outputs = my_apps.get_app_config('quickstart').do_yolov3_forward()
        t = datetime.datetime.now()
        logger.info(t)

        boxes = []
        confidences = []
        classIDs = []
        H, W = img.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if(confidence > conf):
                    x, y, w, h = detection[:4] * np.array([W, H, W, H])
                    p0 = int(x - w//2), int(y - h//2)
                    p1 = int(x + w//2), int(y + h//2)
                    boxes.append([*p0, int(w), int(h)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
        num_objs = len(indices)
        if num_objs > 0:
            if(num_objs > 1):
                message = "Find {} objects!\n".format(num_objs)
            else:
                message = "Find one object!\n"

            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                clr = my_apps.get_app_config('quickstart').get_coco_color(classIDs[i])
                color = [int(c) for c in clr]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cls = my_apps.get_app_config('quickstart').get_coco_class(classIDs[i])
                text = "{}: {:.4f}".format(cls, confidences[i])
                cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                message += (text + "\n")

            url = settings.MY_APP['line_url']

            # make image small to redule network trafic
            img_small = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(path, img_small)

            my_first_line_test_token = settings.MY_APP['line_token']
            access_token = my_first_line_test_token
            headers = {'Authorization': 'Bearer ' + access_token}

            payload = {'message': message}

            files = {'imageFile': open(path, 'rb')}
            r = requests.post(url,headers=headers, params=payload, files=files,)
        else:
            path = os.path.join("data0", filename)
            cv2.imwrite(path, img)
 
        response = "POST API and you have uploaded a {} file {} {}".format(content_type, filename, r)

        logger.info("response: {}".format(response))

        return Response(response)

