#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from ctypes import *

from Method.dlls import initDLL

class DarknetDetector(object):
    def __init__(self):
        self.load_meta = None
        self.load_net = None
        self.load_image = None
        self.predict_image = None
        self.get_network_boxes = None
        self.do_nms_obj = None
        self.free_image = None
        self.free_detections = None

        self.meta = None
        self.model = None
        return

    def loadDLL(self, dll_folder_path):
        if dll_folder_path[-1] != "/":
            dll_folder_path += "/"

        if not os.path.exists(dll_folder_path):
            print("[ERROR][DarknetDetector::loadDLL]")
            print("\t dll folder not exist!")
            return False

        if not os.path.exists(dll_folder_path + "libdarknet.so"):
            print("[ERROR][DarknetDetector::loadDLL]")
            print("\t dll folder not exist!")
            return False

        self.load_meta, self.load_net, self.load_image, \
            self.predict_image, self.get_network_boxes, self.do_nms_obj, \
            self.free_image, self.free_detections = initDLL(dll_folder_path)
        return True

    def loadModel(self, meta_path, config_path, model_path):
        if not os.path.exists(meta_path):
            print("[ERROR][DarknetDetector::loadModel]")
            print("\t meta file not exist!")
            return False
        if not os.path.exists(config_path):
            print("[ERROR][DarknetDetector::loadModel]")
            print("\t config file not exist!")
            return False
        if not os.path.exists(model_path):
            print("[ERROR][DarknetDetector::loadModel]")
            print("\t model file not exist!")
            return False

        meta_path_bytes = bytes(meta_path, encoding='utf-8')
        config_path_bytes = bytes(config_path, encoding='utf-8')
        model_path_bytes = bytes(model_path, encoding='utf-8')
        self.meta = self.load_meta(meta_path_bytes)
        self.model = self.load_net(config_path_bytes, model_path_bytes, 0)
        return True

    def detect(self, image, thresh=.5, hier_thresh=.5, nms=.45):
        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(self.model, image)
        dets = self.get_network_boxes(self.model, image.w, image.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (nms): self.do_nms_obj(dets, num, self.meta.classes, nms);

        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((self.meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(image)
        self.free_detections(dets, num)
        return res

    def detectFile(self, image_path, thresh=.5, hier_thresh=.5, nms=.45):
        image_path_bytes = bytes(image_path, encoding='utf-8')
        image = self.load_image(image_path_bytes, 0, 0)
        return self.detect(image, thresh, hier_thresh, nms)

def demo():
    dll_folder_path = "/home/chli/github/darknet-copy/"
    meta_path = "/home/chli/github/darknet-copy/Config/railway/voc.data"
    config_path = "./Config/railway/yolov3.cfg"
    model_path = "./Config/railway/yolov3_train_2c_detect_2class.backup"
    image_path = "/home/chli/chLi/Download/DeepLearning/Dataset/RailwayStation/2C_mask/train_dataset/0.jpg"

    darknet_detector = DarknetDetector()
    darknet_detector.loadDLL(dll_folder_path)
    darknet_detector.loadModel(meta_path, config_path, model_path)
    result = darknet_detector.detectFile(image_path)

    print(result)
    return True

if __name__ == "__main__":
    demo()

