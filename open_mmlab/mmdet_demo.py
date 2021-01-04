#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : mmdet_demo.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/4 下午4:10
# @ Software   : PyCharm
#-------------------------------------------------------

import os
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

from mmdet import models

def main():
    config = './configs/faster_rcnn_r50_fpn_1x_coco.py'
    # download the checkpoint from model zoo and put it in `checkpoints/`
    # url: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    checkpoint = './checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    device = 'cuda:0'
    image = '../Dataset/demo.jpg'
    score_thr = 0.3

    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint=checkpoint, device=device)
    # test a single image
    result = inference_detector(model, image)
    # show the results
    show_result_pyplot(model, image, result, score_thr=score_thr)



if __name__ == "__main__":
    main()