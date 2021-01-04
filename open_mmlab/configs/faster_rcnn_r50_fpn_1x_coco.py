import os
from mmdet import models

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(models.__file__)))

_base_ = [
    os.path.join(base_dir, 'configs', '_base_', 'models', 'faster_rcnn_r50_fpn.py'),
    os.path.join(base_dir, 'configs', '_base_', 'datasets', 'coco_detection.py'),
    os.path.join(base_dir, 'configs', '_base_', 'schedules', 'schedule_1x.py'),
    os.path.join(base_dir, 'configs', '_base_', 'default_runtime.py'), ]