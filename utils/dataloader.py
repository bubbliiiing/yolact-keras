import math
import os.path as osp

import numpy as np
from keras.utils import Sequence
from PIL import Image
from utils.utils import cvtColor, preprocess_input


def encode(matched, anchors, variances = [0.1, 0.2]):
    #--------------------------------------------#
    #   计算中心调整参数
    #--------------------------------------------#
    g_cxcy  = (matched[:, :2] + matched[:, 2:]) / 2 - anchors[:, :2] 
    g_cxcy  /= (variances[0] * anchors[:, 2:]) 

    #--------------------------------------------#
    #   计算宽高调整参数
    #--------------------------------------------#
    g_wh    = (matched[:, 2:] - matched[:, :2]) / anchors[:, 2:] 
    g_wh    = np.log(g_wh) / variances[1]

    offsets = np.concatenate([g_cxcy, g_wh], 1) 
    return offsets

def intersect(box_a, box_b):
    A = np.shape(box_a)[1]
    B = np.shape(box_b)[1]
    #--------------------------------------------#
    #   计算交集的左上角和右下角，然后求面积
    #--------------------------------------------#
    max_xy  = np.minimum(np.tile(np.expand_dims(box_a[:, :, 2:], 2), (1, 1, B, 1)), np.tile(np.expand_dims(box_b[:, :, 2:], 1), (1, A, 1, 1)))
    min_xy  = np.maximum(np.tile(np.expand_dims(box_a[:, :, :2], 2), (1, 1, B, 1)), np.tile(np.expand_dims(box_b[:, :, :2], 1), (1, A, 1, 1)))
    inter   = np.maximum((max_xy - min_xy), np.zeros_like((max_xy - min_xy)))
    return inter[:, :, :, 0] * inter[:, :, :, 1]

def jaccard(box_a, box_b, iscrowd: bool = False):
    use_batch = True
    if len(np.shape(box_a)) == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]
        
    A = np.shape(box_a)[1]
    B = np.shape(box_b)[1]
    #--------------------------------------------#
    #   求box_a和box_b的交集
    #--------------------------------------------#
    inter = intersect(box_a, box_b)

    area_a  = np.tile(np.expand_dims(((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1])), 2), [1, 1, B])  # [A,B]
    area_b  = np.tile(np.expand_dims(((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1])), 1), [1, A, 1])  # [A,B]
    union   = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else np.squeeze(out, 0)


def match(pos_thresh, neg_thresh, box_gt, anchors, class_gt, crowd_boxes):
    #------------------------------#
    #   获得先验框的左上角和右下角
    #------------------------------#
    decoded_anchors = np.concatenate((anchors[:, :2] - anchors[:, 2:] / 2, anchors[:, :2] + anchors[:, 2:] / 2), 1)
    #--------------------------------------------#
    #   overlaps [num_objects, num_anchors]
    #--------------------------------------------#
    overlaps        = jaccard(box_gt, decoded_anchors)

    #--------------------------------------------#
    #   每个真实框重合程度最大的先验框
    #--------------------------------------------#
    each_box_index      = np.argmax(overlaps, 1)
    #--------------------------------------------#
    #   每个先验框重合程度最大的真实框以及得分
    #--------------------------------------------#
    each_anchor_max     = np.max(overlaps, 0) 
    each_anchor_index   = np.argmax(overlaps, 0)
    #--------------------------------------------#
    #   保证每个真实框至少有一个对应的
    #--------------------------------------------#
    each_anchor_max[each_box_index] = 2

    for j in range(np.shape(each_box_index)[0]):
        each_anchor_index[each_box_index[j]] = j

    #--------------------------------------------#
    #   获得每一个先验框对应的真实框的坐标
    #--------------------------------------------#
    each_anchor_box = box_gt[each_anchor_index]
    #--------------------------------------------#
    #   获得每一个先验框对应的种类
    #--------------------------------------------#
    conf            = class_gt[each_anchor_index] + 1
    #--------------------------------------------#
    #   将neg_thresh到pos_thresh之间的进行忽略
    #--------------------------------------------#
    conf[each_anchor_max < pos_thresh] = -1
    conf[each_anchor_max < neg_thresh] = 0

    #--------------------------------------------#
    #   把crowd_boxes部分忽略了
    #--------------------------------------------#
    if crowd_boxes is not None:
        crowd_overlaps      = jaccard(decoded_anchors, crowd_boxes, iscrowd=True)
        best_crowd_overlap  = np.max(crowd_overlaps, 1)
        conf[(conf <= 0) & (best_crowd_overlap > 0.7)] = -1

    offsets = encode(each_anchor_box, anchors)

    return offsets, conf, each_anchor_box, each_anchor_index

class COCODetection(Sequence):
    def __init__(self, image_path, coco, num_classes, anchors, batch_size, augmentation=None):
        self.image_path     = image_path

        self.coco           = coco
        self.ids            = list(self.coco.imgToAnns.keys())

        self.num_classes    = num_classes
        self.anchors        = anchors
        self.batch_size     = batch_size

        self.augmentation   = augmentation

        self.label_map      = {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
            9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
            18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
            27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
            37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
            46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
            54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
            62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
            74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
            82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80
        }
        self.length         = len(self.ids)
        self.pos_thre       = 0.5
        self.neg_thre       = 0.4

    def __getitem__(self, index):
        #------------------------------#
        #   当输入为550, 550时为19248
        #------------------------------#
        num_anchors      = np.shape(self.anchors)[0]

        images      = []
        mask_gts    = []
        segment_gts = []
        offsets_gts = np.zeros([self.batch_size, num_anchors, 4], np.float32)
        conf_gts    = np.zeros([self.batch_size, num_anchors,], np.float32)
        anchor_max_boxes    = np.zeros([self.batch_size, num_anchors, 4], np.float32)
        anchor_max_indexes  = np.zeros([self.batch_size, num_anchors,], np.float32)

        for i, global_index in enumerate(range(index * self.batch_size, (index + 1) * self.batch_size)):  
            global_index = global_index % self.length

            image, boxes, mask_gt, num_crowds = self.pull_item(global_index)
            #------------------------------#
            #   获得框的坐标
            #------------------------------#
            box_gt      = boxes[:, :-1]
            #------------------------------#
            #   获得种类
            #------------------------------#
            class_gt    = boxes[:,  -1]

            if num_crowds > 0:
                mask_gt     = mask_gt[: -num_crowds]
                box_gt      = box_gt[: -num_crowds]
                class_gt    = class_gt[: -num_crowds]
                crowd_boxes = box_gt[-num_crowds: ]
            else:
                crowd_boxes = None

            #------------------------------------------------------------#
            #   offsets_gts         [batch_size, num_anchors, 4]
            #   conf_gts            [batch_size, num_anchors]
            #   anchor_max_boxes    [batch_size, num_anchors, 4]
            #   anchor_max_indexes  [batch_size, num_anchors]
            #------------------------------------------------------------#
            offsets_gts[i], conf_gts[i], anchor_max_boxes[i], anchor_max_indexes[i] = match(self.pos_thre, self.neg_thre,
                                                                                     box_gt, self.anchors, class_gt, crowd_boxes)

            num_gt, height, width   = np.shape(mask_gt)
            segment_gt              = np.zeros((self.num_classes - 1, height, width), np.int8)
            for i in range(num_gt):
                c               = int(class_gt[i])
                segment_gt[c]   = np.maximum(segment_gt[c], mask_gt[i])

            images.append(image)
            mask_gts.append(mask_gt)
            segment_gts.append(np.transpose(segment_gt, [1, 2, 0]))

        num_mask = int(np.max([len(mask_gt) for mask_gt in mask_gts]))
        for i in range(len(mask_gts)):
            mask_gt                     = mask_gts[i]
            mask_gt_pad                 = np.zeros([num_mask, np.shape(mask_gt)[1], np.shape(mask_gt)[2]], np.int8)
            mask_gt_pad[:len(mask_gt)]  = mask_gt
            mask_gts[i]                 = mask_gt_pad
            
        images      = np.array(images, np.float32)
        mask_gts    = np.array(mask_gts, np.int8)
        segment_gts = np.array(segment_gts, np.int8)
        return [images, offsets_gts, conf_gts, mask_gts, segment_gts, anchor_max_boxes, anchor_max_indexes], np.zeros((self.batch_size,))

    def __len__(self):
        return math.ceil(len(self.ids) / float(self.batch_size))

    def pull_item(self, index):
        #------------------------------#
        #   载入coco序号
        #   根据coco序号载入目标信息
        #------------------------------#
        image_id    = self.ids[index]
        target      = self.coco.loadAnns(self.coco.getAnnIds(imgIds = image_id))

        #------------------------------#
        #   根据目标信息判断是否为
        #   iscrowd
        #------------------------------#
        target      = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        crowd       = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
        num_crowds  = len(crowd)
        #------------------------------#
        #   将不是iscrowd的目标
        #       是iscrowd的目标进行堆叠
        #------------------------------#
        target      += crowd

        image_path  = osp.join(self.image_path, self.coco.loadImgs(image_id)[0]['file_name'])
        image       = Image.open(image_path)
        image       = cvtColor(image)
        image       = np.array(image, np.float32)
        height, width, _ = image.shape

        if len(target) > 0:
            masks = np.array([self.coco.annToMask(obj).reshape(-1) for obj in target], np.float32)
            masks = masks.reshape((-1, height, width)) 

            boxes_classes = []
            for obj in target:
                bbox        = obj['bbox']
                final_box   = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], self.label_map[obj['category_id']] - 1]
                boxes_classes.append(final_box)
            boxes_classes = np.array(boxes_classes, np.float32)
            boxes_classes[:, [0, 2]] /= width
            boxes_classes[:, [1, 3]] /= height

        if self.augmentation is not None:
            if len(boxes_classes) > 0:
                image, masks, boxes, labels = self.augmentation(image, masks, boxes_classes[:, :4], {'num_crowds': num_crowds, 'labels': boxes_classes[:, 4]})
                num_crowds  = labels['num_crowds']
                labels      = labels['labels']
                boxes       = np.concatenate([boxes, np.expand_dims(labels, axis=1)], -1)
        image = preprocess_input(image)
        return image, boxes, masks, num_crowds
