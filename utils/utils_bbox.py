import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf


class BBoxUtility(object):
    def __init__(self, nms_thresh=0.45, top_k=200, max_detection=100):
        self._nms_thresh    = nms_thresh
        self._top_k         = top_k
        self.max_detection  = max_detection
        self.boxes          = K.placeholder(dtype='float32', shape=(None, 4))
        self.scores         = K.placeholder(dtype='float32', shape=(None,))
        self.nms            = tf.image.non_max_suppression(self.boxes, self.scores, self._top_k, iou_threshold=self._nms_thresh)
        self.sess           = K.get_session()

    def decode_boxes(self, pred_box, anchors, variances = [0.1, 0.2]):
        #---------------------------------------------------------#
        #   anchors[:, :2] 先验框中心
        #   anchors[:, 2:] 先验框宽高
        #   对先验框的中心和宽高进行调整，获得预测框
        #---------------------------------------------------------#
        boxes = np.concatenate((anchors[:, :2] + pred_box[:, :2] * variances[0] * anchors[:, 2:], 
                                anchors[:, 2:] * np.exp(pred_box[:, 2:] * variances[1])), 1)

        #---------------------------------------------------------#
        #   获得左上角和右下角
        #---------------------------------------------------------#
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def jaccard(self, box_a, box_b, iscrowd: bool = False):
        use_batch = True
        if len(np.shape(box_a)) == 2:
            use_batch = False
            box_a = box_a[None, ...]
            box_b = box_b[None, ...]
            
        A = np.shape(box_a)[1]
        B = np.shape(box_b)[1]
        #--------------------------------------------#
        #   求box_a和box_b的交集
        #   计算交集的左上角和右下角，然后求面积
        #--------------------------------------------#
        max_xy  = np.minimum(np.tile(np.expand_dims(box_a[:, :, 2:], 2), (1, 1, B, 1)), np.tile(np.expand_dims(box_b[:, :, 2:], 1), (1, A, 1, 1)))
        min_xy  = np.maximum(np.tile(np.expand_dims(box_a[:, :, :2], 2), (1, 1, B, 1)), np.tile(np.expand_dims(box_b[:, :, :2], 1), (1, A, 1, 1)))
        inter   = np.maximum((max_xy - min_xy), np.zeros_like((max_xy - min_xy)))
        inter   =  inter[:, :, :, 0] * inter[:, :, :, 1]

        area_a  = np.tile(np.expand_dims(((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1])), 2), [1, 1, B])  # [A,B]
        area_b  = np.tile(np.expand_dims(((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1])), 1), [1, A, 1])  # [A,B]
        union   = area_a + area_b - inter

        out = inter / area_a if iscrowd else inter / union
        return out if use_batch else np.squeeze(out, 0)

    def fast_non_max_suppression(self, box_thre, class_thre, mask_thre, nms_iou, top_k, max_detections):
        #---------------------------------------------------------#
        #   先进行tranpose，方便后面的处理
        #   [80, num_of_kept_boxes]
        #---------------------------------------------------------#
        class_thre      = np.transpose(class_thre, (1, 0))
        #---------------------------------------------------------#
        #   [80, num_of_kept_boxes]
        #   每一行坐标为该种类所有的框的得分，
        #   对每一个种类单独进行排序
        #---------------------------------------------------------#
        idx             = np.argsort(class_thre, 1)[:, ::-1]
        class_thre      = np.sort(class_thre, 1)[:, ::-1]
        
        idx             = idx[:, :top_k]
        class_thre      = class_thre[:, :top_k]

        num_classes, num_dets = np.shape(idx)
        #---------------------------------------------------------#
        #   将num_classes作为第一维度，对每一个类进行非极大抑制
        #   [80, num_of_kept_boxes, 4]    
        #   [80, num_of_kept_boxes, 32]    
        #---------------------------------------------------------#
        box_thre    = np.reshape(box_thre[np.reshape(idx, (-1)), :], (num_classes, num_dets, 4))  
        mask_thre   = np.reshape(mask_thre[np.reshape(idx, (-1)), :], (num_classes, num_dets, -1)) 
        #---------------------------------------------------------#
        #   [80, num_of_kept_boxes, num_of_kept_boxes]
        #   取矩阵的上三角部分
        #---------------------------------------------------------#
        iou         = self.jaccard(box_thre, box_thre)
        iou         = np.triu(iou, k=1)
        iou_max     = np.max(iou, axis=1) 

        #---------------------------------------------------------#
        #   获取和高得分重合程度比较低的预测结果
        #---------------------------------------------------------#
        keep        = (iou_max <= nms_iou)
        class_ids   = np.tile(np.arange(num_classes)[:, None],[1, np.shape(keep)[1]])
        box_nms, class_nms, class_ids, mask_nms = box_thre[keep], class_thre[keep], class_ids[keep], mask_thre[keep]

        idx = np.argsort(class_nms, 0)[::-1][:max_detections]
        box_nms, class_nms, class_ids, mask_nms = box_nms[idx], class_nms[idx], class_ids[idx], mask_nms[idx]
        return box_nms, class_nms, class_ids, mask_nms

    def traditional_non_max_suppression(self, box_thre, class_thre, mask_thre, pred_class_max, max_detections):
        _, num_classes = np.shape(class_thre)
        pred_class_arg = np.argmax(class_thre, axis = -1)
        box_nms, class_nms, class_ids, mask_nms = [], [], [], []
        for c in range(num_classes):
            #--------------------------------#
            #   取出属于该类的所有框的置信度
            #   判断是否大于门限
            #--------------------------------#
            c_confs_m = pred_class_arg == c
            if len(c_confs_m) > 0:
                #-----------------------------------------#
                #   取出得分高于confidence的框
                #-----------------------------------------#
                boxes_to_process = box_thre[c_confs_m]
                confs_to_process = pred_class_max[c_confs_m]
                masks_to_process = mask_thre[c_confs_m]
                #-----------------------------------------#
                #   进行iou的非极大抑制
                #-----------------------------------------#
                idx         = self.sess.run(self.nms, feed_dict={self.boxes: boxes_to_process, self.scores: confs_to_process})
                #-----------------------------------------#
                #   取出在非极大抑制中效果较好的内容
                #-----------------------------------------#
                good_boxes  = boxes_to_process[idx]
                confs       = confs_to_process[idx]
                labels      = c * np.ones((len(idx)), np.int8)
                good_masks  = masks_to_process[idx]
                box_nms.append(good_boxes)
                class_nms.append(confs)
                class_ids.append(labels)
                mask_nms.append(good_masks)
        box_nms, class_nms, class_ids, mask_nms = np.concatenate(box_nms, axis = 0), np.concatenate(class_nms, axis = 0), \
                                                  np.concatenate(class_ids, axis = 0), np.concatenate(mask_nms, axis = 0)

        idx = np.argsort(class_nms, 0)[::-1][:max_detections]
        box_nms, class_nms, class_ids, mask_nms = box_nms[idx], class_nms[idx], class_ids[idx], mask_nms[idx]
        return box_nms, class_nms, class_ids, mask_nms

    def sigmoid(self, x):
        return np.exp(x) / (1 + np.exp(x))

    def yolact_correct_boxes(self, boxes, image_shape):
        image_shape         = np.array(image_shape)[::-1]
        boxes               = boxes * np.concatenate([image_shape, image_shape], axis=-1)
        boxes[:, [0, 1]]    = np.minimum(boxes[:, [0, 1]], boxes[:, [2, 3]])
        boxes[:, [2, 3]]    = np.maximum(boxes[:, [0, 1]], boxes[:, [2, 3]])
        boxes[:, [0, 1]]    = np.maximum(boxes[:, [0, 1]], np.zeros_like(boxes[:, [0, 1]]))
        boxes[:, [2, 3]]    = np.minimum(boxes[:, [2, 3]], np.tile(np.expand_dims(image_shape, 0), [np.shape(boxes)[0], 1]))
        return boxes

    def crop(self, masks, boxes):
        h, w, n = np.shape(masks)
        x1, x2 = boxes[:, 0], boxes[:, 2]
        y1, y2 = boxes[:, 1], boxes[:, 3]

        rows = np.tile(np.reshape(np.arange(w),(1, -1, 1)), (h, 1, n))
        cols = np.tile(np.reshape(np.arange(h),(-1, 1, 1)), (1, w, n))

        masks_left  = rows >= np.reshape(x1, (1, 1, -1))
        masks_right = rows < np.reshape(x2, (1, 1, -1))
        masks_up    = cols >= np.reshape(y1, (1, 1, -1))
        masks_down  = cols < np.reshape(y2, (1, 1, -1))

        crop_mask = masks_left * masks_right * masks_up * masks_down

        return masks * crop_mask

    def decode_nms(self, outputs, anchors, confidence, image_shape, traditional_nms):
        #---------------------------------------------------------#
        #   pred_box    [19248, 4]  对应每个先验框的调整情况
        #   pred_class  [19248, 81] 对应每个先验框的种类      
        #   pred_mask   [19248, 32] 对应每个先验框的语义分割情况
        #   pred_proto  [128, 128, 32]  需要和结合pred_mask使用
        #---------------------------------------------------------#
        pred_box    = outputs[0][0]                 
        pred_class  = outputs[1][0]
        pred_mask   = outputs[2][0]
        pred_proto  = outputs[3][0]

        #---------------------------------------------------------#
        #   将先验框调整获得预测框，
        #   [19248, 4] boxes是左上角、右下角的形式。
        #---------------------------------------------------------#
        boxes       = self.decode_boxes(pred_box, anchors) 

        #---------------------------------------------------------#
        #   除去背景的部分，并获得最大的得分 
        #   [19248, 80]
        #   [19248]
        #---------------------------------------------------------#
        pred_class      = pred_class[:, 1:]  
        pred_class_max  = np.max(pred_class, axis=1)  
        keep            = (pred_class_max > confidence)
        #---------------------------------------------------------#
        #   保留满足得分的框，如果没有框保留，则返回None
        #---------------------------------------------------------#
        box_thre    = boxes[keep, :]
        class_thre  = pred_class[keep, :]
        mask_thre   = pred_mask[keep, :]
        if np.shape(class_thre)[0] == 0:
            return None, None, None, None, None

        if not traditional_nms:
            box_thre, class_thre, class_ids, mask_thre = self.fast_non_max_suppression(box_thre, class_thre, mask_thre, self._nms_thresh, self._top_k, self.max_detection)
            keep        = class_thre > confidence
            box_thre    = box_thre[keep]
            class_thre  = class_thre[keep]
            class_ids   = class_ids[keep]
            mask_thre   = mask_thre[keep]
        else:
            box_thre, class_thre, class_ids, mask_thre = self.traditional_non_max_suppression(box_thre, class_thre, mask_thre, pred_class_max[keep], self.max_detection)
        
        box_thre    = self.yolact_correct_boxes(box_thre, image_shape)

        #---------------------------------------------------------#
        #   pred_proto      [128, 128, 32]
        #   mask_thre       [num_of_kept_boxes, 32]
        #   masks_sigmoid   [128, 128, num_of_kept_boxes]
        #---------------------------------------------------------#
        masks_sigmoid   = self.sigmoid(pred_proto.dot(mask_thre.T))
        #----------------------------------------------------------------------#
        #   masks_sigmoid   [image_shape[0], image_shape[1], num_of_kept_boxes]
        #----------------------------------------------------------------------#
        masks_sigmoid   = cv2.resize(masks_sigmoid, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)
        masks_sigmoid   = np.expand_dims(masks_sigmoid, -1) if len(np.shape(masks_sigmoid)) == 2 else masks_sigmoid
        masks_sigmoid   = self.crop(masks_sigmoid, box_thre)

        #----------------------------------------------------------------------#
        #   masks_arg   [image_shape[0], image_shape[1]]    
        #   获得每个像素点所属的实例
        #----------------------------------------------------------------------#
        masks_arg       = np.argmax(masks_sigmoid, axis=-1)
        #----------------------------------------------------------------------#
        #   masks_arg   [image_shape[0], image_shape[1], num_of_kept_boxes]
        #   判断每个像素点是否满足门限需求
        #----------------------------------------------------------------------#
        masks_sigmoid   = masks_sigmoid > 0.5

        return box_thre, class_thre, class_ids, masks_arg, masks_sigmoid

