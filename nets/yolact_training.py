import math
from functools import partial

import keras
import tensorflow as tf

# def crop(pred, boxes):
#     # num_pos, 136, 136
#     pred_shape = tf.shape(pred)
#     # [1, ……, 136]
#     w = tf.cast(tf.range(pred_shape[1]), tf.float32)
#     # [[1], ……, [136]]
#     h = tf.expand_dims(tf.cast(tf.range(pred_shape[2]), tf.float32), axis=-1)

#     cols = tf.broadcast_to(w, pred_shape)
#     rows = tf.broadcast_to(h, pred_shape)

#     xmin = tf.broadcast_to(tf.reshape(boxes[:, 0], [-1, 1, 1]), pred_shape)
#     ymin = tf.broadcast_to(tf.reshape(boxes[:, 1], [-1, 1, 1]), pred_shape)
#     xmax = tf.broadcast_to(tf.reshape(boxes[:, 2], [-1, 1, 1]), pred_shape)
#     ymax = tf.broadcast_to(tf.reshape(boxes[:, 3], [-1, 1, 1]), pred_shape)

#     mask_left   = (cols >= tf.cast(xmin, cols.dtype))
#     mask_right  = (cols <= tf.cast(xmax, cols.dtype))
#     mask_bottom = (rows >= tf.cast(ymin, rows.dtype))
#     mask_top    = (rows <= tf.cast(ymax, rows.dtype))

#     crop_mask = tf.math.logical_and(tf.math.logical_and(mask_left, mask_right),
#                                     tf.math.logical_and(mask_bottom, mask_top))
#     crop_mask = tf.cast(crop_mask, tf.float32)

#     return pred * crop_mask

def crop(masks, boxes):
    masks_shape = tf.shape(masks)
    x1, x2 = boxes[:, 0], boxes[:, 2]
    y1, y2 = boxes[:, 1], boxes[:, 3]

    rows = tf.tile(tf.reshape(tf.range(masks_shape[1]),(1, -1, 1)), (masks_shape[0], 1, masks_shape[2]))
    cols = tf.tile(tf.reshape(tf.range(masks_shape[0]),(-1, 1, 1)), (1, masks_shape[1], masks_shape[2]))

    masks_left  = tf.cast(rows, x1.dtype) >= tf.reshape(x1, (1, 1, -1))
    masks_right = tf.cast(rows, x2.dtype) < tf.reshape(x2, (1, 1, -1))
    masks_up    = tf.cast(cols, y1.dtype) >= tf.reshape(y1, (1, 1, -1))
    masks_down  = tf.cast(cols, y2.dtype) < tf.reshape(y2, (1, 1, -1))

    crop_mask = tf.math.logical_and(tf.math.logical_and(masks_left, masks_right),
                                    tf.math.logical_and(masks_down, masks_up))
    crop_mask = tf.cast(crop_mask, tf.float32)

    return masks * crop_mask

def map_to_center_form(x):
    h = x[:, 2] - x[:, 0]
    w = x[:, 3] - x[:, 1]
    cy = x[:, 0] + (h / 2)
    cx = x[:, 1] + (w / 2)
    return tf.stack([cx, cy, w, h], axis=-1)

def loss_location(pred_offset, true_offsets, true_classes):
    #------------------------------------------#
    #   找到哪些先验框是正样本
    #------------------------------------------#
    positive_indices = tf.where(true_classes > 0)

    #------------------------------------------#
    #   取出这些先验框
    #------------------------------------------#
    pred_offset     = tf.gather_nd(pred_offset, positive_indices)
    true_offsets    = tf.gather_nd(true_offsets, positive_indices)

    #------------------------------------------#
    #   计算smooth l1回归损失
    #------------------------------------------#
    regression_diff = tf.abs(pred_offset - true_offsets)
    regression_loss = tf.where(
        tf.less(regression_diff, 1.0),
        0.5 * tf.pow(regression_diff, 2),
        regression_diff - 0.5 
    )

    #------------------------------------------#
    #   对损失进行归一化
    #------------------------------------------#
    normalizer  = tf.maximum(1, tf.shape(positive_indices)[0])
    normalizer  = tf.cast(normalizer, dtype=tf.float32)
    loss        = keras.backend.sum(regression_loss) / normalizer
    return loss

def loss_ohem_conf(pred_classes, true_classes, negpos_ratio = 3):
    #------------------------------------------#
    #   判断哪些框是正样本
    #   positive_bool   batch_size, num_anchors
    #------------------------------------------#
    positive_bool   = true_classes > 0

    #------------------------------------------------------------------------------------#
    #   区分难分类样本
    #   batch_conf      batch_size * num_anchors, num_classes
    #   batch_conf_max  batch_size * num_anchor
    #   mark            batch_size, num_anchors
    #   
    #   mark代表所有负样本的难分类程度。
    #------------------------------------------------------------------------------------#
    batch_conf      = tf.reshape(pred_classes, (-1, tf.shape(pred_classes)[-1]))  
    batch_conf_max  = tf.reduce_max(batch_conf)

    mark = tf.math.log(tf.reduce_sum(tf.math.exp(batch_conf - batch_conf_max), -1)) + batch_conf_max - batch_conf[:, 0]
    mark = tf.reshape(mark, (tf.shape(pred_classes)[0], -1)) 
    mark = tf.where(tf.equal(true_classes, 0), mark, tf.zeros_like(mark))

    #------------------------------------------------------------------------------------#
    #   idx         batch_size, num_anchors
    #   idx_rank    batch_size, num_anchors
    #------------------------------------------------------------------------------------#
    idx      = tf.argsort(mark, direction='DESCENDING', axis=1)
    idx_rank = tf.argsort(idx, axis=1)

    #------------------------------------------------------------------------------------#
    #   positive_bool   batch_size, num_anchors
    #   num_pos         batch_size, 1
    #   num_neg         batch_size, 1
    #------------------------------------------------------------------------------------#
    num_pos = tf.reduce_sum(tf.cast(positive_bool, tf.float32), axis=-1, keep_dims=True)
    num_neg = tf.minimum(negpos_ratio * num_pos, tf.cast(tf.shape(positive_bool)[1], tf.float32) - num_pos)

    #------------------------------------------------------------------------------------#
    #   negative_bool   batch_size, num_anchors
    #------------------------------------------------------------------------------------#
    negative_bool = idx_rank < tf.cast(tf.broadcast_to(num_neg, tf.shape(idx_rank)), tf.int32)
    negative_bool = tf.where(positive_bool, tf.zeros_like(negative_bool), negative_bool)
    negative_bool = tf.where(true_classes < 0, tf.zeros_like(negative_bool), negative_bool)

    #------------------------------------------------------------------------------------#
    #   pos_neg_bool    batch_size, num_anchors
    #------------------------------------------------------------------------------------#
    pos_neg_bool    = tf.logical_or(positive_bool, negative_bool)
    indices         = tf.where(pos_neg_bool)

    classes_pred_selected       = tf.gather_nd(pred_classes, indices)
    classes_gt_selected         = tf.gather_nd(true_classes, indices)
    classes_gt_selected_one_hot = tf.one_hot(tf.cast(classes_gt_selected, tf.int32), tf.shape(pred_classes)[-1])

    #------------------------------------------#
    #   对损失进行归一化
    #------------------------------------------#
    loss_c  = tf.nn.softmax_cross_entropy_with_logits(labels = classes_gt_selected_one_hot, logits = classes_pred_selected)
    num_pos = tf.maximum(1.0, tf.cast(tf.reduce_sum(num_pos), tf.float32))
    loss_c  = tf.reduce_sum(loss_c) / num_pos
    return loss_c
    
def loss_lincomb_mask(pred_mask_coef, pred_proto, mask_gt, anchor_max_box, anchor_max_index, true_classes):
    #------------------------------------------#
    #   判断哪些框是正样本
    #   positive_bool   batch_size, num_anchors
    #------------------------------------------#
    positive_bool = true_classes > 0

    #------------------------------------------#
    #   计算高、宽
    #   136, 136
    #------------------------------------------#
    proto_h = tf.shape(pred_proto)[1]  
    proto_w = tf.shape(pred_proto)[2]

    i         = 0
    n         = tf.shape(pred_mask_coef)[0] 
    total_loss = 0.
    total_pos = 0
    def cond(i, n, total_loss, total_pos):
        return i < n

    def body(i, n, total_loss, total_pos):
        #------------------------------------------#
        #   取出正样本对应的先验框
        #   pos_anchor_index  num_pos, 
        #   pos_anchor_box    num_pos, 4
        #   pos_coef          num_pos, 32
        #------------------------------------------#
        pos_anchor_index    = tf.boolean_mask(anchor_max_index[i], positive_bool[i])
        pos_anchor_box      = tf.boolean_mask(anchor_max_box[i], positive_bool[i])
        pos_coef            = tf.boolean_mask(pred_mask_coef[i], positive_bool[i])
        
        num_pos             = tf.shape(pos_coef)[0]
        total_pos           += num_pos
        #--------------------------------------------------------------------------#
        #   num_objects, h, w                   -> 
        #   num_objects, h, w, 1                -> 
        #   num_objects, proto_h, proto_w, 1    -> 
        #   num_objects, proto_h, proto_w
        #--------------------------------------------------------------------------#
        downsampled_masks = mask_gt[i]
        downsampled_masks = tf.squeeze(tf.image.resize(tf.expand_dims(downsampled_masks, axis=-1), [proto_h, proto_w], method=tf.image.ResizeMethod.BILINEAR), axis=-1)
        downsampled_masks = tf.cast((downsampled_masks > 0.5), pred_mask_coef.dtype)

        #--------------------------------------------------------------------------#
        #   取出每一个先验框对应的mask
        #--------------------------------------------------------------------------#
        pos_mask_gt         = tf.transpose(tf.gather(downsampled_masks, tf.cast(pos_anchor_index, tf.int32)), [1, 2, 0])
        pos_anchor_box      = pos_anchor_box * tf.cast([[proto_w, proto_h, proto_w, proto_h]], pos_anchor_box.dtype)

        #--------------------------------------------------------------------------#
        #   temp_pred_proto     136 * 136, 32
        #   pos_coef            num_pos, 32
        #
        #   136 * 136, 32 @ 32, num_pos -> 136 * 136, num_pos -> num_pos, 136, 136
        #--------------------------------------------------------------------------#
        temp_pred_proto     = tf.reshape(pred_proto[i], [-1, tf.shape(pred_proto[i])[-1]])
        mask_p              = tf.sigmoid(tf.linalg.matmul(temp_pred_proto, tf.transpose(pos_coef, [1, 0])))
        mask_p              = tf.reshape(mask_p, [proto_h, proto_w, -1])
        mask_p              = crop(mask_p, pos_anchor_box)  

        mask_loss           = keras.backend.binary_crossentropy(pos_mask_gt, mask_p)
        #-----------------------------------------------------#
        #   每个先验框各自计算平均值
        #-----------------------------------------------------#
        bbox_center         = map_to_center_form(tf.cast(pos_anchor_box, tf.float32))
        mask_loss           = tf.reduce_sum(mask_loss, axis=[0, 1]) / bbox_center[:, 2] / bbox_center[:, 3]
        total_loss          += tf.reduce_sum(mask_loss)
        i = i + 1
        return i, n, total_loss, total_pos

    i, n, total_loss, total_pos = tf.while_loop(cond, body, [i, n, total_loss, total_pos])
    
    total_pos = tf.maximum(1.0, tf.cast(tf.reduce_sum(total_pos), total_loss.dtype))
    return total_loss / tf.cast(proto_h, tf.float32) / tf.cast(proto_w, tf.float32) / total_pos

def loss_semantic_segmentation(segmentation_p, segment_gt):
    #-----------------------------------------------------#
    #   segmentation_p  n, 69, 69, 80
    #   segment_gt      n, h, w, 80
    #-----------------------------------------------------#
    n, mask_h, mask_w = tf.shape(segmentation_p)[0], tf.cast(tf.shape(segmentation_p)[1], tf.float32), tf.cast(tf.shape(segmentation_p)[2], tf.float32)

    #-----------------------------------------------------#
    #   n, proto_h, proto_w, 80
    #-----------------------------------------------------#
    downsampled_masks = tf.image.resize(segment_gt, [mask_h, mask_w], method=tf.image.ResizeMethod.BILINEAR)
    downsampled_masks = tf.cast((downsampled_masks > 0.5), tf.float32)
    
    loss_s = keras.backend.binary_crossentropy(downsampled_masks, segmentation_p)
    return tf.reduce_sum(loss_s) / mask_h / mask_w / tf.cast(n, tf.float32)

def yolact_Loss(args):
    #-----------------------------------------------------------------#
    #   pred_offset         batch_size, num_anchors, 4
    #   true_offsets        batch_size, num_anchors, 4
    #   pred_classes        batch_size, num_anchors, num_classes
    #   true_classes        batch_size, num_anchors
    #   pred_mask_coef      batch_size, num_anchors, 32
    #   pred_proto          batch_size, 136, 136, 32
    #   mask_gt             batch_size, mask_gt_pad, h, w
    #   seg                 batch_size, 68, 68, num_classes - 1
    #   segment_gt          batch_size, h, w, num_classes - 1
    #   anchor_max_box      batch_size, num_anchors, 4
    #   anchor_max_index    batch_size, num_anchors
    #-----------------------------------------------------------------#
    pred_offset, true_offsets, pred_classes, true_classes, pred_mask_coef, pred_proto, mask_gt, seg, segment_gt, anchor_max_box, anchor_max_index = args
    loc_loss    = loss_location(pred_offset, true_offsets, true_classes) * 1.5
    conf_loss   = loss_ohem_conf(pred_classes, true_classes)
    mask_loss   = loss_lincomb_mask(pred_mask_coef, pred_proto, mask_gt, anchor_max_box, anchor_max_index, true_classes) * 6.125
    seg_loss    = loss_semantic_segmentation(seg, segment_gt)
    total_loss  = loc_loss + conf_loss + mask_loss + seg_loss
    # total_loss = tf.Print(total_loss, [loc_loss,conf_loss,mask_loss,seg_loss], summarize=100)
    return total_loss

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

