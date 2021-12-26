from itertools import product
from math import sqrt

import numpy as np


def make_anchors(conv_h, conv_w, scale, input_shape=[550, 550], aspect_ratios=[1, 1 / 2, 2]):
    prior_data = []
    for j, i in product(range(conv_h), range(conv_w)):
        x = (i + 0.5) / conv_w
        y = (j + 0.5) / conv_h

        for ar in aspect_ratios:
            ar = sqrt(ar)
            w = scale * ar / input_shape[1]
            h = scale / ar / input_shape[0]

            prior_data += [x, y, w, h]

    return prior_data

#---------------------------------------------------#
#   用于计算共享特征层的大小
#---------------------------------------------------#
def get_img_output_length(height, width):
    filter_sizes    = [7, 3, 3, 3, 3, 3, 3]
    padding         = [3, 1, 1, 1, 1, 1, 1]
    stride          = [2, 2, 2, 2, 2, 2, 2]
    feature_heights = []
    feature_widths  = []

    for i in range(len(filter_sizes)):
        height  = (height + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        width   = (width + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    return np.array(feature_heights)[-5:], np.array(feature_widths)[-5:]
    
def get_anchors(input_shape = [550, 550], anchors_size = [24, 48, 96, 192, 384]):
    feature_heights, feature_widths = get_img_output_length(input_shape[0], input_shape[1])
    
    all_anchors = []
    for i in range(len(feature_heights)):
        anchors     = make_anchors(feature_heights[i], feature_widths[i], anchors_size[i], input_shape)
        all_anchors += anchors
    
    all_anchors = np.reshape(all_anchors, [-1, 4])
    return all_anchors

if __name__ == "__main__":
    anchors = get_anchors([550, 550])
    print(anchors)
