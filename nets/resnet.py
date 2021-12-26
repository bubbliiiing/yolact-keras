#-------------------------------------------------------------#
#   ResNet50的网络部分
#-------------------------------------------------------------#
from keras import layers
from keras.layers import (Activation, BatchNormalization, Conv2D, Input,
                          MaxPooling2D, ZeroPadding2D)


def identity_block(input_tensor, kernel_size, filters, name=""):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, (1, 1), name=name+'.conv1', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=1e-5, name=name+'.bn1')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(filters2, kernel_size, padding='valid', name=name+'.conv2', use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-5, name=name+'.bn2')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=name+'.conv3', use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-5, name=name+'.bn3')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, strides=(2, 2), name=""):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, (1, 1), name=name+'.conv1', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=1e-5, name=name+'.bn1')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(filters2, kernel_size, strides=strides, padding='valid', name=name+'.conv2', use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-5, name=name+'.bn2')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=name+'.conv3', use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-5, name=name+'.bn3')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=name+'.downsample.0', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=1e-5, name=name+'.downsample.1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(inputs):
    # 544, 544, 3 -> 272, 272, 64
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='backbone.conv1',use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-5, name='backbone.bn1')(x)
    x = Activation('relu')(x)

    # 272, 272, 64 -> 136, 136, 64
    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(x)

    # 136, 136, 64 -> 136, 136, 256
    x = conv_block(x, 3, [64, 64, 256], strides=(1, 1), name="backbone.layers.0.0")
    x = identity_block(x, 3, [64, 64, 256], name="backbone.layers.0.1")
    x = identity_block(x, 3, [64, 64, 256], name="backbone.layers.0.2")

    # 136, 136, 256 -> 68, 68, 512
    x = conv_block(x, 3, [128, 128, 512], name="backbone.layers.1.0")
    x = identity_block(x, 3, [128, 128, 512], name="backbone.layers.1.1")
    x = identity_block(x, 3, [128, 128, 512], name="backbone.layers.1.2")
    x = identity_block(x, 3, [128, 128, 512], name="backbone.layers.1.3")
    y1 = x

    # 68, 68, 512 -> 34, 34, 1024
    x = conv_block(x, 3, [256, 256, 1024], name="backbone.layers.2.0")
    x = identity_block(x, 3, [256, 256, 1024], name="backbone.layers.2.1")
    x = identity_block(x, 3, [256, 256, 1024], name="backbone.layers.2.2")
    x = identity_block(x, 3, [256, 256, 1024], name="backbone.layers.2.3")
    x = identity_block(x, 3, [256, 256, 1024], name="backbone.layers.2.4")
    x = identity_block(x, 3, [256, 256, 1024], name="backbone.layers.2.5")
    y2 = x

    # 34, 34, 1024 -> 17, 17, 2048
    x = conv_block(x, 3, [512, 512, 2048], name="backbone.layers.3.0")
    x = identity_block(x, 3, [512, 512, 2048], name="backbone.layers.3.1")
    x = identity_block(x, 3, [512, 512, 2048], name="backbone.layers.3.2")
    y3 = x
    return y1, y2, y3

