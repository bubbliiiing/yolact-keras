import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from pycocotools.coco import COCO

from nets.yolact import get_train_model, yolact
from utils.anchors import get_anchors
from utils.augmentations import Augmentation
from utils.callbacks import ExponentDecayScheduler, LossHistory
from utils.dataloader import COCODetection
from utils.utils import get_classes

if __name__ == "__main__":
    #------------------------------------#
    #   训练自己的数据集必须要修改
    #   所需要区分的类别对应的txt文件
    #------------------------------------#
    classes_path    = 'model_data/shape_classes.txt'   
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path为主干网络的权值，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    #
    #   网络一般不从0开始训练，至少会使用主干部分的权值，有些论文提到可以不用预训练，主要原因是他们 数据集较大 且 调参能力优秀。
    #   如果一定要训练网络的主干部分，可以了解imagenet数据集，首先训练分类模型，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = "model_data/yolact_weights_coco.h5"
    #------------------------------------------------------#
    #   输入的shape大小
    #------------------------------------------------------#
    input_shape     = [544, 544]
    #------------------------------------------------------#
    #   获取先验框大小
    #------------------------------------------------------#
    anchors_size    = [24, 48, 96, 192, 384]

    #----------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，batch_size最小为2，不能为1。
    #----------------------------------------------------#
    #----------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #----------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 8
    Freeze_lr           = 1e-4
    #----------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #----------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 4
    Unfreeze_lr         = 1e-5
    #------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    #------------------------------------------------------#
    Freeze_Train        = True
    #------------------------------------------------------#
    #   用于设置是否使用多线程读取数据，1代表关闭多线程
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   keras里开启多线程有些时候速度反而慢了许多
    #   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    #------------------------------------------------------#
    num_workers         = 1
    #----------------------------------------------------#
    #   获得图片路径和标签
    #   默认指向根目录下面的datasets/coco文件夹
    #----------------------------------------------------#
    train_image_path        = "datasets/coco/JPEGImages"
    train_annotation_path   = "datasets/coco/Jsons/train_annotations.json"
    val_image_path          = "datasets/coco/JPEGImages"
    val_annotation_path     = "datasets/coco/Jsons/val_annotations.json"

    #----------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    num_classes = num_classes + 1
    anchors     = get_anchors(input_shape, anchors_size)
    
    model_body  = yolact([input_shape[0], input_shape[1], 3], num_classes, train_mode = True)
    if model_path != "":
        #------------------------------------------------------#
        #   载入预训练权重
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        model_body.load_weights(model_path, by_name=True, skip_mismatch=True)

    model = get_train_model(model_body)
    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    logging         = TensorBoard(log_dir = 'logs/')
    checkpoint      = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                        monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
    reduce_lr       = ExponentDecayScheduler(decay_rate = 0.94, verbose = 1)
    early_stopping  = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    loss_history    = LossHistory('logs/')

    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    train_coco  = COCO(train_annotation_path)
    val_coco    = COCO(val_annotation_path)
    num_train   = len(list(train_coco.imgToAnns.keys()))
    num_val     = len(list(val_coco.imgToAnns.keys()))

    if Freeze_Train:
        for i in range(173):
            model.layers[i].trainable = False

    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch
        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        train_dataloader    = COCODetection(train_image_path, train_coco, num_classes, anchors, batch_size, Augmentation(input_shape))
        val_dataloader      = COCODetection(val_image_path, train_coco, num_classes, anchors, batch_size, Augmentation(input_shape))

        model.compile(loss={'yolact_Loss': lambda y_true, y_pred: y_pred}, optimizer = keras.optimizers.Adam(lr=lr))

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            generator           = train_dataloader,
            steps_per_epoch     = epoch_step,
            validation_data     = val_dataloader,
            validation_steps    = epoch_step_val,
            epochs              = end_epoch,
            initial_epoch       = start_epoch,
            use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )

    if Freeze_Train:
        for i in range(173):
            model.layers[i].trainable = True

    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch

        epoch_step          = num_train // batch_size
        epoch_step_val      = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        train_dataloader    = COCODetection(train_image_path, train_coco, num_classes, anchors, batch_size, Augmentation(input_shape))
        val_dataloader      = COCODetection(val_image_path, train_coco, num_classes, anchors, batch_size, Augmentation(input_shape))

        model.compile(loss={'yolact_Loss': lambda y_true, y_pred: y_pred}, optimizer = keras.optimizers.Adam(lr=lr))

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            generator           = train_dataloader,
            steps_per_epoch     = epoch_step,
            validation_data     = val_dataloader,
            validation_steps    = epoch_step_val,
            epochs              = end_epoch,
            initial_epoch       = start_epoch,
            use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )
