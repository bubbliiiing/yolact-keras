## Yolact-keras实例分割模型在keras当中的实现
---

## 目录
1. [性能情况 Performance](#性能情况)
2. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [训练步骤 How2train](#训练步骤)
5. [预测步骤 How2predict](#预测步骤)
6. [评估步骤 How2eval](#评估步骤)
7. [参考资料 Reference](#Reference)

## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | bbox mAP 0.5:0.95 | bbox mAP 0.5 | segm mAP 0.5:0.95 | segm mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: | :-----: | :-----: | 
| COCO-Train2017 | [yolact_weights_coco.h5](https://github.com/bubbliiiing/yolact-keras/releases/download/v1.0/yolact_weights_coco.h5) | COCO-Val2017 | 544x544 | 30.3 | 51.8 | 27.1 | 47.2

## 所需环境
keras==2.1.5   
tensorflow-gpu==1.13.2

## 文件下载
训练所需的预训练权值可在百度网盘中下载。  
链接: https://pan.baidu.com/s/1OIxe9w2t5nImstDEpjncnQ    
提取码: eik3   

shapes数据集下载地址如下，该数据集是使用labelme标注的结果，尚未经过其它处理，用于区分三角形和正方形：  
链接: https://pan.baidu.com/s/1hrCaEYbnSGBOhjoiOKQmig   
提取码: jk44    

## 训练步骤
### a、训练shapes形状数据集（训练自己的数据集）
1. 数据集的准备   
在**文件下载**部分，通过百度网盘下载数据集，下载完成后解压，将图片和对应的json文件放入根目录下的datasets/before文件夹。

2. 数据集的处理   
打开coco_annotation.py，里面的参数默认用于处理shapes形状数据集，直接运行可以在datasets/coco文件夹里生成图片文件和标签文件，并且完成了训练集和测试集的划分。

3. 开始网络训练   
train.py的默认参数用于训练shapes数据集，默认指向了根目录下的数据集文件夹，直接运行train.py即可开始训练。   

4. 训练结果预测   
训练结果预测需要用到两个文件，分别是yolact.py和predict.py。我们首先需要去yolact.py里面修改model_path以及classes_path，这两个参数必须要修改。    
**model_path指向训练好的权值文件，在logs文件夹里。   
classes_path指向检测类别所对应的txt。**    
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。   

### b、训练自己的数据集
1. 数据集的准备  
**本文使用labelme工具进行标注，标注好的文件有图片文件和json文件，二者均放在before文件夹里，具体格式可参考shapes数据集。**    

2. 数据集的处理  
在完成数据集的摆放之后，我们需要利用coco_annotation.py获得训练用的标签文件。    
修改coco_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。    
训练自己的数据集时，可以自己建立一个cls_classes.txt，里面写自己所需要区分的类别。    
model_data/cls_classes.txt文件内容为：      
```python
cat
dog
...
```  
修改coco_annotation.py中的classes_path，使其对应cls_classes.txt，并运行coco_annotation.py。    

3. 开始网络训练  
**训练的参数较多，均在train.py中，大家可以在下载库后仔细看注释，其中最重要的部分依然是train.py里的classes_path。**   
**classes_path用于指向检测类别所对应的txt，这个txt和coco_annotation.py里面的txt一样！训练自己的数据集必须要修改！**    
修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。   

4. 训练结果预测  
训练结果预测需要用到两个文件，分别是yolact.py和predict.py。在yolact.py里面修改model_path以及classes_path。    
**model_path指向训练好的权值文件，在logs文件夹里。     
classes_path指向检测类别所对应的txt。**     
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。     

### c、训练coco数据集
1. 数据集的准备  
coco训练集 http://images.cocodataset.org/zips/train2017.zip   
coco验证集 http://images.cocodataset.org/zips/val2017.zip   
coco训练集和验证集的标签 http://images.cocodataset.org/annotations/annotations_trainval2017.zip   

2. 开始网络训练  
解压训练集、验证集及其标签后。打开train.py文件，修改其中的classes_path指向model_data/coco_classes.txt。   
修改train_image_path为训练图片的路径，train_annotation_path为训练图片的标签文件，val_image_path为验证图片的路径，val_annotation_path为验证图片的标签文件。   

3. 训练结果预测  
训练结果预测需要用到两个文件，分别是yolact.py和predict.py。在yolact.py里面修改model_path以及classes_path。   
**model_path指向训练好的权值文件，在logs文件夹里。   
classes_path指向检测类别所对应的txt。**   
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。   

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载权值，放入model_data，运行predict.py，输入   
```python
img/street.jpg
```
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。   
### b、使用自己训练的权重
1. 按照训练步骤训练。    
2. 在yolact.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。   
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #
    #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
    #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/yolact_weights_shape.h5',
    "classes_path"      : 'model_data/shape_classes.txt',
    #---------------------------------------------------------------------#
    #   输入图片的大小
    #---------------------------------------------------------------------#
    "input_shape"       : [544, 544],
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #---------------------------------------------------------------------#
    #   先验框的大小
    #---------------------------------------------------------------------#
    "anchors_size"      : [24, 48, 96, 192, 384],
    #---------------------------------------------------------------------#
    #   传统非极大抑制
    #---------------------------------------------------------------------#
    "traditional_nms"   : True
}
```
3. 运行predict.py，输入    
```python
img/street.jpg
```
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。    

## 评估步骤 
### a、评估自己的数据集
1. 本文使用coco格式进行评估。    
2. 如果在训练前已经运行过coco_annotation.py文件，代码会自动将数据集划分成训练集、验证集和测试集。如果想要修改测试集的比例，可以修改coco_annotation.py文件下的trainval_percent。trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1。train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1。
3. 利用coco_annotation.py划分测试集后，前往eval.py文件修改classes_path，classes_path用于指向检测类别所对应的txt，这个txt和训练时的txt一样。评估自己的数据集必须要修改。  
4. 在yolact.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**    
5. 运行eval.py即可获得评估结果。  

### b、评估coco的数据集
1、下载好coco数据集。  
2、前往eval.py设置classes_path，指向model_data/coco_classes.txt。  
3、在yolact.py里面修改model_path以及classes_path。**model_path指向coco数据集的权重，在logs文件夹里。classes_path指向model_data/coco_classes.txt。**    
4、运行eval.py即可获得评估结果。  

## Reference
https://github.com/feiyuhuahuo/Yolact_minimal   
