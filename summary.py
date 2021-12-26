#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from nets.yolact import yolact

if __name__ == "__main__":
    model = yolact([544, 544, 3], 81, mode="predict")
    model.summary()

    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)