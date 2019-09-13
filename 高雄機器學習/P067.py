
#MNIST Sample MNIST的图像数据是28 像素× 28 像素的灰度图像（1 通道）
# MNIST的图像数据是28 像素× 28 像素的灰度图像（1 通道），各个像素的取值在0 到255 之间。每个图像数据都相应地标有“7”“2”“1”等标签。
# 使用mnist.py中的load_mnist()函数，就可以按下述方式轻松读入MNIST数据
# import sys, os
# os.chdir('/Users/shuhuimeng/ch03/')
# sys.path.append(os.pardir)  
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
# 输出各个数据的形状
print(label)
print(img.shape)# 把图像的形状变成原来的尺寸
img = img.reshape(28, 28)
print(img.shape)
print(x_train.shape) # (60000, 784)

print(t_train.shape) # (60000,)
print(x_test.shape) # (10000, 784)
print(t_test.shape) # (10000,)
img_show(img)

# load_mnist函数以“( 训练图像, 训练标签)，( 测试图像，测试标签)”的形式返回读入的MNIST数据。此外，还可以像load_mnist(normalize=True,

# flatten=True, one_hot_label=False) 这样，设置3 个参数。第1 个参数normalize设置是否将输入图像正规化为0.0～1.0 的值。如果将该参数设置

# 为False，则输入图像的像素会保持原来的0～255。第2 个参数flatten设置是否展开输入图像（变成一维数组）。如果将该参数设置为False，则输入图

# 像为1 × 28 × 28 的三维数组；若设置为True，则输入图像会保存为由784 个元素构成的一维数组。第3 个参数one_hot_label设置是否将标签保存为

# onehot表示（one-hot representation）。one-hot 表示是仅正确解标签为1，其余皆为0 的数组，就像[0,0,1,0,0,0,0,0,0,0]这样。

# 当one_hot_label为False时，只是像7、2这样简单保存正确解标签；当one_hot_label为True时，标签则保存为one-hot 表示。



