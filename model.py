# -*- coding: UTF-8 -*-
import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation,Flatten,MaxPool2D,Conv2D

from keras.optimizers import SGD, Adam, RMSprop

from keras.utils.np_utils import to_categorical

from keras.utils.vis_utils import plot_model

from sklearn.model_selection import train_test_split

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import VotingClassifier
import keras

import itertools
batch_size = 256 # 在计算力允许的情况下，batch_size越大越好

nb_classes = 10

nb_epoch = 30

originPath='.\data\\'
train_data=pd.read_csv(originPath+'train.csv')
test_data=pd.read_csv(originPath+'test.csv')

train_data.info()
print('_____________________________________________________')
test_data.info()
X_train=train_data.drop(columns=['label'])         #删除名为"label"列
Y_train=train_data.label                           #选取label这一列
print('_____________________________________________________')

# del train_data                         #删除变量train_data，并不删除数据
# 绘制计数直方图
sns.countplot(Y_train)                 #计数直方图
plt.show()
# 使用pd.Series.value_counts()
print(Y_train.value_counts())
X_train=train_data.drop(columns=['label'])
Y_train=train_data.label
del train_data
# 改变维度：第一个参数是图片数量，后三个参数是每个图片的维度
X_train = X_train.values.reshape(-1,28,28,1)                   #图片数量，图片宽度，图片长度，通道数
test_data = test_data.values.reshape(-1,28,28,1)
print(X_train.shape)
print(test_data.shape)
print("Train Sample:",X_train.shape[0])
print("Test Sample:",test_data.shape[0])
# 归一化：将数据进行归一化到0-1 因为图像数据最大是255
X_train=X_train/255.0
test_data=test_data/255.0
# 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵
Y_train = keras.utils.to_categorical(Y_train, num_classes = 10) #讲Y值写成one-hot形式
X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.1)
plt.imshow(X_train[0][:,:,0], cmap="Greys")
plt.show()
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),activation ='relu', input_shape = (28,28,1))) #用维度5的32个卷积块进行卷积，激活函数为relu
model.add(Conv2D(filters = 32, kernel_size = (5,5),activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2))) #最大值池化，使图片维度变为原来的1/2
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))  #最大值池化，使图片维度变为原来的1/2,步长为2
model.add(Dropout(0.25))  #每次更新参。数时按一定0.25 随机断开输入神经元，Dropout层用于防止过拟合
model.add(Flatten())      #把多维的输入一维化
model.add(Dense(256, activation = "relu"))   #输出维度为256的向量，并且使用激活函数rule
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax")) #输出维度为10的向量，并且使用激活函数sotamax,归一化处理
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # 图片随机转动的角度
        zoom_range = 0.1, #随机缩放的幅度
        width_shift_range=0.1,  # 图片水平偏移的幅度
        height_shift_range=0.1,  # 图片垂直偏移的幅度
        horizontal_flip=False,  # 进行随机水平翻转
        vertical_flip=False)  # 进行竖直水平翻转




# 当评价指标monitor不在提升时，减少学习率
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs =nb_epoch, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size+1,
                              callbacks=[learning_rate_reduction])
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
ax[0].legend(loc='best', shadow=True)
ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax[1].legend(loc='best', shadow=True)
plt.show()

score = model.evaluate(X_val, Y_val, verbose=0)  #对modle进行评估
print('Val loss:', score[0])
print('Val accuracy:', score[1])

results = model.predict(test_data)
print('results loss:', results)
print('results accuracy:', results[1])
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv(originPath+"submit.csv",index=False)
