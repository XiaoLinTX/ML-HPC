import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import caffe as cf
#import caffe as cf
def one_hot(labels):
  labels_hot = np.zeros((10,labels.shape[0]))  #对一维列表，大小元组只有只有１个数
  for i in range(labels.shape[0]):
    labels_hot[labels[i],i]=1
  return labels_hot

#数据预处理，分离出训练集，标签，测试集
def data_read():
  train = pd.read_csv(r'kaggle_手写数字识别/train.csv')
  test=pd.read_csv(r'kaggle_手写数字识别/test.csv')

  x_train=train.iloc[:,1:].to_numpy().astype(np.float)
  x_test=test.to_numpy().astype(np.float)

  #信息归一在0-1
  x_train=np.multiply(x_train,1.0/255.0).T #x_train.shape(42000,784),转置为（784,42000）
  x_test=np.multiply(x_test,1.0/255.0).T

  labels=train.iloc[:,0].to_numpy().T
  labels=one_hot(labels)

  return x_train,labels,x_test

def data_write(predict_labels):
  id_list=[x for x in range(1,len(predict_labels)+1)]
  dataFrame=pd.DataFrame({'ImageId':id_list,'Label':predict_labels})   
  dataFrame.to_csv('kaggle_手写数字识别/submission.csv',index=None)    

if __name__=='__main__':

  x_train,labels,x_test=data_read()

  # #设置批次大小每批100，
  # x=tf.placeholder('float',shape=[784,100])
  # y=tf.placeholder('float',shape=[10,100])   #输出层对应10个标签

  #输入层,10个神经元(w,b参数设定时当成一个特征向量看)
  weights=tf.Variable(tf.zeros([10,784]))   #10对应10个输出神经元
  biases=tf.Variable(tf.zeros([10,1]))
  prediction=tf.nn.softmax(tf.matmul(weights,x)+biases) #单张图片看，得10*1的概率矩阵,不过此处是从多张图片看,得10*100的概率矩阵
  output=tf.argmax(prediction,axis=0)  #求每一列的最大值所在的下标,即是标签本身

  #损失函数:交叉熵平均值
  loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))#使得输出的10*100的softmax概率矩阵趋近于10*100的0-1标签矩阵
  #梯度下降法优化参数
  train_step=tf.train.AdamOptimizer(1e-2).minimize(loss)     
                                           


    for epoch in range(200): #迭代总次数
      print("第"+str(epoch)+'次迭代训练')
      for batch in range(int(42000/100)):#每批次100组训练集特征向量
        batch_x=x_train[:,batch*100:(batch+1)*100]
        batch_y=labels[:,batch*100:(batch+1)*100]  
        sess.run(train_step,feed_dict={x:batch_x,y:batch_y}) 

    predict_labels=[]
    for batch in range(int(28000/100)):#每批次100组测试集特征向量
      batch_x=x_test[:,batch*100:(batch+1)*100]
      predict_labels=predict_labels+sess.run(output,feed_dict={x:batch_x}).tolist()

  data_write(predict_labels)
  


















