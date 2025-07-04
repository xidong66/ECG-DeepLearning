##########################################################################################################
#使用#from utils import load_data,check_gpu_availability,plot_confusion_matrix,plot_loss_accuracy
##########################################################################################################
import os
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import tensorflow as tf
import imblearn.over_sampling as imb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
##########################################################################################################
#查看当前各类标签个数
##########################################################################################################
#from utils import count_labels
def count_labels(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    return label_counts
##########################################################################################################
#准确率和损失函数
##########################################################################################################
from matplotlib import pyplot as plt
def plot_loss_accuracy(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    #plt.savefig('loss_accuracy_plot.png', dpi=600)
    plt.show()
    
import matplotlib.pyplot as plt
import os
from datetime import datetime  # 正确导入 datetime 类

def plot_loss_accuracytupian(history, save_dir="C:\\Users\\Administrator\\Desktop\\p-final\\tupian"):
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  plt.figure(figsize=(12, 4))
  plt.subplot(1, 2, 1)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.title('Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(accuracy, label='Training Accuracy')
  plt.plot(val_accuracy, label='Validation Accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()

  plt.tight_layout()

  # 检查文件夹是否存在，如果不存在，则创建
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  # 获取当前日期和时间，用于文件命名
  current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
  file_name = f"loss_accuracy_plot_{current_time}.png"

  # 保存图像到指定的文件夹，并自动命名
  plt.savefig(os.path.join(save_dir, file_name), dpi=600)
  plt.show()


import matplotlib.pyplot as plt
import os
from datetime import datetime  # 正确导入 datetime 类

def plot_loss_accuracy_separate2(history, save_dir="C:\\Users\\Administrator\\Desktop\\p-final\\tupian"):
  # 提取损失函数和准确率数据
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  # 检查文件夹是否存在，如果不存在，则创建
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  # 获取当前日期和时间，用于文件命名
  current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

  # 绘制损失函数图
  plt.figure(figsize=(6, 4))
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.title('Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  loss_file_name = f"loss_plot_{current_time}.png"
  plt.savefig(os.path.join(save_dir, loss_file_name), dpi=600)
  plt.show()

  # 绘制准确率图
  plt.figure(figsize=(6, 4))
  plt.plot(accuracy, label='Training Accuracy')
  plt.plot(val_accuracy, label='Validation Accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  accuracy_file_name = f"accuracy_plot_{current_time}.png"
  plt.savefig(os.path.join(save_dir, accuracy_file_name), dpi=600)
  plt.show()

##########################################################################################################

import matplotlib.pyplot as plt
import os
from datetime import datetime  # 正确导入 datetime 类

def plot_loss_accuracytupian(history, save_dir="C:\\Users\\Administrator\\Desktop\\pic"):
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  plt.figure(figsize=(12, 4))
  plt.subplot(1, 2, 1)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.title('Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(accuracy, label='Training Accuracy')
  plt.plot(val_accuracy, label='Validation Accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()

  plt.tight_layout()

  # 检查文件夹是否存在，如果不存在，则创建
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  # 获取当前日期和时间，用于文件命名
  current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
  file_name = f"loss_accuracy_plot_{current_time}.png"

  # 保存图像到指定的文件夹，并自动命名
  plt.savefig(os.path.join(save_dir, file_name), dpi=600)
  plt.show()


##########################################################################################################
#混淆矩阵
##########################################################################################################
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    #plt.savefig('confusion_matrix1.png', dpi=600)
    plt.show()
##########################################################################################################
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix2(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    num_samples = np.sum(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix (n={num_samples})')
    #plt.savefig('confusion_matrix2.png', dpi=600)
    plt.show()
##########################################################################################################
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix3(y_true, y_pred, classes, save_dir="C:\\Users\\Administrator\\Desktop\\pic"):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    file_name = 'confusion_matrix_{}.png'.format(len(os.listdir(save_dir)) + 1)
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, dpi=600)
    plt.show()

##########################################################################################################
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import datetime
import os

def plot_confusion_matrixtupian(y_true, y_pred, classes, directory='C:\\Users\\Administrator\\Desktop\\p-final\\tupian'):
  """
  绘制混淆矩阵，并保存在指定文件夹。

  参数:
  y_true (list): 真实标签。
  y_pred (list): 预测标签。
  classes (list): 类别名称。
  directory (str): 保存图片的目录。
  """
  # 生成文件名
  timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  filename = f'confusion_matrix_{timestamp}.png'

  # 确保目录存在
  if not os.path.exists(directory):
    os.makedirs(directory)

  cm = confusion_matrix(y_true, y_pred)
  num_samples = np.sum(cm)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.title(f'Confusion Matrix (n={num_samples})')

  # 保存图片
  full_path = os.path.join(directory, filename)
  plt.savefig(full_path, dpi=600)
  plt.show()
  print(f"图像已保存为: {full_path}")

# 使用示例:
# plot_confusion_matrix2(y_true, y_pred, classes, '/path/to/your/folder')
 
##########################################################################################################
#gpu检查
##########################################################################################################
import tensorflow as tf
def check_gpu_availability():
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print('GPU 可用')
    else:
        print('GPU 不可用')
##########################################################################################################
import tensorflow as tf
def get_gpu_info():
    gpus = tf.config.list_physical_devices('GPU')
    gpu_models = []
    for gpu in gpus:
        gpu_models.append(gpu.name)
    gpu_count = len(gpus)
    print('当前正在使用的显卡型号:')
    for model in gpu_models:
        print(model)
    print('你有 {} 个显卡可用'.format(gpu_count))
##########################################################################################################
#导入数据的集成
##########################################################################################################
import numpy as np
def load_data(datafilename1, datafilename2):
    data1 = np.load(datafilename1)
    data2 = np.load(datafilename2) 
    X_train, y_train = data1['ecgs'], data1['labels']
    X_test, y_test = data2['ecgs'], data2['labels']
    return X_train, y_train, X_test, y_test
##########################################################################################################
#评估的集成
##########################################################################################################
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
def evaluate_model2(model, X_test, y_test):
    y_pred_classes = np.argmax(model.predict(X_test), axis=-1)
    y_test_classes = np.argmax(y_test, axis=1)
    precision = precision_score(y_test_classes, y_pred_classes, average='macro')
    recall = recall_score(y_test_classes, y_pred_classes, average='macro')
    f1 = f1_score(y_test_classes, y_pred_classes, average='macro')
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 Score: ', f1)
    print('Accuracy: ', accuracy)
    f1_scores = f1_score(np.argmax(y_test, axis=1), y_pred_classes, average=None)
    for i, score in enumerate(f1_scores):
        print(f"F1 Score for Class {i+1}: {score}")
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# def evaluate_model(model, X_test, y_test):
#     # 预测测试集
#     y_pred = model.predict(X_test)
#     y_pred_classes = np.argmax(y_pred, axis=-1)
#     y_test_classes = np.argmax(y_test, axis=1)
    
#     # 计算宏观平均精确度、召回率、F1分数和总准确率
#     precision_macro = precision_score(y_test_classes, y_pred_classes, average='macro')
#     recall_macro = recall_score(y_test_classes, y_pred_classes, average='macro')
#     f1_macro = f1_score(y_test_classes, y_pred_classes, average='macro')
#     accuracy = accuracy_score(y_test_classes, y_pred_classes)
    
#     print('Precision (Macro Average):', precision_macro)
#     print('Recall (Macro Average):', recall_macro)
#     print('F1 Score (Macro Average):', f1_macro)
#     print('Accuracy:', accuracy)
    
#     # 计算每个类的精确度、召回率和F1分数
#     precisions = precision_score(y_test_classes, y_pred_classes, average=None)
#     recalls = recall_score(y_test_classes, y_pred_classes, average=None)
#     f1_scores = f1_score(y_test_classes, y_pred_classes, average=None)
    
#     # 打印每个类别的评分
#     for i in range(len(precisions)):
#         print(f'Class {i+1} - Precision: {precisions[i]}, Recall: {recalls[i]}, F1 Score: {f1_scores[i]}')
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
def evaluate_model(model, X_test, y_test):
    # 预测测试集
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=-1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # 计算宏观平均精确度、召回率、F1分数和总准确率
    precision = precision_score(y_test_classes, y_pred_classes, average='macro')
    recall = recall_score(y_test_classes, y_pred_classes, average='macro')
    f1 = f1_score(y_test_classes, y_pred_classes, average='macro')
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Accuracy: {accuracy}')
    
    # 计算每个类的精确度、召回率和F1分数
    precisions = precision_score(y_test_classes, y_pred_classes, average=None)
    recalls = recall_score(y_test_classes, y_pred_classes, average=None)
    f1_scores = f1_score(y_test_classes, y_pred_classes, average=None)
    
    # 打印每个类别的评分
    for i, (p, r, f) in enumerate(zip(precisions, recalls, f1_scores)):
        print(f'Class {i+1} - Precision: {p}, Recall: {r}, F1 Score: {f}')
    
    # 计算并打印每个类别的准确率
    for i in range(np.max(y_test_classes) + 1):
        class_accuracy = accuracy_score(y_test_classes == i, y_pred_classes == i)
        print(f'Class {i+1} Accuracy: {class_accuracy}')

##########################################################################################################
#标签处理的集成
##########################################################################################################
def encode_labels(labels):
    new_labels = []
    for label in labels:
        if label == 'A':
            new_labels.append(0)
        elif label == 'N':
            new_labels.append(1)
        elif label == 'O':
            new_labels.append(2)
        elif label == '~':
            new_labels.append(3)
    return np.array(new_labels)
##########################################################################################################
#输出标签的集成
##########################################################################################################
def printlabels(data):
    variable_names = data.keys()
    for name in variable_names:
        print(name)
##########################################################################################################
#tsne的制图
##########################################################################################################
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
def plot_tsne(features, labels, epoch, fileNameDir=None):
    if not os.path.exists(fileNameDir):
        os.makedirs(fileNameDir)
    tsne = TSNE(n_components=2, init='pca', random_state=0,learning_rate=0.01, n_iter=1000)
    tsne_features = tsne.fit_transform(features)
    x_min, x_max = np.min(tsne_features, 0), np.max(tsne_features, 0)
    tsne_features = (tsne_features - x_min) / (x_max - x_min)
    plt.figure()
    colors = ['r', 'g', 'b', 'y']  
    for i in range(len(labels)):
        plt.scatter(tsne_features[i, 0], tsne_features[i, 1], c=colors[int(labels[i])], marker='.')
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(os.path.join(fileNameDir, f"{epoch}.jpg"), format='jpg', dpi=600)
    plt.close()
##########################################################################################################
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
def plot_tsne2(features, labels):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    class_num = len(np.unique(labels))
    tsne_features = tsne.fit_transform(features) 
    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tsne_features[:,0]
    df["comp-2"] = tsne_features[:,1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", class_num), data=df)
    plt.savefig("Fig1.png")
    plt.show()
##########################################################################################################
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
def plot_tsne3(features, labels, fileName="Fig1.png"):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    class_num = len(np.unique(labels))
    tsne_features = tsne.fit_transform(features) 
    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tsne_features[:,0]
    df["comp-2"] = tsne_features[:,1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", class_num), data=df)
    plt.savefig(fileName)
    plt.show()
##########################################################################################################
#回调函数
##########################################################################################################

class EarlyStopping:
    def __init__(self, monitor, patience):
        self.monitor = monitor
        self.patience = patience
        self.wait = 0
        self.best_loss = float('inf')
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print("Epoch {}: early stopping".format(epoch))
                self.stopped_epoch = epoch
                return True
        return False
##########################################################################################################
#四阶巴特沃斯滤波
##########################################################################################################
import numpy as np
from scipy.signal import butter, lfilter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def apply_butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    filtered_data = butter_bandpass_filter(data, lowcut, highcut, fs, order)
    return filtered_data
##########################################################################################################
#波形图启动器
##########################################################################################################
import matplotlib.pyplot as plt
def plot_waveform(data, num_samples, waveform_index, fs, title):
    waveform = data[waveform_index][:num_samples]  # 获取指定波形的数据
    t = range(num_samples)  # 生成时间序列
    plt.figure(figsize=(10, 4))
    plt.plot(t, waveform)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)  # 设置图片标题
    plt.grid(True)
    plt.show()
#plot_waveform(X_val, 6000,0, 300, "")
import matplotlib.pyplot as plt
import numpy as np

def plot_waveform2(data, num_samples, fs, title):
  """
  绘制波形图。

  参数:
  data (np.array): 波形数据。
  num_samples (int): 采样点数。
  fs (int): 采样频率。
  title (str): 图表标题。
  """
  # 生成时间序列
  t = np.linspace(0, num_samples / fs, num_samples, endpoint=False)

  # 绘制波形图
  plt.figure(figsize=(10, 4))
  plt.plot(t, data[:num_samples])
  plt.xlabel('Time (s)')
  plt.ylabel('Amplitude')
  plt.title(title)
  plt.grid(True)
  plt.show()

# 示例调用
# plot_waveform(your_data_array, 9000, fs, '原波形图')
# 其中 your_data_array 是你要绘制的波形数据，fs 是采样频率。
##########################################################################################################
#究极数据包装函数
##########################################################################################################
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
def load_and_preprocess_data(datafilename1,test_size=0.1, random_state1=42,random_state2=42):
    data1 = np.load(datafilename1) 
    X_train, y_train = data1['ecgs'], data1['labels']
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state1)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=test_size, random_state=random_state2)
    y_train = encode_labels(y_train)
    y_test = encode_labels(y_test)
    y_val = encode_labels(y_val)
    y_train = to_categorical(y_train, num_classes=4)
    y_val = to_categorical(y_val, num_classes=4)
    y_test = to_categorical(y_test, num_classes=4)
    return X_train, X_val, X_test, y_train, y_val, y_test
##########################################################################################################
#小波滤波
##########################################################################################################
import pywt
def denoise(data):
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata
def denoise2(data):
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cA9.fill(0)
    cD9.fill(0)
    cD8.fill(0)
    cD7.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata
import numpy as np
import pywt


##########################################################################################################
#评估模型
##########################################################################################################
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
def evaluate_predictionss(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("R平方:", r2)
    print("均方根误差:", rmse)
    print("平均绝对误差:", mae)
    print("均方误差:", mse)


import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_model3(model, X_test, y_test):
    # 预测测试集
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=-1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # 计算宏观平均精确度、召回率、F1分数和总准确率
    precision = precision_score(y_test_classes, y_pred_classes, average='macro')
    recall = recall_score(y_test_classes, y_pred_classes, average='macro')
    f1 = f1_score(y_test_classes, y_pred_classes, average='macro')
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Accuracy: {accuracy}')
    
    # 计算每个类的精确度、召回率和F1分数
    precisions = precision_score(y_test_classes, y_pred_classes, average=None)
    recalls = recall_score(y_test_classes, y_pred_classes, average=None)
    f1_scores = f1_score(y_test_classes, y_pred_classes, average=None)
    
    # 打印每个类别的评分
    for i, (p, r, f) in enumerate(zip(precisions, recalls, f1_scores)):
        print(f'Class {i+1} - Precision: {p}, Recall: {r}, F1 Score: {f}')
    
    # 计算并打印每个类别的准确率
    for i in range(np.max(y_test_classes) + 1):
        class_accuracy = accuracy_score(y_test_classes == i, y_pred_classes == i)
        print(f'Class {i+1} Accuracy: {class_accuracy}')

##########################################################################################################
#数据增强
##########################################################################################################
import numpy as np
def data_augmentation(X, y):
  X_aug = []
  y_aug = []
  for ecg, label in zip(X, y):
    if label == '\~':
      continue
    X_aug.append(ecg[0:6000])
    y_aug.append(label)
    X_aug.append(ecg[1500:7500]) 
    y_aug.append(label)
    X_aug.append(ecg[3000:9000])
    y_aug.append(label)
  X_aug = np.array(X_aug)
  y_aug = np.array(y_aug)
  return X_aug, y_aug
##########################################################################################################
#
##########################################################################################################
import numpy as np

def data_augmentation2(X, y):

  X_aug2, y_aug2 = [], []
  X_aug3, y_aug3 = [], []
  X_aug4, y_aug4 = [], []
  X_aug5, y_aug5 = [], []
  for ecg, label in zip(X, y):
    if label == '\\~':  
        ecg1 = ecg[0:3000]
        extended_ecg = ecg1
        X_aug2=np.concatenate((X_aug3, X_aug4, X_aug5, X_test), axis=0)
        X_aug2.append(extended_ecg)
        y_aug2.append('\~')

  X_aug2 = np.array(X_aug2)
  X_aug3 = X_aug2[0:6000]
  y_aug3.append('\~')
  
  X_aug4 = X_aug2[3000:6000]
  y_aug4 .append(label)
  X_aug5 = X_aug2[6000:12000]
  y_aug5 = np.full(6000, '\~')

  X_test = np.concatenate((X_aug3, X_aug4, X_aug5, X_test), axis=0)
  y_test = np.concatenate((y_aug3, y_aug4, y_aug5, y_test), axis=0)

  return X_test, y_test

##########################################################################################################
#ROC曲线
##########################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

def plot_roc_curve_multiclass(y_test, y_scores):
  num_classes = 4  
  y_test_bin = label_binarize(y_test, classes=range(num_classes))
  
  if y_scores.ndim == 1:
    y_scores = np.reshape(y_scores, (-1, num_classes))

  plt.figure(figsize=(8, 6))  
  colors = [(232/255, 56/255, 71/255), (221/255, 108/255, 76/255), (69/255, 123/255, 157/255), (29/255, 53/255, 87/255)]
  linestyles = ['-', '--', '-.', ':']
  labels = ['ROC curve of AF', 'ROC curve of normal', 'ROC curve of other', 'ROC curve of noise']
  
  for i in range(num_classes):
    fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], y_scores[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], linestyle=linestyles[i], lw=2, label='{} (AUC = {:.2f})'.format(labels[i], roc_auc))

  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')

  plt.legend(loc='lower right', fontsize=12)
  plt.grid(True)
  plt.show()

def plot_roc_curve_multiclass2(y_test, y_scores):
  num_classes = 4
  y_test_bin = label_binarize(y_test, classes=range(num_classes))
  if y_scores.ndim == 1:
    y_scores = np.reshape(y_scores, (-1, num_classes))

  plt.figure(figsize=(8, 6))
  colors = [(232/255, 56/255, 71/255), (221/255, 108/255, 76/255), (69/255, 123/255, 157/255), (29/255, 53/255, 87/255)]
  linestyles = ['-', '--', '-.', ':']
  labels = ['ROC curve of AF', 'ROC curve of normal', 'ROC curve of other', 'ROC curve of noise']

  # 计算每个类别的ROC曲线和AUC
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[i], linestyle=linestyles[i], lw=2, label='{} (AUC = {:.2f})'.format(labels[i], roc_auc[i]))

  # 计算微观平均ROC曲线和AUC
  fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_scores.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
  plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='Micro-average ROC curve (AUC = {:.2f})'.format(roc_auc["micro"]))

  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Multi-class ROC with Micro-average')
  plt.legend(loc='lower right', fontsize=12)
  plt.grid(True)
  plt.show()

# 测试数据
# y_test = np.array([...])  # 测试集标签
# y_scores = np.array([...])  # 模型预测的得分
# plot_roc_curve_multiclass(y_test, y_scores)
##########################################################################################################
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import datetime
import os

def plot_roc_curve_multiclasstupian(y_test, y_scores, directory='C:\\Users\\Administrator\\Desktop\\p-final\\tupian'):
  """
  绘制多类别的ROC曲线，并保存在指定文件夹。

  参数:
  y_test (array): 真实标签数组。
  y_scores (array): 预测得分数组。
  directory (str): 保存图片的目录。
  """
  num_classes = 4
  y_test_bin = label_binarize(y_test, classes=range(num_classes))
  if y_scores.ndim == 1:
    y_scores = np.reshape(y_scores, (-1, num_classes))

  plt.figure(figsize=(8, 6))
  colors = [(232/255, 56/255, 71/255), (221/255, 108/255, 76/255), (69/255, 123/255, 157/255), (29/255, 53/255, 87/255)]
  linestyles = ['-', '--', '-.', ':']
  labels = ['ROC curve of AF', 'ROC curve of normal', 'ROC curve of other', 'ROC curve of noise']

  # 计算每个类别的ROC曲线和AUC
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[i], linestyle=linestyles[i], lw=2, label='{} (AUC = {:.2f})'.format(labels[i], roc_auc[i]))

  # 计算微观平均ROC曲线和AUC
  fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_scores.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
  plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='Micro-average ROC curve (AUC = {:.2f})'.format(roc_auc["micro"]))
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Multi-class ROC with Micro-average')
  plt.legend(loc='lower right', fontsize=12)
  plt.grid(True)

  # 生成文件名并保存图像
  timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  filename = f'roc_curve_{timestamp}.png'
  if not os.path.exists(directory):
    os.makedirs(directory)
  full_path = os.path.join(directory, filename)
  plt.savefig(full_path, dpi=600)
  plt.close()
  print(f"ROC曲线图像已保存为: {full_path}")

# 使用示例:
# plot_roc_curve_multiclass2(y_test, y_scores, '/path/to/your/folder')

##########################################################################################################
#查全率查准率曲线
##########################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# 定义一个函数用于绘制多类别的查全率-查准率曲线
def plot_precision_recall_curve_multiclass(y_test, y_scores):
  num_classes = 4
  y_test_bin = label_binarize(y_test, classes=range(num_classes))

  # 确保预测得分的维度是正确的
  if y_scores.ndim == 1:
    y_scores = np.reshape(y_scores, (-1, num_classes))

  plt.figure(figsize=(8, 6))
  colors = [(232/255, 56/255, 71/255), (221/255, 108/255, 76/255), 
        (69/255, 123/255, 157/255), (29/255, 53/255, 87/255)]
  linestyles = ['-', '--', '-.', ':']
  labels = ['Precision-Recall curve of AF', 'Precision-Recall curve of normal', 
        'Precision-Recall curve of other', 'Precision-Recall curve of noise']

  # 计算每个类别的查全率-查准率曲线和平均查准率
  precision = dict()
  recall = dict()
  average_precision = dict()
  for i in range(num_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_scores[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_scores[:, i])
    plt.plot(recall[i], precision[i], color=colors[i], linestyle=linestyles[i], 
         lw=2, label='{} (AP = {:.2f})'.format(labels[i], average_precision[i]))

  # 计算微观平均查全率-查准率曲线和平均查准率
  precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_scores.ravel())
  average_precision["micro"] = average_precision_score(y_test_bin, y_scores, average="micro")
  plt.plot(recall["micro"], precision["micro"], color='deeppink', linestyle=':', 
       linewidth=4, label='Micro-average Precision-Recall curve (AP = {:.2f})'.format(average_precision["micro"]))

  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Multi-class Precision-Recall with Micro-average')
  plt.legend(loc='lower left', fontsize=12)
  plt.grid(True)
  plt.show()
# 测试数据
# y_test = np.array([...])  # 测试集标签
# y_scores = np.array([...])  # 模型预测的得分
# plot_precision_recall_curve_multiclass(y_test, y_scores)
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
import datetime
import os

def plot_precision_recall_curve_multiclasstupian(y_test, y_scores, directory='C:\\Users\\Administrator\\Desktop\\p-final\\tupian'):
  """
  绘制多类别的查全率-查准率曲线，并保存在指定文件夹。

  参数:
  y_test (array): 真实标签数组。
  y_scores (array): 预测得分数组。
  directory (str): 保存图片的目录。
  """
  num_classes = 4
  y_test_bin = label_binarize(y_test, classes=range(num_classes))
  if y_scores.ndim == 1:
    y_scores = np.reshape(y_scores, (-1, num_classes))

  plt.figure(figsize=(8, 6))
  colors = [(232/255, 56/255, 71/255), (221/255, 108/255, 76/255), 
        (69/255, 123/255, 157/255), (29/255, 53/255, 87/255)]
  linestyles = ['-', '--', '-.', ':']
  labels = ['Precision-Recall curve of AF', 'Precision-Recall curve of normal', 
        'Precision-Recall curve of other', 'Precision-Recall curve of noise']

  # 计算每个类别的查全率-查准率曲线和平均查准率
  precision = dict()
  recall = dict()
  average_precision = dict()
  for i in range(num_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_scores[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_scores[:, i])
    plt.plot(recall[i], precision[i], color=colors[i], linestyle=linestyles[i], 
         lw=2, label='{} (AP = {:.2f})'.format(labels[i], average_precision[i]))

  # 计算微观平均查全率-查准率曲线和平均查准率
  precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_scores.ravel())
  average_precision["micro"] = average_precision_score(y_test_bin, y_scores, average="micro")
  plt.plot(recall["micro"], precision["micro"], color='deeppink', linestyle=':', 
       linewidth=4, label='Micro-average Precision-Recall curve (AP = {:.2f})'.format(average_precision["micro"]))

  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Multi-class Precision-Recall with Micro-average')
  plt.legend(loc='lower left', fontsize=12)
  plt.grid(True)

  # 生成文件名并保存图像
  timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  filename = f'precision_recall_curve_{timestamp}.png'
  if not os.path.exists(directory):
    os.makedirs(directory)
  full_path = os.path.join(directory, filename)
  plt.savefig(full_path, dpi=600)
  plt.close()
  print(f"查全率-查准率曲线图像已保存为: {full_path}")

# 使用示例:
# plot_precision_recall_curve_multiclass(y_test, y_scores, '/path/to/your/folder')

##########################################################################################################
#频域能量图
##########################################################################################################
import numpy as np
import pywt
import matplotlib.pyplot as plt

def plot_magnitude_scalogram(ecg_signal):
  """
  绘制ECG信号的magnitude scalogram。
  
  参数:
  ecg_signal (np.array): ECG信号数组。s
  """
  # 进行连续小波变换
  scales = np.arange(1, 128)  # 设置小波变换的尺度
  coefficients, frequencies = pywt.cwt(ecg_signal, scales, 'morl')
  plt.imshow(np.abs(coefficients), extent=[0, len(ecg_signal), 1, 128], cmap='jet', aspect='auto')
  plt.colorbar(label='Magnitude')
  plt.xlabel('Time')
  plt.ylabel('Scale')
  plt.title('Magnitude Scalogram of ECG Signal')
  plt.show()
##########################################################################################################
#g-mean
##########################################################################################################
from sklearn.metrics import confusion_matrix
import numpy as np

def calculate_g_mean(y_true, y_pred_prob, num_classes=4):
  """
  Calculate the G-mean for a multi-class classification problem.
  :param y_true: The true class labels as one-hot encoded.
  :param y_pred_prob: The predicted class probabilities.
  :param num_classes: The number of classes.
  :return: The G-mean value.
  """
  # 转换 y_true 从 one-hot 编码到类别标签
  y_true_labels = np.argmax(y_true, axis=1)
  # 转换 y_pred_prob 从概率到类别标签
  y_pred_labels = np.argmax(y_pred_prob, axis=1)
  # 计算混淆矩阵
  cm = confusion_matrix(y_true_labels, y_pred_labels)
  # 计算每个类别的真正例率（True Positive Rate, TPR）
  tpr_list = []
  for i in range(num_classes):
    tpr = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
    tpr_list.append(tpr) 
  # 计算 G-mean
  g_mean = np.prod(tpr_list)**(1/num_classes)
  return g_mean
# 示例使用
# y_test_classes = np.argmax(y_test, axis=1)
# y_pred_classes = np.argmax(model.predict(X_test), axis=-1)
# g_mean_value = calculate_g_mean(y_test_classes, y_pred_classes, num_classes=4)
# print("G-mean:", g_mean_value)
##########################################################################################################
#g-mean
##########################################################################################################
import numpy as np
import pywt
import random

def denoise2_iterative(data, iterations=5):
  """
  对信号进行迭代去噪处理。
  
  参数:
  data (np.array): 输入的信号。
  iterations (int): 迭代次数。
  
  返回:
  np.array: 去噪后的信号。
  """
  for _ in range(iterations):
    # 对信号进行小波分解
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    threshold = (np.median(np.abs(coeffs[-1])) / 0.6745) * (np.sqrt(2 * np.log(len(coeffs[-1]))))

    # 将高频和部分低频层级置零
    coeffs[0].fill(0)  # cA9
    for i in range(1, 4):
      coeffs[i].fill(0)  # cD9, cD8, cD7

    # 对其余层级应用阈值处理
    for i in range(4, len(coeffs)):
      coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 重构信号
    data = pywt.waverec(coeffs=coeffs, wavelet='db5')

  return data

# 示例用法
# denoised_data = denoise2_iterative(your_data_array, iterations=5)
# 其中 your_data_array 是你要处理的原始信号数组。
##########################################################################################################
#数据批量增强
##########################################################################################################
def soft_thresholding(data, threshold):
  """
  对数据应用软阈值处理。
  
  参数:
  data (np.array): 输入数据。
  threshold (float): 阈值。
  
  返回:
  np.array: 软阈值处理后的数据。
  """
  return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)
def denoise2_iterative2(data, iterations=5):
  """
  对信号进行迭代去噪处理，随机选择高频和低频进行软阈值处理。
  
  参数:
  data (np.array): 输入的信号。
  iterations (int): 迭代次数。
  
  返回:
  np.array: 去噪后的信号。
  """
  for _ in range(iterations):
    # 对信号进行边缘扩展
    data_padded = np.pad(data, (0, 2**9), mode='symmetric')

    # 对扩展后的信号进行小波分解
    coeffs = pywt.wavedec(data=data_padded, wavelet='db5', level=9)

    # 随机选择系数进行软阈值处理
    selected_coeffs = random.sample(range(1, len(coeffs)), random.randint(1, len(coeffs) - 1))
    threshold = random.uniform(0.2, 0.25)

    for i in selected_coeffs:
      coeffs[i] = soft_thresholding(coeffs[i], threshold)

    # 重构信号
    data = pywt.waverec(coeffs=coeffs, wavelet='db5')[:len(data)]

  return data

# 示例用法
# denoised_data = denoise2_iterative2(your_data_array, iterations=5)
# 其中 your_data_array 是你要处理的原始信号数组。
import numpy as np
import pywt
import random
def soft_thresholding(data, threshold):
  return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)

def denoise2_iterative3(datas, iterations=5):
  denoised_data = []
  for data in datas:
    for _ in range(iterations):
      # 对信号进行边缘扩展
      data_padded = np.pad(data, (0, 2**9), mode='symmetric')
      # 对扩展后的信号进行小波分解
      coeffs = pywt.wavedec(data=data_padded, wavelet='db5', level=9)
      # 随机选择系数进行软阈值处理
      selected_coeffs = random.sample(range(1, len(coeffs)), random.randint(1, len(coeffs) - 1))
      threshold = random.uniform(0.2, 0.25)
      for i in selected_coeffs:
        coeffs[i] = soft_thresholding(coeffs[i], threshold)
      # 重构信号
      data = pywt.waverec(coeffs=coeffs, wavelet='db5')[:len(data)]
    denoised_data.append(data)

  return denoised_data

##########################################################################################################
#focal损失函数
##########################################################################################################
import tensorflow as tf
def focal_loss(gamma=2., alpha=.25):
  def focal_loss_fixed(y_true, y_pred):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1)) - tf.reduce_sum((1-alpha) * tf.pow(pt_0, gamma) * tf.log(1. - pt_0))
  return focal_loss_fixed

from tensorflow.keras.callbacks import Callback

class AdjustLearningRateCallback(Callback):
    def __init__(self, factor, patience, min_lr=0, monitor='val_loss'):
        super().__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.monitor = monitor
        self.wait = 0
        self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(self.min_lr, self.model.optimizer.lr * self.factor)
                self.model.optimizer.lr = new_lr
                self.wait = 0
                print(f'Reduced learning rate to {new_lr}.')
