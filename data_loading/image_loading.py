import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def load_cifar100_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        images = dict[b'data']
        labels = dict[b'fine_labels']
        return images, labels

# Define the path to your train batch file
train_file_path = r'C:\Users\沐阳\Desktop\数据结构模型\非结构数据\图像数据\cifar-100-python\train'  # 修改为实际文件路径
save_path = r'C:\Users\沐阳\Desktop\数据结构模型\非结构数据\图像数据\图像训练集'  # 设置保存路径

# Load training data
train_images, train_labels = load_cifar100_batch(train_file_path)

def save_images(images, save_path, num_images=20):
    os.makedirs(save_path, exist_ok=True)  # 创建保存路径，如果不存在则创建
    for i in range(num_images):
        img = images[i].reshape(3, 32, 32)  # CIFAR-100 images are in 3x32x32 format
        img = np.transpose(img, (1, 2, 0))  # Change to HxWxC format
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(os.path.join(save_path, f'image_{i+1}.png'))  # 保存为 PNG 格式
        plt.close()  # 关闭当前图像以释放内存

# Save the first 5 images
save_images(train_images, save_path, num_images=20)