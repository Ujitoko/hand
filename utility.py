import matplotlib.pyplot as plt
import os

from PIL import Image
import numpy as np
from enum import Enum
import librosa, librosa.display
from scipy.signal import hamming
import scipy
import cv2
import tensorflow as tf

def save_metrics(acc, model_name, num_iteration):
    # make directory if there is not
    dir_path = "metrics_" + model_name
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    # save metrics
    plt.figure(figsize=(10,8))
    plt.plot(acc["train"], label="train accuracy", color="blue")
    plt.plot(acc["test"], "--", label="test accuracy", color="darkcyan")
    plt.legend()
    plt.savefig(os.path.join(dir_path, "acc_" + str(num_iteration) + ".png"))
    plt.close()


# plot images
def save_imgs(model_name, images, plot_dim=(3,4), size=(12,8), name=None):
    # make directory if there is not
    path = "generated_figures_" + model_name
    if not os.path.isdir(path):
        os.makedirs(path)

    numImgs = images.shape[0]
    #eachNumImgs = images.shape[0]/2

    #num_examples = plot_dim[0]*plot_dim[1]
    num_examples = numImgs
    #size = (eachNumImgs*4, 2*4)
    #plot_dim = (2, eachNumImgs)

    fig = plt.figure(figsize=size)
    for i in range(num_examples):
        plt.subplot(plot_dim[0], plot_dim[1], i+1)
        img = images[3*(i//3) + i%3, :]
        if i%3 == 0:
            img = img.reshape((256, 256, 3))
        else:
            img = img.reshape((256,256,3))
        plt.tight_layout()
        plt.imshow(img)
        plt.axis("off")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(os.path.join(path, str(name) + ".png"))
    plt.close()




class hand_dataset:
    def __init__(self):

        #self.org_path = "/home/ujitoko/works/hand/Hands"
        self.org_path = "./Hands/"

        g_imgs_training_d_txt = "./1/g_imgs_training_d.txt"
        g_training_d_txt = "./1/g_training_d.txt"

        g_imgs_testing_d_txt = "./1/g_imgs_testing_d.txt"
        g_testing_d_txt = "./1/g_testing_d.txt"

        f = open(g_imgs_training_d_txt, "r")
        self.train_image_files = f.readlines()
        self.train_image_files = self.replace_line_feeds(self.train_image_files)

        f = open(g_imgs_testing_d_txt, "r")
        self.test_image_files = f.readlines()
        self.test_image_files = self.replace_line_feeds(self.test_image_files)

        f = open(g_training_d_txt, "r")
        self.train_labels = f.readlines()
        self.train_labels = self.replace_line_feeds(self.train_labels)

        f = open(g_testing_d_txt, "r")
        self.test_labels = f.readlines()
        self.test_labels = self.replace_line_feeds(self.test_labels)

        self.resized_height = 150
        self.resized_width = 200

    def replace_line_feeds(self, str_list):
        for i in range(len(str_list)):
            str_list[i] = str_list[i].replace("\n", "")
        return str_list

    def extract_random_minibatch(self, batch_size, img_size, mixup=False):
        if mixup==True:
            batch_size = batch_size*2

        rand_index = np.random.randint(0, len(self.train_image_files)-1, size=batch_size)

        imgs_np = np.empty((0, self.resized_height, self.resized_width, 3), np.float32)
        label_np = np.empty((0, 2), np.float32)

        for i in rand_index:
            img = cv2.imread(os.path.join(self.org_path, self.train_image_files[i]))
            img = cv2.resize(img, (self.resized_width, self.resized_height))
            img_ = img[np.newaxis, :]
            imgs_np = np.append(imgs_np, img_, axis=0)

            # label
            sex = np.zeros([1, 2])
            if self.train_labels[i] == "male":
                sex[:, 0] = 1
            else:
                sex[:, 1] = 1
            label_np = np.append(label_np, sex, axis=0)

        if mixup==True:
            alpha = 0.2
            l = np.random.beta(alpha, alpha, batch_size//2)
            X_l = l.reshape(batch_size//2, 1, 1, 1)
            y_l = l.reshape(batch_size//2, 1)

            imgs_np = imgs_np[:batch_size//2] * X_l + imgs_np[batch_size//2:] * (1-X_l)
            label_np = label_np[:batch_size//2] * y_l + label_np[batch_size//2:] * (1-y_l)

        return imgs_np, label_np

    def extract_next_minibatch(self, num, size, batch_index, test=True):
        if test==False:
            if (batch_index+1)*num > len(self.train_image_files):
                return 0, 0, False;
            index = np.arange(batch_index*num, (batch_index+1)*num)

            imgs_np = np.empty((0, self.resized_height, self.resized_width, 3), np.float32)
            label_np = np.empty((0, 2), np.float32)

            for i in index:
                img = cv2.imread(os.path.join(self.org_path, self.train_image_files[i]))
                img = cv2.resize(img, (self.resized_width, self.resized_height))
                img_ = img[np.newaxis, :]
                imgs_np = np.append(imgs_np, img_, axis=0)

                # label
                sex = np.zeros([1, 2])
                if self.train_labels[i] == "male":
                    sex[:, 0] = 1
                else:
                    sex[:, 1] = 1
                label_np = np.append(label_np, sex, axis=0)

            imgs_np = imgs_np.reshape((-1, self.resized_height, self.resized_width, 3))
            return imgs_np, label_np, True

        else:
            if (batch_index+1)*num > len(self.test_image_files):
                return 0, 0, False;
            index = np.arange(batch_index*num, (batch_index+1)*num)

            imgs_np = np.empty((0, self.resized_height, self.resized_width, 3), np.float32)
            label_np = np.empty((0, 2), np.float32)

            for i in index:
                img = cv2.imread(os.path.join(self.org_path, self.test_image_files[i]))
                img = cv2.resize(img, (self.resized_width, self.resized_height))
                img_ = img[np.newaxis, :]
                imgs_np = np.append(imgs_np, img_, axis=0)

                # label
                sex = np.zeros([1, 2])
                if self.test_labels[i] == "male":
                    sex[:, 0] = 1
                else:
                    sex[:, 1] = 1
                label_np = np.append(label_np, sex, axis=0)

            imgs_np = imgs_np.reshape((-1, self.height, self.width, 3))
            return imgs_np, label_np, True


def img_augment(input_img, img_size, test=True, re=False):
    import random


    rand_width = tf.random_uniform([], 130, 150, dtype=tf.int32)

    #print(rand_width.shape)
    img = input_img

    if test != True:
        img = tf.image.random_brightness(img, max_delta=1.2)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.0)

    cropped_img = tf.random_crop(img, [rand_width, rand_width, 3])
    if test != True:
        cropped_img = tf.image.random_flip_left_right(cropped_img)
        cropped_img = tf.image.random_flip_up_down(cropped_img)

    random_rotate = tf.random_uniform([], 0, 360, dtype=tf.float32)

    #rotate_angle = random.randint(0, 360)
    img = cropped_img
    if test != True:
        img = tf.contrib.image.rotate(img, random_rotate)

    img = tf.image.resize_images(img, [img_size, img_size])

    if test != True:
        if re == True:
            p = np.random.uniform(0, 1)
            if p > 0.5:

                s = np.random.uniform(0.02, 0.4) * img_size * img_size
                r = np.random.uniform(-np.log(3.0), np.log(3.0))
                r = np.exp(r)
                w = int(np.sqrt(s / r))
                w = np.min((img_size-1, w))
                h = int(np.sqrt(s * r))
                h = np.min((img_size-1, h))
                #print(w.shape)
                #print(h.shape)
                #print(w)
                #print(h)

                left = np.random.randint(0, img_size - w)
                top = np.random.randint(0, img_size -h)
                c = np.random.randint(0, 255)

                img_0 = np.ones([img_size, img_size, 3])
                img_0[top:top+h, left:left+w, :] = 0

                img_c = np.zeros([img_size, img_size, 3])
                img_c[top:top+h, left:left+w, :] = c

                res_img = tf.multiply(img, img_0)
                img = tf.add(res_img, img_c)

    return img


def imgs_augment(batch_size, input_imgs, test=True, re=False):

    aug_imgs_list = []
    img_size = 128
    for i in range(batch_size):
        augmented_img = img_augment(input_imgs[i], img_size, test, re)
        aug_imgs_list.append(augmented_img)

    return tf.stack(aug_imgs_list, axis=0)
