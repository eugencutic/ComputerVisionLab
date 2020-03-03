import os
import random
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# img = cv.resize(cv.cvtColor(cv.imread('data\\colectiiImagini\\set1\\im01.jpg'), cv.COLOR_BGR2GRAY), (100, 100))

### (a)
# x = np.sort(img.flatten())
# plt.plot(x)
# plt.show()

### (b)
# dreapta_jos = img[50:, 50:]
# cv.imshow('window', dreapta_jos)
# cv.waitKey(0)
# cv.destroyAllWindows()

### (c)
# t = np.median(img)

### (d)
# B = np.zeros(img.shape)
# B[img >= t] = 255
# cv.imshow('window', B)
# cv.waitKey(0)
# cv.destroyAllWindows()

### (e)
# C = img.copy()
# C = C - img.mean()
# C[C < 0] = 0
# cv.imshow('window', C)
# cv.waitKey(0)
# cv.destroyAllWindows()

### (f)
# min_val = np.min(img)
# lin, col = np.where(img == min_val)


### 1.7
def colectie_imagini(dir):
    files = os.listdir(dir)
    images = []
    gray_scale_images = []
    for file in files:
        img = cv.imread(os.path.join(dir, file))
        images.append(img)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_scale_images.append(gray_img)

    images = np.array(images) # images.shape NxHxWx3
    mean_image = np.mean(images, axis=0)
    gray_scale_images = np.array(gray_scale_images)
    mean_intensity = gray_scale_images.mean(axis=0)
    std_image = np.std(gray_scale_images, axis=0)
    cv.imshow('win1', np.uint8(mean_image))
    cv.imshow('win2', np.uint8(mean_intensity))
    cv.imshow('win3', np.uint8(std_image))
    cv.waitKey(0)
    cv.destroyAllWindows()


# colectie_imagini('data\\colectiiImagini\\set1')

def get_closest_patch(patch_to_match, patches):
    pass


img = cv.imread('data\\butterfly.jpeg')
sub_images = []
for i in range(500):
    random_line = random.randint(0, img.shape[0] - 20)
    random_col = random.randint(0, img.shape[1] - 20)
    sub_img = img[random_line:random_line + 20, random_col:random_col + 20]
    sub_images.append(sub_img)


