import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pdb

from cod.AddPiecesMosaic import *
from cod.Parameters import *


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def construct_image(data):
    image = np.zeros((32, 32, 3), np.uint8)
    current_data_idx = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j][0] = data[current_data_idx + 2048]
            image[i][j][1] = data[current_data_idx + 1024]
            image[i][j][2] = data[current_data_idx]
            current_data_idx += 1
    return image


def load_pieces(params: Parameters):
    # citeste toate cele N piese folosite la mozaic din directorul corespunzator
    # toate cele N imagini au aceeasi dimensiune H x W x C, unde:
    # H = inaltime, W = latime, C = nr canale (C=1  gri, C=3 color)
    # functia intoarce pieseMozaic = matrice H x W x C x N in params
    # pieseMoziac(:,:,:,i) reprezinta piesa numarul i
    images = []
    if params.small_images_dir == 'cifar':
        metadict = unpickle('./../data/cifar-10-batches-py/batches.meta')
        datadict = unpickle('./../data/cifar-10-batches-py/data_batch_1')
        labels_dict = {}
        for idx, label in enumerate(metadict['label_names']):
            labels_dict[label] = idx

        for idx, data in enumerate(datadict['data']):
            if datadict['labels'][idx] == labels_dict['dog']:
                image = construct_image(data)
                images.append(image)
        images = np.asarray(images)
    else:
        for file in os.listdir(params.small_images_dir):
            img = cv.imread(os.path.join(params.small_images_dir, file))
            images.append(img)

    images = np.asarray(images)
    # citeste imaginile din director

    if params.show_small_images:
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, i * 10 + j + 1)
                # OpenCV reads images in BGR format, matplotlib reads images in RBG format
                im = images[i * 10 + j].copy()
                # BGR to RGB, swap the channels
                im = im[:, :, [2, 1, 0]]
                plt.imshow(im)
        plt.show()

    params.small_images = images


def compute_dimensions(params: Parameters):
    # calculeaza dimensiunile mozaicului
    # obtine si imaginea de referinta redimensionata avand aceleasi dimensiuni
    # ca mozaicul

    # completati codul
    # calculeaza automat numarul de piese pe verticala
    image = np.asarray(params.image)
    h = image.shape[0]
    w = image.shape[1]

    params.num_pieces_vertical = (params.num_pieces_horizontal * h) // w

    # redimensioneaza imaginea
    new_h = params.num_pieces_vertical * params.small_images.shape[1]
    new_w = params.num_pieces_horizontal * params.small_images.shape[2]
    params.image_resized = cv.resize(params.image, (new_w, new_h))

def build_mosaic(params: Parameters):
    # incarcam imaginile din care vom forma mozaicul
    load_pieces(params)
    # calculeaza dimensiunea mozaicului
    compute_dimensions(params)

    img_mosaic = None
    if params.layout == 'caroiaj':
        if params.hexagon is True:
            img_mosaic = add_pieces_hexagon(params)
        else:
            img_mosaic = add_pieces_grid(params)
    elif params.layout == 'aleator':
        img_mosaic = add_pieces_random(params)
    else:
        print('Wrong option!')
        exit(-1)

    return img_mosaic
