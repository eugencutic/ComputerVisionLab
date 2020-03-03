import sys
import cv2 as cv
import numpy as np

from cod.Parameters import *
from cod.SelectPath import *

import pdb


def compute_energy(img):
    """
    calculeaza energia la fiecare pixel pe baza gradientului
    :param img: imaginea initiala
    :return:E - energia
    """
    # urmati urmatorii pasi:
    # 1. transformati imagine in grayscale
    # 2. folositi filtru sobel pentru a calcula gradientul in directia X si Y
    # 3. calculati magnitudinea imaginii

    # BACK TO UINT8
    img = np.uint8(img)
    img_gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(img_gray_scale, ddepth=cv.CV_16S, dx=1, dy=0, borderType=cv.BORDER_CONSTANT)
    grad_y = cv.Sobel(img_gray_scale, ddepth=cv.CV_16S, dx=0, dy=1, borderType=cv.BORDER_CONSTANT)

    E = abs(grad_x) + abs(grad_y)

    return E


def show_path(img, path, color):
    new_image = img.copy()
    for row, col in path:
        new_image[row, col] = color

    cv.imshow('path', np.uint8(new_image))
    cv.waitKey(100)


def delete_path(img, pathway):
    """
     elimina drumul vertical din imagine
    :param img: imaginea initiala
    :pathway - drumul vertical
    return: updated_img - imaginea initiala din care s-a eliminat drumul vertical
    """
    updated_img = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]), np.uint8)
    for i in range(img.shape[0]):
        col = pathway[i][1]
        # copiem partea din stanga
        updated_img[i, :col] = img[i, :col].copy()
        # copiem partea din dreapta
        updated_img[i, col:] = img[i, col + 1:].copy()
    return updated_img


def decrease_width(params: Parameters, num_pixels):
    img = params.image.copy() # copiaza imaginea originala
    for i in range(num_pixels):
        print('Eliminam drumul vertical numarul %i dintr-un total de %d.' % (i + 1, num_pixels))

        # calculeaza energia dupa ecuatia (1) din articol
        E = compute_energy(img)
        # pentru stergerea de obiect:
        # E[y_min:y_max, x_min:x_max] = -1000 => fortam path ul sa o ia peste obiect
        # x_max -= 1 pentru ca am sters din obiect
        # Pentru cazul in care rotim imaginea facem initial E_inf = np.zeros(E.shape)
        # E_inf[y_min:y_max, x_min:x_max] = -1000
        # il rotesti si pe E_inf si apoi E += E_inf
        # si y_max -= 1
        path = select_path(E, params.method_select_path)
        if params.show_path:
            show_path(img, path, params.color_path)
        img = delete_path(img, path)

    cv.destroyAllWindows()
    return img


def delete_object(params: Parameters, x_min, x_max, y_min, y_max):
    img = params.image.copy()  # copiaza imaginea originala
    num_pixels = x_max - x_min + 1
    for i in range(num_pixels):
        print('Eliminam drumul vertical numarul %i dintr-un total de %d.' % (i + 1, num_pixels))

        # calculeaza energia dupa ecuatia (1) din articol
        E = compute_energy(img)
        # pentru stergerea de obiect:
        E[y_min:y_max, x_min:x_max] = -1000 # fortam path ul sa o ia peste obiect
        x_max -= 1 # pentru ca am sters din obiect
        path = select_path(E, params.method_select_path)
        if params.show_path:
            show_path(img, path, params.color_path)
        img = delete_path(img, path)

    cv.destroyAllWindows()
    return img


def resize_image(params: Parameters):

    if params.resize_option == 'micsoreazaLatime':
        # redimensioneaza imaginea pe latime
        resized_image = decrease_width(params, params.num_pixels_width)
        return resized_image
    elif params.resize_option == 'micsoreazaInaltime':
        # redimensioneaza imaginea pe inaltime
        # rotim imaginea si apelam functia apoi rotim imaginea inapoi
        # np.rot(img, k=3) si inapoi np.rot(img, k=1)
        params.image = np.rot90(params.image, k=1)
        resized_image = decrease_width(params, params.num_pixel_height)
        params.image = np.rot90(params.image, k=-1)
        return np.rot90(resized_image, k=-1)
    elif params.resize_option == 'maresteLatime':
        # mareste imaginea pe latime
        img_copy = params.image.copy()
        paths = []
        for i in range(50):
            E = compute_energy(img_copy)
            path = select_path(E, method=params.method_select_path)
            delete_path(img_copy, path)
            paths.append(path)

        img = params.image.copy().astype(np.float)
        for i in range(50):
            img2 = np.zeros((img.shape[0], img.shape[1] + 1, img.shape[2])).astype(np.float)
            img2[:, :-1, :] = img
            for j in range(len(paths[i])):
                c = paths[i][j][1]
                img2[i, 0:c] = img[i, 0:c]
                img2[i, c + 2:] = img[i, c + 1:]
                img2[i, c+1] = (img[i, c] + img[i, c+1]) / 2
                for k in range(i, 50):
                    if paths[k][j][1] > c:
                        paths[k][j][1] += 1
            img = img2
        return np.uint8(img)
    elif params.resize_option == 'maresteInaltime':
        # mareste imaginea pe inaltime
        pass
    elif params.resize_option == 'amplificaContinut':
        # amplifica continutul imaginii
        # mai intai marim apoi apelam functia
        factor = 1.5
        resized_image = cv.resize(params.image, (0, 0), fx=factor, fy=factor)
        pixel_w = resized_image.shape[1] - params.image.shape[1]
        pixel_h = resized_image.shape[0] - params.image.shape[0]
        img = params.image.copy()
        params.image = resized_image
        resized_image = decrease_width(params, pixel_w)
        resized_image = np.rot90(resized_image, k=1)
        params.image = resized_image
        resized_image = decrease_width(params, pixel_h)
        resized_image = np.rot90(resized_image, k=-1)
        params.image = img.copy()
        return resized_image
    elif params.resize_option == 'eliminaObiect':
        # elimina obiect din imagine
        params.image = np.uint8(params.image)
        x0, y0, w, h = cv.selectROI('win', params.image)
        params.image = params.image.astype(np.float)
        x_min = x0
        x_max = x0 + w
        y_min = y0
        y_max = y0 + h

        new_img = delete_object(params, x_min, x_max, y_min, y_max)
        return new_img
    else:
        print('The option is not valid!')
        sys.exit(-1)
