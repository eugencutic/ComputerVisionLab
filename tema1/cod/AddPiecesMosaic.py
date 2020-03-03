import random

from cod.Parameters import *
import numpy as np
import pdb
import timeit
import cv2 as cv


def add_pieces_grid(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                index = np.random.randint(low=0, high=N, size=1)
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    elif params.criterion == 'distantaCuloareMedie':
        small_images_avg_colors = [img.mean(axis=(0, 1)) for img in params.small_images]
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                patch_avg_color = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W, :].mean(axis=(0, 1))
                distances = [np.linalg.norm(patch_avg_color - small_img_color) for small_img_color in
                             small_images_avg_colors]
                if params.vecini_dif:
                    index = 0
                    min_distance = np.inf
                    found_valid = False
                    for idx in range(len(distances)):
                        small_image = params.small_images[idx]
                        valid = True
                        if i > 0:
                            patch_above = img_mosaic[(i - 1) * H: i * H, j * W: (j + 1) * W, :]
                            if np.array_equal(small_image, patch_above):
                                valid = False
                        if i < params.num_pieces_vertical - 1:
                            patch_under = img_mosaic[(i + 2) * H: (i + 1) * H, j * W: (j + 1) * W, :]
                            if np.array_equal(small_image, patch_under):
                                valid = False
                        if j > 0:
                            patch_left = img_mosaic[i * H: (i + 1) * H, (j - 1) * W: j * W, :]
                            if np.array_equal(small_image, patch_left):
                                valid = False
                        if j < params.num_pieces_horizontal - 1:
                            patch_right = img_mosaic[i * H: (i + 1) * H, (j + 2) * W: (j + 1) * W, :]
                            if np.array_equal(small_image, patch_right):
                                valid = False
                        if valid:
                            if distances[idx] < min_distance:
                                min_distance = distances[idx]
                                index = idx
                            found_valid = True
                    if not found_valid:
                        index = distances.index(min(distances))
                else:
                    index = distances.index(min(distances))

                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_random(params: Parameters):
    img_mosaic = np.full(params.image_resized.shape, -1)
    N, H, W, C = params.small_images.shape

    small_images_avg_colors = [img.mean(axis=0) for img in params.small_images]
    num_pieces = 0
    seen = set()
    rand_line = random.randint(0, img_mosaic.shape[0] - H)
    rand_col = random.randint(0, img_mosaic.shape[1] - W)
    not_covered = (img_mosaic == -1).sum()
    while not_covered > 0:
        seen.add((rand_line, rand_col))
        patch_avg_color = params.image_resized[rand_line: rand_line + H, rand_col: rand_col + W, :] \
            .mean(axis=0)
        distances = [np.linalg.norm(patch_avg_color - small_img_color) for small_img_color in
                     small_images_avg_colors]
        index = distances.index(min(distances))
        not_covered -= (img_mosaic[rand_line: rand_line + H, rand_col: rand_col + W, :] == -1).sum()
        img_mosaic[rand_line: rand_line + H, rand_col: rand_col + W, :] = params.small_images[index]
        num_pieces += 1
        if num_pieces % 100 == 0:
            print('Num pieces: ' + str(num_pieces))
            print('Not covered: ' + str(not_covered))
        rand_line = random.randint(0, img_mosaic.shape[0] - H)
        rand_col = random.randint(0, img_mosaic.shape[1] - W)
        while (rand_line, rand_col) in seen:
            rand_line = random.randint(0, img_mosaic.shape[0] - H)
            rand_col = random.randint(0, img_mosaic.shape[1] - W)
    print('Num pieces: ' + str(num_pieces))

    return img_mosaic.astype(np.uint8)


def add_pieces_hexagon(params: Parameters):
    mask = np.full(params.small_images[0].shape, 0)
    black = np.full(3, 1)

    for i in range(14):
        mask[i, (13 - i): (26 + i), :] = black
    for i in range(14, 28):
        mask[i, (26 + i) - 40: (13 - i) + 40, :] = black


    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    small_images_avg_colors = [img.mean(axis=(0, 1)) for img in params.small_images]

    lin = 1
    for y in range(14, params.image_resized.shape[0] - H, H):
        col = 0
        for x in range(0, params.image_resized.shape[1] - W, W + W // 3):
            patch = params.image_resized[y: y + H, x: x + W, :]
            patch_avg_color = patch.mean(axis=(0, 1))
            distances = [np.linalg.norm(patch_avg_color - small_img_color) for small_img_color in
                         small_images_avg_colors]
            index = distances.index(min(distances))
            img_mosaic[y: y + H, x: x + W, :] = \
                params.small_images[index] * mask + img_mosaic[y: y + H, x: x + W, :] * (1 - mask)
            print('Building mosaic first half %.2f%%' %
                  (100 * (y / H * params.num_pieces_horizontal + x / (W + W // 3) + 1) / num_pieces))
            col += 2
        lin += 2
    print('Building mosaic first half 100%')
    lin = 0
    for y in range(0, params.image_resized.shape[0] - H, H):
        col = 1
        for x in range( int(2/3 * W), params.image_resized.shape[1] - W, W + W // 3):
            patch = params.image_resized[y: y + H, x: x + W, :]
            patch_avg_color = patch.mean(axis=(0, 1))
            distances = [np.linalg.norm(patch_avg_color - small_img_color) for small_img_color in
                         small_images_avg_colors]
            index = distances.index(min(distances))
            img_mosaic[y: y + H, x: x + W, :] = \
                params.small_images[index] * mask + img_mosaic[y: y + H, x: x + W, :] * (1 - mask)
            print('Building mosaic second half %.2f%%' %
                  (100 * (y / H * params.num_pieces_horizontal + x / (W + W // 3) + 1) / num_pieces))
            col += 2
        lin += 2
    print('Building mosaic second half 100%')
    # vecinii sunt la
    # i - 2, j
    # i - 1, j - 1
    # i - 1, j + 1
    # (i + 1, j - 1)
    # (i + 1, j + 1)
    # (i + 2, j)

    return img_mosaic[H // 2: -H, W // 3: -(W + W // 3), :]


def overlap(left_patch, hexagon):
    hex_copy = np.copy(hexagon)
    left_patch = left_patch[:, 20:, :]
    hex_copy_left = hex_copy[:, 0:20, :]
    hex_copy_left[hex_copy_left == 0] = left_patch[hex_copy_left == 0]
    hex_copy[:, :20, :] = hex_copy_left[:, :, :]

    return hex_copy
