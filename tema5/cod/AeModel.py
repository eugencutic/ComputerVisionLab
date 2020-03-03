import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
import keras.models
import cv2 as cv
import os
# import SGD and Adam optimizers
from keras.optimizers import SGD, Adam, RMSprop
from cod.DataSet import *


class AeModel:

    def __init__(self, data_set: DataSet):
        self.data_set = data_set
        self.num_epochs = 30
        self.batch_size = 8
        self.learning_rate = 10 ** -4
        self.model = None
        self.checkpoint_dir = './checkpoints_%s' % self.data_set.scene_name

    def define_the_model(self):
        # defineste autoencoderul
        self.model = keras.models.Sequential([
            layers.InputLayer(input_shape=(self.data_set.network_input_size[0], self.data_set.network_input_size[1], 1)),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(filters=2, kernel_size=(3, 3), activation='tanh', strides=(1, 1), padding='same'),
        ])
        # afiseaza arhitectura modelului
        self.model.summary()

    def compile_the_model(self):
        # compilam modelul
        # defineste optimizatorul
        optimizer = Adam(lr=self.learning_rate)
        # apeleaza functia 'compile' cu parametrii corespunzatori.
        self.model.compile(optimizer=optimizer, loss='mse')

    def train_the_model(self):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # definim callback-ul pentru checkpoint
        checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_dir + '/model.{epoch:05d}.hdf5')
        # apelam metoda 'fit' cu parametrii corespunzatori.
        self.model.fit(self.data_set.input_training_images, self.data_set.ground_truth_training_images,
                       epochs=self.num_epochs, batch_size=self.batch_size, callbacks=[checkpoint_callback])

    def evaluate_the_model(self):
        best_epoch = self.num_epochs  # puteti incerca si cu alta epoca de exemplu cu prima epoca,
                                      # sa vedeti diferenta dintre ultima epoca si prima
        # incarcam modelul
        best_model = keras.models.load_model(os.path.join(self.checkpoint_dir, 'model.%05d.hdf5') % best_epoch)
        for i in range(len(self.data_set.input_test_images)):
            # prezicem canalele ab pe baza input_test_images[i]
            predictions = self.model.predict(np.array([self.data_set.input_test_images[i]]))
            ab_channels = predictions[0][:, :, 1:2] * 128
            # reconstruim reprezentarea Lab
            lab_image = np.array((ab_channels.shape[0], ab_channels.shape[1], 3))
            lab_image[:, :, 0] = self.data_set.input_test_images[i]
            lab_image[:, :, 1:2] = ab_channels[:, :, 0:1]
            # convertim din Lab in BGR
            pred_image = cv.cvtColor(lab_image, cv.COLOR_LAB2BGR) * 255
            # convertim imaginea de input din L in 'grayscale'
            input_image = np.uint8(self.data_set.input_test_images[i] / 100 * 255)
            # imaginea ground-truth in format bgr
            gt_image = np.uint8(self.data_set.ground_truth_bgr_test_images[i])
            # pred_image este imaginea prezisa in format BGR.
            concat_images = self.concat_images(input_image, pred_image, gt_image)
            cv.imwrite(os.path.join(self.data_set.dir_output_images, '%d.png' % i), concat_images)

    def concat_images(self, input_image, pred, ground_truth):
        """
        :param input_image: imaginea grayscale (canalul L din reprezentarea Lab).
        :param pred: imaginea prezisa.
        :param ground_truth: imaginea ground-truth.
        :return: concatenarea imaginilor.
        """
        h, w, _ = input_image.shape
        space_btw_images = int(0.2 * h)
        image = np.ones((h, w * 3 + 2 * space_btw_images, 3)) * 255
        # add input_image
        image[:, :w] = input_image
        # add predicted
        offset = w + space_btw_images
        image[:, offset: offset + w] = pred
        # add ground truth
        offset = 2 * (w + space_btw_images)
        image[:, offset: offset + w] = ground_truth
        return np.uint8(image)