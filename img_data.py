import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class RandomBatchImg:
    def __init__(self, left_img_path, right_img_path, seg_path, depth_path, val_ratio=0.1, batch_size=5,
                 img_shape=(128, 448), pattern_list=('0', '1', '', '')):
        if val_ratio != 0:
            self.left_namelist, self.val_left_namelist = self.img_path_array(left_img_path, val_ratio=val_ratio,
                                                                             pattern=pattern_list[0])
            self.right_namelist, self.val_right_namelist = self.img_path_array(right_img_path, val_ratio=val_ratio,
                                                                               pattern=pattern_list[1])
            self.seg_namelist, self.val_seg_namelist = self.img_path_array(seg_path, val_ratio=val_ratio,
                                                                           pattern=pattern_list[2])
            self.depth_namelist, self.val_depth_namelist = self.img_path_array(depth_path, val_ratio=val_ratio,
                                                                               pattern=pattern_list[3])
        else:
            self.left_namelist = self.img_path_array(left_img_path, pattern=pattern_list[0])
            self.right_namelist = self.img_path_array(right_img_path, pattern=pattern_list[1])
            self.seg_namelist = self.img_path_array(seg_path, pattern=pattern_list[2])
            self.depth_namelist = self.img_path_array(depth_path, pattern=pattern_list[3])

        self.img_shape = img_shape
        self.batch_size = batch_size
        self.num_of_train_samples = len(self.left_namelist)
        self.steps_ind = np.arange(self.num_of_train_samples // self.batch_size + 1) * self.batch_size
        self.random_index = np.arange(self.num_of_train_samples)
        np.random.shuffle(self.random_index)

        self.aug_rate = 1 * self.num_of_train_samples
        # augment random degree
        self.ran_aug = np.around(np.random.rand(self.aug_rate), decimals=2)
        # random select augment
        self.ran_sel = np.around(np.random.rand(self.aug_rate), decimals=2)

    def img_path_array(self, path, val_ratio=0., pattern=''):
        if pattern == '':
            path_array = np.array([os.path.join(path, x) for x in os.listdir(path)])
            if val_ratio:
                train_path_array, val_path_array = self.train_val_split(path_array, val_ratio)
                return train_path_array, val_path_array
            else:
                return path_array
        else:
            path_array = np.array(
                [os.path.join(path, x) for x in os.listdir(path) if os.path.splitext(x)[0][-1] == pattern])
            if val_ratio:
                train_path_array, val_path_array = self.train_val_split(path_array, val_ratio)
                return train_path_array, val_path_array
            else:
                return path_array

    @staticmethod
    def imglabel_to_multiclass(image, n_classes=10, img_size=(128, 448)):
        image_ = image[:, :, 0]
        image_ = np.uint8(image_ / image_.max() * n_classes)
        count_index = {}
        index_count = {}
        img_uni = np.unique(image_)
        for i in img_uni:
            ph = np.sum(image_ == i)
            count_index[ph] = i
            index_count[i] = ph

        masks = []
        for i, c in enumerate(count_index):
            image_ = np.uint8(image_ == count_index[c])
            masks.append(image_)
        masks = np.array(masks)
        mask_label = np.reshape(masks, (n_classes, img_size[0] * img_size[1]))
        mask_label = np.transpose(mask_label, (1, 0))
        return mask_label

    @staticmethod
    def img_preprocess(image, normalize=True, prepro=True):
        if prepro:
            # darkness improve
            sd = np.floor(255 / np.log2(255))
            image = (sd * np.log2(image + 0.0001)).astype(np.uint8)

            # gamma enhance contrast
            sg = np.floor(np.power(255, 1.8) / 255)
            image = np.power(image, 1.8) / sg

        # normalize image
        if normalize:
            # image = image - np.mean(image)
            image = np.around((image - np.mean(image)) / np.std(image), decimals=3)

        return image

    @staticmethod
    def imglabel_reg_normalize(image, normalize=False):
        image = image[:, :, 0]
        # normalize image
        if normalize:
            image = np.log2(image + 0.0001)
        return image

    @staticmethod
    def cvload_img(image_path, resize=(64, 112), temp_ratio=1.5):
        image = cv.imread(image_path)
        image = cv.resize(image, (np.int(resize[1] * 2), np.int(resize[0] * 2)), interpolation=cv.INTER_LINEAR)
        image = image[:, 0:resize[1], :]
        image = cv.resize(image, (np.int(resize[1] * temp_ratio), np.int(resize[0] * temp_ratio)),
                          interpolation=cv.INTER_LINEAR)
        image = np.transpose(image, (2, 0, 1))
        return image

    @staticmethod
    def train_val_split(images_path_array, val_ratio=0.1):
        num_of_samples = len(images_path_array)
        val_rat = np.int(val_ratio * num_of_samples)
        val_img_array = np.array(images_path_array[:val_rat])
        train_img_array = np.array(images_path_array[val_rat:])
        return train_img_array, val_img_array

    @staticmethod
    def augment(input_img, ran_degree, ran_aug,
                angle=30,
                img_size=(64, 112),
                temp_ratio=1.5):
        # width_shift_range=0.5
        # height_shift_range=0.2
        onehalf_size = (np.int(img_size[0] * temp_ratio), np.int(img_size[1] * temp_ratio))
        wshift_range = onehalf_size[1] - img_size[1]
        hshift_range = onehalf_size[0] - img_size[0]
        np_ran = ran_degree * 2 - 1
        # half_size = (img_size[1]//2, img_size[0]//2)
        # quar_size = (onehalf_size[1]//4, onehalf_size[0]//4)
        # eith_size = (onehalf_size[1]//8, onehalf_size[0]//8)
        '''
        # perspective_transform
        pts1 = np.float32([[quar_size[0], quar_size[1]], [3 * quar_size[0], quar_size[1]],
                           [quar_size[0], 3 * quar_size[1]], [3 * quar_size[0], 3 * quar_size[1]]])
        pts2 = np.float32([[quar_size[0] - np_ran * eith_size[0], quar_size[1] - np_ran * eith_size[1]],
                           [3 * quar_size[0] + np_ran * eith_size[0], quar_size[1] - np_ran * eith_size[1]],
                           [quar_size[0] - np_ran * eith_size[0], 3 * quar_size[1] + np_ran * eith_size[1]],
                           [3 * quar_size[0] + np_ran * eith_size[0], 3 * quar_size[1] + np_ran * eith_size[1]]])
        M_perspective = cv.getPerspectiveTransform(pts1, pts2)
        # input_img = cv.warpPerspective(input_img, M_perspective, (img_size[1], img_size[0]), borderMode=cv.BORDER_REFLECT)
        input_img = cv.warpPerspective(input_img, M_perspective, (img_size[1], img_size[0]))
        '''

        if ran_aug < 0.3:
            # scale nad rotate by random -30~30 degree
            M_rot = cv.getRotationMatrix2D((onehalf_size[1] // 2, onehalf_size[0] // 2), angle * np_ran, 1)
            # input_img = cv.warpAffine(input_img, M_rot, (img_size[1], img_size[0]), borderMode=cv.BORDER_REFLECT)
            input_img = cv.warpAffine(input_img, M_rot, (img_size[1], img_size[0]))
        elif ran_aug < 0.6:
            # scale nad rotate by random 150~210 degree
            M_rot = cv.getRotationMatrix2D((onehalf_size[1] // 2, onehalf_size[0] // 2), 180 + angle * np_ran, 1)
            # input_img = cv.warpAffine(input_img, M_rot, (img_size[1], img_size[0]), borderMode=cv.BORDER_REFLECT)
            input_img = cv.warpAffine(input_img, M_rot, (img_size[1], img_size[0]))
        else:
            # random_crop
            ran2 = (ran_aug - 0.6) * 2.5
            ind_h = np.int(hshift_range * ran_degree)
            ind_w = np.int(wshift_range * ran2)
            input_img = input_img[ind_h:img_size[0] + ind_h, ind_w:img_size[1] + ind_w]
            '''
        else:
            input_img = cv.resize(input_img, (img_size[1], img_size[0]), interpolation=cv.INTER_LINEAR)
            input_img = np.fliplr(input_img)
        ran_degree = ran_degree * 0.3 + 0.7
        # random lower brightness
        input_img = (input_img * ran_degree).astype(np.uint8)
        '''
        return input_img

    def load_stereo_batch(self, ind=None, val=False):
        ind_int = (ind, ind + self.batch_size)
        if val:
            left_batch_img = np.array(list(map(lambda x: self.img_preprocess(self.cvload_img(x, self.img_shape, temp_ratio=1), normalize=True, prepro=True),
                                               self.val_left_namelist)))
            right_batch_img = np.array(list(map(lambda x: self.img_preprocess(self.cvload_img(x, self.img_shape, temp_ratio=1), normalize=True, prepro=True),
                                                self.val_right_namelist)))
        else:
            left_batch_img = np.array(list(map(lambda x: self.augment(
                self.img_preprocess(self.cvload_img(x, self.img_shape), normalize=True, prepro=True),
                self.ran_aug[ind_int[0]:ind_int[1]], self.ran_sel[ind_int[0]:ind_int[1]]),
                                               self.left_namelist[self.random_index[ind_int[0]:ind_int[1]]])))
            right_batch_img = np.array(list(map(lambda x: self.augment(
                self.img_preprocess(self.cvload_img(x, self.img_shape), normalize=True, prepro=True),
                self.ran_aug[ind_int[0]:ind_int[1]], self.ran_sel[ind_int[0]:ind_int[1]]),
                                                self.right_namelist[self.random_index[ind_int[0]:ind_int[1]]])))
        return left_batch_img, right_batch_img

    def load_seg_batch(self, ind=None, val=False):
        ind_int = (ind, ind + self.batch_size)
        if val:
            seg_batch_img = np.array(list(map(lambda x: self.imglabel_to_multiclass(self.cvload_img(x, self.img_shape, temp_ratio=1)),
                                              self.val_seg_namelist)))
        else:
            seg_batch_img = np.array(list(map(lambda x: self.augment(
                self.imglabel_to_multiclass(self.cvload_img(x, self.img_shape)),
                self.ran_aug[ind_int[0]:ind_int[1]], self.ran_sel[ind_int[0]:ind_int[1]]),
                                              self.seg_namelist[self.random_index[ind_int[0]:ind_int[1]]])))
        return seg_batch_img

    def load_depth_batch(self, ind=None, val=False):
        ind_int = (ind, ind + self.batch_size)
        if val:
            depth_batch_img = np.array(list(map(lambda x: self.imglabel_reg_normalize(self.cvload_img(x, self.img_shape, temp_ratio=1), normalize=True),
                                                self.val_depth_namelist)))
        else:
            depth_batch_img = np.array(list(map(lambda x: self.augment(
                self.imglabel_reg_normalize(self.cvload_img(x, self.img_shape), normalize=True),
                self.ran_aug[ind_int[0]:ind_int[1]], self.ran_sel[ind_int[0]:ind_int[1]]),
                                              self.depth_namelist[self.random_index[ind_int[0]:ind_int[1]]])))
        return depth_batch_img


def img_visualize(images):
    plt.figure(figsize=(20, 15))
    for i in range(0, 8):
        plt.subplot(6, 3, i + 1)
        plt.imshow(images[i])
    plt.show()
