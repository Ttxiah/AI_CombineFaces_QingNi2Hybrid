import os
import numpy as np
import cv2
import dlib
from collections import OrderedDict


def get_landmarks(img):
    rects = detector(img, 1)
    return np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])


def linear_transformation_matrix(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def get_mask(im, part_marks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    points = cv2.convexHull(part_marks)
    cv2.fillConvexPoly(im, points, color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


if __name__ == '__main__':
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    FACE = cv2.imread('faces/anqi.jpeg')
    face_marks = get_landmarks(FACE)
    face_shape = FACE.shape

    # '1mouth', '2eyebrow', '3eye', '4nose'
    switch_list = ['2eyebrow', '3eye', '4nose']

    FACIAL_SOURCE_IMGS = OrderedDict([
        ('1mouth', cv2.imread('faces/jinzihan.jpeg')),
        ('2eyebrow', cv2.imread('faces/wangchengxuan.jpeg')),
        ('3eye', cv2.imread('faces/qisui.jpeg')),
        ('4nose', cv2.imread('faces/liuyuxin.jpeg'))
    ])

    FACIAL_LANDMARKS_IDXS = OrderedDict([
        ('1mouth', (48, 68)),
        ('2eyebrow', (17, 27)),
        ('3eye', (36, 48)),
        ('4nose', (27, 35))
    ])

    ALIGN_POINTS = (list(range(48, 61)) + list(range(17, 27)) +
                    list(range(36, 48)) + list(range(27, 35)))

    for name in switch_list:
        part_im = FACIAL_SOURCE_IMGS[name]
        landmarks = get_landmarks(part_im)

        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        part_idx = list(range(j, k))

        # 展示找到的五官轮廓

        # for coor in landmarks[part_idx]:
        #     x, y = coor[0, 0], coor[0, 1]
        #     cv2.circle(part_im, (x, y), 1, (0, 0, 255), -1)
        # cv2.imshow('found_' + name, part_im)
        # cv2.waitKey(0)

        # 找到角度方向大小的变形矩阵

        Matrix = linear_transformation_matrix(face_marks[ALIGN_POINTS], landmarks[ALIGN_POINTS])

        # 画一个空板挖出五官的空/mask

        FEATHER_AMOUNT = 11
        mask = get_mask(part_im, landmarks[part_idx])
        warped_mask = warp_im(mask, Matrix, face_shape)
        combined_mask = np.max([get_mask(FACE, face_marks[part_idx]), warped_mask], axis=0)

        warped_part = warp_im(part_im, Matrix, face_shape)

        FACE = FACE * (1.0 - combined_mask) + warped_part * combined_mask

    cv2.imwrite('QingNi2HybiridFace.jpeg', FACE)



