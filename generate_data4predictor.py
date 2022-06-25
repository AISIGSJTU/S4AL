from albumentations import Compose, Rotate, ElasticTransform, IAAPiecewiseAffine
import cv2
import numpy as np
import os
from tqdm import trange


def rectangle(x, y, w, h, img, flag=-1, ite=3):
    x = max(x, 0)
    y = max(y, 0)
    x2 = min(x + w, 511)
    y2 = min(y + h, 511)
    img[int(x):int(x2), int(y):int(y2)] = 255
    flag *= -1
    ite -= 1
    if ite > 0:
        if flag == 1:
            w2 = np.random.randint(int(h / 4), int(3 * h / 4 + 1))
            h2 = np.random.randint(int(w / 4), int(3 * w / 4 + 1))
            eps = np.random.randint(1, int(5 * h / 6))
            img = rectangle(x - w2, y + h / 3, w2, h2, img, flag, ite)
            w2 = np.random.randint(int(h / 4), int(3 * h / 4 + 1))
            h2 = np.random.randint(int(w / 4), int(3 * w / 4 + 1))
            eps = np.random.randint(1, int(5 * h / 6))
            img = rectangle(x, y + eps, w2, h2, img, flag, ite)
        else:
            w2 = np.random.randint(int(h / 4), int(3 * h / 4 + 1))
            h2 = np.random.randint(int(w / 4), int(3 * w / 4 + 1))
            eps = np.random.randint(1, int(5 * w / 6))
            img = rectangle(x + eps, y - h2, w2, h2, img, flag, ite)
            w2 = np.random.randint(int(h / 4), int(3 * h / 4 + 1))
            h2 = np.random.randint(int(w / 4), int(3 * w / 4 + 1))
            eps = np.random.randint(1, int(5 * w / 6))
            img = rectangle(x + eps, y + h, w2, h2, img, flag, ite)
    return img


def main():
    out_path = "datasets/predictor/train"
    os.makedirs(out_path, exist_ok=False)

    max_num = 50000  # train
    for i in trange(0, max_num):
        scale = np.random.uniform(0, 0.5)
        alpha = np.random.randint(0, 10000)
        sigma = np.random.randint(0, 60)
        aug = Compose([
            IAAPiecewiseAffine(
                scale=(scale, scale + 1e-6),
                always_apply=True,
            ),
            ElasticTransform(
                alpha=alpha,
                sigma=sigma,
                always_apply=True,
            ),
            Rotate(
                limit=90,
                always_apply=True
            ),
        ], p=1)

        img = np.zeros((512, 512))
        x = np.random.randint(10, 250)
        w = np.random.randint(15, 20)
        y = np.random.randint(2, 80)
        h = np.random.randint(350, 450)
        ite = np.random.randint(3, 5)
        img = rectangle(x, y, w, h, img, -1, ite)
        img = aug(image=img)['image']
        ipath = "{}/{}.png".format(out_path, str(i).zfill(5))
        cv2.imwrite(ipath, img)

    out_path = "datasets/predictor/test"
    os.makedirs(out_path, exist_ok=False)

    max_num = 5000  # test
    for i in trange(0, max_num):
        scale = np.random.uniform(0, 0.5)
        alpha = np.random.randint(0, 10000)
        sigma = np.random.randint(0, 60)
        aug = Compose([
            IAAPiecewiseAffine(
                scale=(scale, scale + 1e-6),
                always_apply=True,
            ),
            ElasticTransform(
                alpha=alpha,
                sigma=sigma,
                always_apply=True,
            ),
            Rotate(
                limit=90,
                always_apply=True
            ),
        ], p=1)

        img = np.zeros((512, 512))
        x = np.random.randint(10, 250)
        w = np.random.randint(15, 20)
        y = np.random.randint(2, 80)
        h = np.random.randint(350, 450)
        ite = np.random.randint(3, 5)
        img = rectangle(x, y, w, h, img, -1, ite)
        img = aug(image=img)['image']
        ipath = "{}/{}.png".format(out_path, str(i).zfill(5))
        cv2.imwrite(ipath, img)


if __name__ == '__main__':
    main()
