import imutils
import random


class RotateWProb(object):
    """ Rotate an image by a given angle with a certain probability. """
    def __init__(self, angle, p=0.5, always_apply=False):
        self.angle = angle
        self.p = p
        self.always_apply = always_apply

    def __call__(self, src):
        r = random.uniform(0, 1)
        if self.always_apply or r < self.p:
            dst = imutils.rotate(src, angle=self.angle)
            return dst
        else:
            return src
