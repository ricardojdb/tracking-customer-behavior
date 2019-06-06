from threading import Thread
import numpy as np
import traceback
import random
import time
import six
import sys
import cv2
import os

global colors, classes

# Set the color for the sentiment bars
random = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]


def weighted_average(Vdw, dw, beta):
    Vdw = np.asarray(Vdw)
    dw = np.asarray(dw)
    return beta * Vdw + (1-beta) * dw


def draw_box(image, label, box):
    """
    Draws a Bounding Box over a Face.
    Args:
        image (narray): the image containng the face.
        label (str): the label that goes on top of the box.
        gender (str): Gender from the gender model.
        age (int): Age from the age model.
        scores (narray): Facial expression prediciton scores.
        classes (narray): List of predicted emotions.
        colors (dict): each amotion mapped to a color.
        box (narray): Bounding box coordinates [xmin,ymin,xmax,ymax].
    Return:
        result_image (narray): edited image
    """
    xmin, ymin, xmax, ymax = np.array(box, dtype=int)
    img = np.copy(image)

    fontType = cv2.FONT_HERSHEY_DUPLEX

    fontScale_box = 0.5
    thickness_box = 1
    fontScale = 0.4
    thickness = 1

    text_size = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_DUPLEX, fontScale_box, thickness_box)
    center = (xmin+5, ymin - text_size[0][1]//2)
    pt2 = (xmin + text_size[0][0] + 10, ymin)

    box_color = (180, 0, 0)
    # Rectangle around text
    cv2.rectangle(img, (xmin, ymin - text_size[0][1]*2), pt2, box_color, 2)
    cv2.rectangle(img, (xmin, ymin - text_size[0][1]*2), pt2, box_color, -1)
    # detection rectangle
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), box_color, 2)

    # Draw text
    cv2.putText(img, label, center, cv2.FONT_HERSHEY_DUPLEX,
                fontScale_box, (255, 255, 255), thickness_box)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and
        # read the first frame from the stream
        self.stream = cv2.VideoCapture(src)
        # Change depending on the resolution of the camera
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.h = self.stream.get(4)
        self.w = self.stream.get(3)
        (self.grabbed, self.frame) = self.stream.read()

        self.data_list = None
        # initialize the variable used to indicate
        # if the thread should be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        self.stopped = False
        self.thread = Thread(target=self.update, name='camera:0', args=())
        self.thread.start()
        return self

    def update(self):
        cv2.namedWindow('final_image', cv2.WINDOW_NORMAL)
        # keep looping infinitely until the thread is stopped
        while True:
            try:
                # if the thread indicator variable is set, stop the thread
                if self.stopped:
                    self.stream.release()
                    cv2.destroyAllWindows()
                    return

                # otherwise, read the next frame from the stream
                (self.grabbed, self.frame) = self.stream.read()
                self.h = self.stream.get(4)
                self.w = self.stream.get(3)
                img = np.copy(self.frame)

                if not self.grabbed:
                    print('No frames')
                    self.stop()
                    self.stream.release()
                    cv2.destroyAllWindows()
                    return

                if self.data_list is not None:
                    for data in self.data_list:

                        img = draw_box(img, data[0], data[1])

                cv2.imshow('final_image', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stream.release()
                    cv2.destroyAllWindows()
                    self.stop()
                    return

            except Exception as e:
                traceback.print_exc(file=sys.stdout)
                return

    def read(self):
        # return the frame most recently read
        return (self.grabbed, self.frame)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
