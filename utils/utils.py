from threading import Thread
from collections import deque
from heatmappy import Heatmapper
from PIL import Image

import numpy as np
import traceback
import random
import copy
import time
import six
import sys
import cv2
import os


# Set the color for the sentiment bars
def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    # filter removes empty strings (such as last line)
    return list(filter(None, names))

COLORS = {}

for label in load_classes("dockers/detection/models/yolov3/coco.names"):
    COLORS[label] = [random.randint(0, 255) for _ in range(3)]

CAMERA_COORDS = np.asarray([[1106, 646],
                            [764, 228],
                            [1252, 300],
                            [1688, 154]],
                           dtype=np.float32)

PLANE_COORDS = np.asarray([[218, 306],
                           [486, 306],
                           [336, 426],
                           [394, 826]],
                          dtype=np.float32)

PERSPECTIVE_MAT = cv2.getPerspectiveTransform(CAMERA_COORDS, PLANE_COORDS)


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

    box_color = COLORS[label]
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


def perspective_transform(x, y):
    new_pos = np.dot(PERSPECTIVE_MAT, [x, y, 1.0])
    new_pos = new_pos[:2] / new_pos[-1]
    new_x, new_y = new_pos.T
    return new_x, new_y


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
        self.points = deque(maxlen=500)
        self.heatmap_img = cv2.imread('images/square_plane.jpg')

    def start(self):
        # start the thread to read frames from the video stream
        self.stopped = False
        self.thread = Thread(target=self.update, name='camera:0', args=())
        self.thread.start()
        return self

    def update(self):
        cv2.namedWindow('final_image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('heatmap', cv2.WINDOW_NORMAL)
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

                person_detected = False
                if self.data_list is not None:
                    for data in self.data_list:
                        img = draw_box(img, data[0], data[1])
                        xmin, ymin, xmax, ymax = data[1]
                        if data[0] == 'person':
                            person_detected = True
                            xmid = xmin + ((xmax-xmin)//2)
                            x_plane, y_plane = perspective_transform(
                                xmid, ymax)
                            self.points.append([x_plane, y_plane])

                if not person_detected:
                    if len(self.points) != 0:
                        self.points.popleft()

                cv2.imshow('final_image', img)
                cv2.imshow('heatmap', self.heatmap_img)
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


def heatmap_calculator(video_class):
    print('heatmap one time')
    plano_ori_heat = Image.open('images/square_plane.jpg')
    heatmapper = Heatmapper(
            point_diameter=50,
            point_strength=0.25,
            opacity=0.65,
            colours='default',
            grey_heatmapper='PIL')

    while True:
        points = copy.copy(video_class.points)
        if len(points) != 0:
            heatmap_img = heatmapper.heatmap_on_img(points, plano_ori_heat)
            heatmap_img = np.asarray(heatmap_img)
            heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
            video_class.heatmap_img = heatmap_img
        else:
            heatmap_img = np.asarray(plano_ori_heat)
            heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
            video_class.heatmap_img = heatmap_img
        if video_class.stopped:
            return
