from datetime import datetime
import cPickle as pickle

import numpy as np
from scipy import interpolate
from scipy import optimize

import cv2


class Tracker():
    """Class for worm tracking algorithm"""

    def __init__(self, microscope, log_suffix):
        """Initializes the class with serial interface/camera names,
        algorithm control parameters.

        Arguments:
        serial_port -- String of name for the serial interface for the
            microscope Arduino - e.g. 'dev/tty.usb'
        """

        self.microscope = microscope
        self.microscope.set_ring_colour('FF0000')
        self.last_step_time = datetime.now()
        self.start = datetime.now()

        self.worm_spline = np.zeros((1001, 1, 2), dtype=np.int)
        self.tail = np.zeros((2,), dtype=np.int)
        self.head = np.zeros((2,), dtype=np.int)

        self.smoothing = 100
        self.draw_contour = True

        self.tracking_log = open('tracking%s.pickle' % log_suffix, 'ab')
        self.contour_log = open('contour%s.pickle' % log_suffix, 'ab')
        self.skeleton_log = open('skeleton%s.pickle' % log_suffix, 'ab')
        self.head_tail_log = open('head_tail%s.pickle' % log_suffix, 'ab')

    def find_worm(self):
        """Threshold and contouring algorithm to find centroid of worm"""
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        if self.params['adaptive']:
            size = self.params['adaptive_kernel']*2 + 1

            if self.params['adaptive_gauss']:
                mode = cv2.ADAPTIVE_THRESH_MEAN_C
            else:
                mode = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

            self.img_thresh = cv2.adaptiveThreshold(self.img_gray, 255,
                                                    mode,
                                                    cv2.THRESH_BINARY_INV,
                                                    size,
                                                    self.params['adaptive_m'])

        else:
            #Threshold
            ret, self.img_thresh = cv2.threshold(self.img_gray,
                                                 self.params['simple_value'],
                                                 255, cv2.THRESH_BINARY_INV)

        #Perform Morphological Opening/Closing

        ksize = self.params['morph_kernel']*2 + 1
        self.kernel = np.ones((ksize, ksize), np.uint8)

        if self.params['opening'] != 0:
            self.img_thresh = cv2.morphologyEx(self.img_thresh,
                                               cv2.MORPH_OPEN,
                                               self.kernel,
                                               iterations=self.params['opening'])

        if self.params['closing'] != 0:
            self.img_thresh = cv2.morphologyEx(self.img_thresh,
                                               cv2.MORPH_CLOSE,
                                               self.kernel,
                                               iterations=self.params['closing'])

        #Copy image to allow displaying later
        img_contour = self.img_thresh.copy()
        contours, hierarchy = cv2.findContours(img_contour, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)

        #Find the biggest contour
        worm_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > worm_area:
                self.worm = contour
                worm_area = area

        #Compute the centroid of the worm contour
        moments = cv2.moments(self.worm)
        self.x = int(moments['m10']/moments['m00'])
        self.y = int(moments['m01']/moments['m00'])

        #Write centroids to file

        if self.record:
            row = ('centroid', self.time, self.x, self.y)
            pickle.dump(row, self.tracking_log)
            pickle.dump((self.time, self.worm), self.contour_log)

    def skeletonise(self):
        #X and Y points defining worm boundary
        x_in = self.worm[:, 0, 0]
        y_in = self.worm[:, 0, 1]

        #Fit spline to boundary
        tck, u = interpolate.splprep([x_in, y_in], per=True, s=self.params['smoothing'],
                                     quiet=True)

        points = np.arange(0, 1.001, 0.001)

        spline = interpolate.splev(points, tck, der=0)
        x = spline[0]
        y = spline[1]
        self.worm_spline[:, 0, 0] = x
        self.worm_spline[:, 0, 1] = y

        d = interpolate.splev(points, tck, der=1)
        dx = d[0]
        dy = d[1]

        dd = interpolate.splev(points, tck, der=2)
        ddx = dd[0]
        ddy = dd[1]

        k = dx*ddy - dy*ddx
        k = np.absolute(k)
        k = k/((dx**2 + dy**2)**1.5)

        tail_n = np.argmax(k)
        k[tail_n-125:tail_n+125] = 0
        head_n = np.argmax(k)

        self.tail[0] = int(x[tail_n])
        self.tail[1] = int(y[tail_n])

        self.head[0] = int(x[head_n])
        self.head[1] = int(y[head_n])

        if self.record:
            pickle.dump((self.time, self.head, self.tail), self.head_tail_log)

        tail_p = points[tail_n]
        head_p = points[head_n]

        if head_n > tail_n:
            u_a = np.linspace(tail_p, head_p, self.params['centre_points'])
            side_a = interpolate.splev(u_a, tck, der=0)

            n = round((1-head_p)*self.params['centre_points'])
            u_b_1 = np.linspace(head_p, 1, n)
            u_b_2 = np.linspace(0, tail_p, (self.params['centre_points']-n))
            u_b = np.concatenate((u_b_1, u_b_2))
            side_b = interpolate.splev(u_b, tck, der=0)
            side_b = np.fliplr(side_b)

        else:
            u_a = np.linspace(head_p, tail_p, self.params['centre_points'])
            side_a = interpolate.splev(u_a, tck, der=0)
            side_a = np.fliplr(side_a)

            n = round((1-tail_p)*self.params['centre_points'])
            u_b_1 = np.linspace(tail_p, 1, n)
            u_b_2 = np.linspace(0, head_p, (self.params['centre_points']-n))
            u_b = np.concatenate((u_b_1, u_b_2))
            side_b = interpolate.splev(u_b, tck, der=0)

        self.skeleton = (side_a + side_b)/2

        if self.params['refine']:
            spline_a, u_a = interpolate.splprep(side_a, s=100)
            spline_b, u_b = interpolate.splprep(side_b, s=100)
            centre_spline, u = interpolate.splprep(self.skeleton, s=100)

            points = np.linspace(0, 1, self.params['centre_points'])
            self.skeleton = np.array(interpolate.splev(points, centre_spline,
                                     der=0))
            tangents = interpolate.splev(points, centre_spline, der=1)
            tangents = np.array(tangents)
            tangents = tangents/np.linalg.norm(tangents, axis=0)
            normals = np.array([tangents[1, :]*-1, tangents[0, :]])

            args = (spline_a, self.skeleton, normals)
            sol_a = optimize.root(self.intersection, u_a,
                                  args)
            side_a = interpolate.splev(sol_a.x, spline_a, der=0)

            args = (spline_b, self.skeleton, normals)
            sol_b = optimize.root(self.intersection, u_b, args)
            side_b = interpolate.splev(sol_b.x, spline_b, der=0)

            self.skeleton = (np.array(side_a) + np.array(side_b))/2

        if self.record:
            pickle.dump((self.time, self.skeleton), self.skeleton_log)

    def intersection(self, p, spline, origins, normals):

        spline_points = np.array(interpolate.splev(p, spline, der=0))

        delta = origins - spline_points
        displacement = delta - (np.sum(delta*normals, axis=0))*normals

        return displacement[0, :]

    def move_stage(self):
        now = datetime.now()
        elapsed_time = ((now - self.last_step_time).microseconds)/1000

        if self.params['tracking'] and elapsed_time > self.params['step_interval']:
            self.last_step_time = now
            if self.x < self.params['margin']:
                self.microscope.move('x', -1*self.params['step_size'])
                print 'Moving Left'
                if self.record:
                    row = ('step', self.time, 'x', -1)
                    pickle.dump(row, self.tracking_log)

            if self.x > (self.width - self.params['margin']):
                self.microscope.move('x', self.params['step_size'])
                print 'Moving Right'
                if self.record:
                    row = ('step', self.time, 'x', 1)
                    pickle.dump(row, self.tracking_log)

            if self.y < self.params['margin']:
                self.microscope.move('y', -1*self.params['step_size'])
                print 'Moving Up'
                if self.record:
                    row = ('step', self.time, 'y', -1)
                    pickle.dump(row, self.tracking_log)

            if self.y > (self.height - self.params['margin']):
                self.microscope.move('y', self.params['step_size'])
                print 'Moving Down'
                if self.record:
                    row = ('step', self.time, 'y', 1)
                    pickle.dump(row, self.tracking_log)

    def draw_features(self):

        if self.params['show_threshold']:
            self.img = cv2.cvtColor(self.img_thresh, cv2.COLOR_GRAY2BGR)

        #Draw the splined contour if computed, otherwise
        if self.draw_contour:
            if self.params['skeleton']:
                cv2.drawContours(self.img, [self.worm_spline],
                                 -1, (255, 0, 0), 2)
            else:
                cv2.drawContours(self.img, [self.worm], -1, (255, 0, 0), 2)

        #Draw markers for centroid and boundry
        cv2.circle(self.img, (self.x, self.y), 5, (0, 0, 255), -1)

        cv2.rectangle(self.img, (self.params['margin'], self.params['margin']),
                      (self.width-self.params['margin'],
                       self.height-self.params['margin']),
                      (0, 255, 0),
                      2)

        #If skeletonise draw markers for head and tail
        if self.params['skeleton']:
            cv2.circle(self.img, (self.tail[0], self.tail[1]),
                       5, (0, 255, 255), -1)
            cv2.circle(self.img, (self.head[0], self.head[1]),
                       5, (255, 255, 0), -1)

            self.skeleton = np.transpose(self.skeleton)

            cv2.polylines(self.img, np.int32([self.skeleton]),
                          False, (255, 0, 0), 2)

    def process_frame(self, img):

            self.record = self.params['record']

            self.time = datetime.now()
            self.img = img

            height, width, depth = self.img.shape

            self.width = width
            self.height = height

            self.find_worm()

            if self.params['skeleton']:
                self.skeletonise()

            self.draw_features()
            self.move_stage()

            return self.img
