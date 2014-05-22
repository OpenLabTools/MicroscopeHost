import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import optimize

img = cv2.imread('tracking_fig01.jpg', 0)
ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

worm = contours[0]
contour_length = worm.shape[0]

x = worm[:, 0, 0]
y = worm[:, 0, 1]

tck, u = interpolate.splprep([x, y], per=True,  s=100)

p = np.arange(0, 1.001, 0.001)

out = interpolate.splev(p, tck, der=0)
x = out[0]
y = out[1]

d = interpolate.splev(p, tck, der=1)
dx = d[0]
dy = d[1]

dd = interpolate.splev(p, tck, der=2)
ddx = dd[0]
ddy = dd[1]

k = dx*ddy - dy*ddx
k = np.absolute(k)
k = k/((dx**2 + dy**2)**1.5)

plt.plot(k)
plt.show()

tail_n = np.argmax(k)
k[tail_n-250:tail_n+250] = 0
head_n = np.argmax(k)

tail_p = p[tail_n]
head_p = p[head_n]

if head_p > tail_p:
    u_a = np.linspace(tail_p, head_p, 600)
    side_a = interpolate.splev(u_a, tck, der=0)

    spline_a, u_a = interpolate.splprep(side_a, s=100)
    side_a = interpolate.splev(u_a, spline_a, der=0)
    side_a = np.array(side_a)

    n = round((1-head_p)*600)
    u_b_1 = np.linspace(head_p, 1, n)
    u_b_2 = np.linspace(0, tail_p, (600-n))
    u_b = np.concatenate((u_b_1, u_b_2))
    side_b = interpolate.splev(u_b, tck, der=0)
    side_b = np.fliplr(side_b)

    spline_b, u_b = interpolate.splprep(side_b, s=100)
    side_b = interpolate.splev(u_b, spline_b, der=0)
    side_b = np.array(side_b)


seg = (side_a + side_b)/2

centre_spline, u = interpolate.splprep(seg, s=100)

points600 = np.linspace(0, 1, 600)

seg = np.array(interpolate.splev(points600, centre_spline, der=0))

tangents = interpolate.splev(points600, centre_spline, der=1)
tangents = np.array(tangents)
tangents = tangents/np.linalg.norm(tangents, axis=0)

normals = np.array([tangents[1, :]*-1, tangents[0, :]])


def intersectionDistance(p, spline, origins, normals):

    spline_points = np.array(interpolate.splev(p, spline, der=0))

    delta = origins - spline_points
    displacement = delta - (np.sum(delta*normals, axis=0))*normals

    return displacement[0, :]


def jacobian(p, spline, origins, normals):

    spline_derivatives = np.array(interpolate.splev(p, spline, der=1))

    partials = np.zeros((2, 600))

    for i in range(600):
        partials[:, i] = (-1 * normals[:, i] * np.dot(np.array([-1, -1]), normals[:, i]) - 1)*spline_derivatives[:, i]

    return np.diag(partials[0, :])

old_seg = np.copy(seg)

'''
args = spline_a, seg, normals

sol_a = optimize.root(intersectionDistance, u_a, args, jac=jacobian)
side_a = interpolate.splev(sol_a.x, spline_a, der=0)

args = spline_b, seg, normals

sol_b = optimize.root(intersectionDistance, u_b, args, jac=jacobian)
side_b = interpolate.splev(sol_b.x, spline_b, der=0)

seg = (np.array(side_a) + np.array(side_b))/2
'''

d = jacobian(u_a, spline_a, seg, normals)
numerical = intersectionDistance(u_a, spline_a, seg, normals)
numerical = intersectionDistance(u_a + 0.0000001, spline_a, seg, normals) - numerical

numerical = numerical/0.0000001
print numerical - np.diag(d)


plt.plot(x, y, x[tail_n], y[tail_n], 'o', x[head_n], y[head_n], 'o', seg[0],
         seg[1], old_seg[0], old_seg[1])
plt.show()
