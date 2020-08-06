# Polynomiograph Degree Classifier

from polyGenerate import *
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
from PIL import Image
from skimage import filters, morphology, color
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976
from numpy.lib.stride_tricks import as_strided
from scipy.spatial import Delaunay, ConvexHull, convex_hull_plot_2d
import cv2

S = [(110.0, 245.0), (142.0, 255.0), (127.0, 249.0)]

def avg_brightness(pixelmap,cx,cy,r=21):
    sum = 0
    for i in range(cx-r,cx+r):
        for j in range(cx-r,cx+r):
            pixel_avg = ( pixelmap[i][j][0] + pixelmap[i][j][1] + pixelmap[i][j][2] ) / 3
            sum += pixel_avg
    return sum/(4*r*r)

def guess_roots(pmap,metric,r=21,e=0.0005):
    n = 0
    roots = []
    temp = np.array(pmap)
    temp = np.array([[(sum(j)/len(j)) for j in i] for i in temp])
    submatrix = (r,r)
    view_shape = tuple(np.subtract(temp.shape, submatrix) + 1) + submatrix
    window = as_strided(temp, view_shape, temp.strides * 2)
    window = window.reshape((-1,) + submatrix)

    num_of_windows = window.shape[0]

    for x in range(num_of_windows):
        if (metric-e) <= np.average(window[x]) <= (metric+e):
            n += 1
            roots.append( window[x][int(r/2)][int(r/2)] )
    return n, roots

def img_to_cmatrix(filepath):
    # this is the function I used to extract a 'patch' of pixels
    # missing two arguments func(filepath,x,y,n) - extracts a square starting at (x,y) as the top left corner
    img = Image.open(filepath)
    arr = np.array(img)
    arr = list(map(tuple, arr))
    df = pd.DataFrame(arr)
    df.to_csv(filepath[:-4] + ".csv", index=False)
    return arr


def convert(filepath):
    df = pd.read_csv(filepath)
    res = [[None for _ in range(df.shape[0])] for _ in range(df.shape[1])]
    for i in range(0, df.shape[0]):
        for j in range(0, df.shape[1]):
            res[i][j] = tuple(map(int, df.iloc[i, j].replace('[', '').split()[:-1]))
    return res


def comparePixels(arr1, arr2, eps=0.001):
    rgb_pixel1 = [[arr1]]
    rgb_pixel2 = [[arr2]]
    lab1 = color.rgb2lab(rgb_pixel1)
    lab2 = color.rgb2lab(rgb_pixel2)
    color1 = LabColor(lab1[0][0][0], lab1[0][0][1], lab1[0][0][2])
    color2 = LabColor(lab2[0][0][0], lab2[0][0][1], lab2[0][0][2])
    if delta_e_cie1976(color1, color2) <= eps:
        return True
    return False


# can also be called findDarkestPoint(), however, we need to apply this function n times where n = deg of polynomial
def findRoot(img_fp, dark=False):
    radius = 21 # need to figure this out
    img = cv2.imread(img_fp)
    orig = img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray_img, (radius, radius), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    param = maxLoc
    if dark:
        param = minLoc
    z = param[0]+param[1]*1j
    cv2.circle(img, param, radius, (255, 0, 0), 2)
    cv2.imshow("OneRoot", img)
    return param, z

# THESE ARE ALL COPIED METHODS FROM A BLOG POST ******************************************************

def edge_points(image, length_scale=200,n_horizontal_points=None,n_vertical_points=None):
    ymax, xmax = image.shape[:2]

    if n_horizontal_points is None:
        n_horizontal_points = int(xmax / length_scale)

    if n_vertical_points is None:
        n_vertical_points = int(ymax / length_scale)

    delta_x = xmax / n_horizontal_points
    delta_y = ymax / n_vertical_points

    return np.array(
        [[0, 0], [xmax, 0], [0, ymax], [xmax, ymax]]
        + [[delta_x * i, 0] for i in range(1, n_horizontal_points)]
        + [[delta_x * i, ymax] for i in range(1, n_horizontal_points)]
        + [[0, delta_y * i] for i in range(1, n_vertical_points)]
        + [[xmax, delta_y * i] for i in range(1, n_vertical_points)]
    )

#Returns an array of shape, with values based on: amp * exp(-((i-x)**2 +(j-y)**2) / (2 * sigma ** 2))
def gaussian_mask(x, y, shape, amp=1, sigma=15):
    xv, yv = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    g = amp * np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * sigma ** 2))
    return g

#entropy points
def generate_max_entropy_points(image,n_points,entropy_width,filter_width,suppression_width,suppression_amplitude):
    # calculate length scale
    ymax, xmax = image.shape[:2]
    length_scale = np.sqrt(xmax*ymax / n_points)
    entropy_width = length_scale * entropy_width
    filter_width = length_scale * filter_width
    suppression_width = length_scale * suppression_width

    # convert to grayscale
    im2 = color.rgb2gray(image)

    # filter
    im2 = (
        255 * filters.gaussian(im2, sigma=filter_width, multichannel=True)
    ).astype("uint8")

    # calculate entropy
    im2 = filters.rank.entropy(im2, morphology.disk(entropy_width))

    points = []
    for _ in range(n_points):
        y, x = np.unravel_index(np.argmax(im2), im2.shape)
        im2 -= gaussian_mask(x, y,
                             shape=im2.shape[:2],
                             amp=suppression_amplitude,
                             sigma=suppression_width)
        points.append((x, y))

    points = np.array(points)
    return points

def draw_triangles(ax, points, vertices, colours=None, **kwargs):
    if colours is None:
        face_colours = len(vertices) * ["none"]
        line_colours = len(vertices) * ["black"]
    else:
        face_colours = colours
        line_colours = colours

    for triangle, fc, ec in zip(vertices, face_colours, line_colours):
        p = Polygon([points[i]
                     for i in triangle],
                    closed=True, facecolor=fc,
                    edgecolor=ec, **kwargs)
        ax.add_patch(p)

def triangulation(img_fp,n=100,save=False,img_str=''):
    im = cv2.imread(img_fp)
    points = generate_max_entropy_points(im,n)
    points = np.concatenate([points, edge_points(im)])

    tri = Delaunay(points)
    fig, ax = plt.subplots(ncols=1, figsize=(18,8), sharey=True)
    ax.scatter(x=points[:, 0], y=points[:, 1], color="k") #draw points
    draw_triangles(ax, tri.points, tri.vertices)

    if not save:
        #ax = axs[0]
        ax.imshow(im)
        ax.set_title("Triangulation")
        ax.axis("off")
        ax.axis("tight")
        ax.set_aspect("equal")
        ax.autoscale(False)
        plt.show()
        return '',tri
    else:
        s = img_str.split('.',1)[0]
        plt.savefig( "images/"+ s + "_triangluation.png")
        return (s + "_triangulation.png"), tri

# **********************************************************************************************

# find n = degree of polynomial solely from image
def findNumRoots(img_fp):
    im = cv2.imread(img_fp)
    # convert to grayscale
    im2 = (255 * color.rgb2gray(im)).astype("uint8")

    # calculate entropy
    im2 = filters.rank.entropy(im2, morphology.disk(20))

    # plot it
    fig, ax = plt.subplots(figsize=(6,8), sharey=True)
    cax = ax.imshow(im2)
    fig.colorbar(cax);
    plt.show()

def findLargestTriangle(img_fp,show=False):
    img = cv2.imread(img_fp)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    # threshold to get just the signature (INVERTED)
    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, \
                                       type=cv2.THRESH_BINARY_INV)

    image, contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, \
                                       cv2.CHAIN_APPROX_SIMPLE)

    # Find object with the biggest bounding box
    mx = (0,0,0,0)      # biggest bounding box so far
    mx_area = 0
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        area = w*h
        if area > mx_area:
            mx = x,y,w,h
            mx_area = area
    x,y,w,h = mx

    if show:
        roi=img[y:y+h,x:x+w]
        cv2.imshow('largest_shape',roi)
        cv2.waitKey(0)

    return mx_area

def longest(list1):
    l = 0
    if type(list1) is list:
        l = len(list1)
        if l > 0:
            l = max(l,max(longest(elem) for elem in list1))
    return l

def getEntropy(img_fp,n=200,entropy_width=0.2,filter_width=0.1,suppression_width=0.3,suppression_amplitude=3):
    im = cv2.imread(img_fp)
    points = generate_max_entropy_points(im,n,entropy_width,filter_width,suppression_width,suppression_amplitude)
    points = np.concatenate([points, edge_points(im)])

    tri = Delaunay(points)

    return tri

def myPlot(points, borders): #vertices):
    plt.scatter(*zip(*points),color="black")

    for dc in borders:
        if len(dc) > 1:
            plt.scatter(*zip(*dc),color="red")
        elif len(dc) == 1:
            plt.scatter(*zip(*dc),color="blue")
        else:
            continue

    plt.show()

def myPlot2(points,buckets):
    plt.scatter(*zip(*points),color="black")

    for obj in buckets:
        e1 = obj[1]
        e2 = obj[2]
        plt.scatter(*zip(*obj[0]),color="red")
        plt.scatter(e1[0],e1[1],color="green")
        plt.scatter(e2[0],e2[1],color="green")

    plt.show()

def myPlot3(points,buckets): #heatmap based on class size

    l = longest(buckets)
    plt.scatter(*zip(*points),color="black")
    C = []
    for obj in buckets:
        x = len(obj)
        C.append([x/l] * x)

    i = 0
    for obj in buckets:

        if obj == S:
            plt.scatter(*zip(*obj),color='red')
        else:
            plt.scatter(*zip(*obj),c=C[i],cmap='winter')

        i += 1

    plt.show()

def myPlot4(points,buckets,save=False,fpath=""):

    plt.scatter(*zip(*points),color="black")
    for obj in buckets:
        l = obj[0]
        c = obj[1]
        plt.scatter(*zip(*l),color='red')
        plt.scatter(c[0],c[1],color='blue',marker="x")

    if save:
        plt.axis('off')
        plt.savefig(fpath,bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

def myPlot5(centroid_list,colormap):
    i = 0
    for obj in centroid_list:
        l = obj[0]
        c = obj[1]
        if l:
            plt.scatter(*zip(*l),color=colormap[i],edgecolors="black")
            plt.scatter(c[0],c[1],color=colormap[i],marker="x",edgecolors="red")
        i += 1
