from DensityClass import *
from scipy.spatial import Voronoi, voronoi_plot_2d
from pprint import pprint
from shapely.geometry import Polygon
from collections import defaultdict
import time

dir = "/Users/rohanrele/Documents/Research/Polynomiography/Classifier/images/"
bds_dir = "/Users/rohanrele/Documents/Research/Polynomiography/Classifier/BSDS300/images/"
PINK = [240, 128, 128]
RED = [255, 0, 0]
BURGUNDY = [128, 0, 0]
VIOLET = [238, 130, 238]
PURPLE = [128, 0, 128]
INDIGO = [75, 0, 130]
LIGHT_BLUE = [135, 206, 235]
BLUE = [65, 105, 225]
NAVY_BLUE = [0, 0, 128]
LIGHT_GREEN = [144, 238, 144]
GREEN = [0, 128, 0]
DARK_GREEN = [0, 100, 0]
PEACH = [255, 160, 122]
YELLOW = [255, 255, 0]
ORANGE = [255, 165, 0]
DARK_ORANGE = [255, 130, 0]
WHITE = [245, 255, 250]
BLACK = [0, 0, 0]
LIGHT_GRAY = [211, 211, 211]
# GRAY = [128, 128, 128]
DARK_GRAY = [105, 105, 105]
BROWN = [139, 69, 19]
COLORS = [DARK_GRAY, LIGHT_GRAY, BLACK, WHITE,
          DARK_ORANGE, ORANGE, YELLOW, PEACH,
          DARK_GREEN, GREEN, LIGHT_GREEN,
          NAVY_BLUE, BLUE, LIGHT_BLUE,
          VIOLET, PURPLE, INDIGO,
          BURGUNDY, RED, PINK, BROWN]


def roundColor(pixel):
    pr, pg, pb = tuple(pixel)
    min_col = None
    min_DC = 1000000000

    for c in COLORS:
        cr, cg, cb = tuple(c)
        rmean = (cr + pr) / 2
        delta_r = pr - cr
        delta_b = pb - cb
        delta_g = pg - cg

        DC = (2 + rmean / 256) * delta_r ** 2 + 4 * delta_g ** 2 + (2 - (255 - rmean) / 256) * delta_b ** 2

        if DC < min_DC:
            min_col = c
            min_DC = DC

    return min_col


def reservoir_sample(iterator, K):
    result = []
    N = 0

    for item in iterator:
        N += 1
        if len(result) < K:
            result.append(item)
        else:
            s = int(random.random() * N)
            if s < K:
                result[s] = item

    return result


def createColorDict(orig_img,coords):
    res = {tuple(c): [] for c in COLORS}
    for pair in coords:
        res[tuple(roundColor(orig_img[pair]))].append(pair)

    return res


def area(vertices):
    return np.sum([0.5, -0.5] * vertices * np.roll(np.roll(vertices, 1, axis=0), 1, axis=1))


def voronoi_polygons(voronoi, diameter):
    """Generate shapely.geometry.Polygon objects corresponding to the
    regions of a scipy.spatial.Voronoi object, in the order of the
    input points. The polygons for the infinite regions are large
    enough that all points within a distance 'diameter' of a Voronoi
    vertex are contained in one of the infinite polygons.

    """
    centroid = voronoi.points.mean(axis=0)

    # Mapping from (input point index, Voronoi point index) to list of
    # unit vectors in the directions of the infinite ridges starting
    # at the Voronoi point and neighbouring the input point.
    ridge_direction = defaultdict(list)
    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        u, v = sorted(rv)
        if u == -1:
            # Infinite ridge starting at ridge point with index v,
            # equidistant from input points with indexes p and q.
            t = voronoi.points[q] - voronoi.points[p]  # tangent
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t)  # normal
            midpoint = voronoi.points[[p, q]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centroid, n)) * n
            ridge_direction[p, v].append(direction)
            ridge_direction[q, v].append(direction)

    for i, r in enumerate(voronoi.point_region):
        region = voronoi.regions[r]
        if -1 not in region:
            # Finite region.
            yield Polygon(voronoi.vertices[region]), i
            continue
        # Infinite region.
        inf = region.index(-1)  # Index of vertex at infinity.
        j = region[(inf - 1) % len(region)]  # Index of previous vertex.
        k = region[(inf + 1) % len(region)]  # Index of next vertex.
        if j == k:
            # Region has one Voronoi vertex with two ridges.
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # Region has two Voronoi vertices, each with one ridge.
            dir_j, = ridge_direction[i, j]
            dir_k, = ridge_direction[i, k]

        # Length of ridges needed for the extra edge to lie at least
        # 'diameter' away from all Voronoi vertices.
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)

        # Polygon consists of finite part plus an extra edge.
        finite_part = voronoi.vertices[region[inf + 1:] + region[:inf]]
        extra_edge = [voronoi.vertices[j] + dir_j * length,
                      voronoi.vertices[k] + dir_k * length]
        yield Polygon(np.concatenate((finite_part, extra_edge))), i


def Voronize(orig_img, polygon_dict, region, N):
    n_orig, m_orig, _ = orig_img.shape
    pic_frame = np.array([[0, 0], [n_orig, 0], [n_orig, m_orig], [0, m_orig], [0, 0]])

    # if (orig_img == region).all():
    if np.array_equal(orig_img, region):
        coords_x = np.random.randint(n_orig, size=(N, 1))
        coords_y = np.random.randint(m_orig, size=(N, 1))
        coords = np.array((coords_x, coords_y)).T
        coords = [tuple(c) for c in coords[0].tolist()]

        color_point_dict = createColorDict(orig_img, coords)

        temp_keys = list(color_point_dict.keys())
        temp_vals = list(color_point_dict.values())

        ndkeys = []
        for k in temp_keys:
            r, g, b = k
            ndkeys.append(np.array([r / 255, g / 255, b / 255]))

        temp_centroid_bucket = findCentroids(temp_vals, size=True)
        cent_cols = [(x[1], ndkeys[temp_vals.index(x[0])]) for x in temp_centroid_bucket if x[1]]
        centers = [t[0] for t in cent_cols]
        ccols = [t[1] for t in cent_cols]

        vor = Voronoi(centers)
        frame_poly = Polygon(pic_frame).buffer(0)
        # plt.xlim([0, n_orig]), plt.ylim([0, m_orig])
        for p, r in voronoi_polygons(vor, 1000000):
            # x, y = zip(*p.exterior.coords)
            # plt.plot(x, y, "b-")
            obj = p.intersection(frame_poly).exterior.coords
            if obj:
                fx, fy = zip(*obj)
                poly = tuple((i, j) for i, j in zip(fx, fy))
                # plt.fill(*zip(*poly), color=ccols[r])
                polygon_dict[poly] = ccols[r]
            else:
                continue

    for polygon, p_color in polygon_dict.copy().items():

        reg = np.array([np.array(list(p)) for p in polygon])
        reg_poly = Polygon(reg).buffer(0)

        if reg_poly.area < 75:
            return polygon_dict

        xmax, ymax = reg.max(axis=0)
        xmin, ymin = reg.min(axis=0)
        exes = np.random.randint(xmin, xmax, size=(N, 1))
        whys = np.random.randint(ymin, ymax, size=(N, 1))
        pre_sample = np.array((exes, whys)).T
        pre_sample = [tuple(c) for c in pre_sample[0].tolist()]

        tmppath = Path(polygon)
        bools = tmppath.contains_points(pre_sample)
        ind_bools = np.where(bools)
        sample = np.array(pre_sample)[ind_bools]
        sample = [tuple(n) for n in sample]

        cpd = createColorDict(orig_img, sample)

        kees = list(cpd.keys())
        valoos = list(cpd.values())

        normal_k = []
        for k in kees:
            r, g, b = k
            normal_k.append(np.array([r / 255, g / 255, b / 255]))

        centr_bucket = findCentroids(valoos, size=True, sig=True, sigsize=10)
        mster = [(x[1], normal_k[valoos.index(x[0])]) for x in centr_bucket if x[1]]
        cents = [t[0] for t in mster]
        cols = [t[1] for t in mster]

        if len(cents) >= 3:

            V = Voronoi(cents)
            # plt.xlim([xmin, xmax]), plt.ylim([ymin, ymax])
            for p, r in voronoi_polygons(V, 1000000):
                # x, y = zip(*p.exterior.coords)
                # plt.plot(x, y, "r-")
                obj = p.intersection(reg_poly).exterior.coords
                if obj:
                    fx, fy = zip(*obj)
                    poly = tuple((i, j) for i, j in zip(fx, fy))
                    # plt.fill(*zip(*poly), color=cols[r])
                    polygon_dict[poly] = cols[r]
                else:
                    continue

            del polygon_dict[polygon]
            Voronize(orig_img, polygon_dict, reg, N // 2)
            polygon_dict[polygon] = p_color

    return polygon_dict

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)

    return edges

def find_edges_with(i, edge_set):
    i_first = [j for (x,j) in edge_set if x==i]
    i_second = [j for (j,x) in edge_set if x==i]
    return i_first,i_second

def stitch_boundaries(edges):
    edge_set = edges.copy()
    boundary_lst = []
    while len(edge_set) > 0:
        boundary = []
        edge0 = edge_set.pop()
        boundary.append(edge0)
        last_edge = edge0
        while len(edge_set) > 0:
            i,j = last_edge
            j_first, j_second = find_edges_with(j, edge_set)
            if j_first:
                edge_set.remove((j, j_first[0]))
                edge_with_j = (j, j_first[0])
                boundary.append(edge_with_j)
                last_edge = edge_with_j
            elif j_second:
                edge_set.remove((j_second[0], j))
                edge_with_j = (j, j_second[0])  # flip edge rep
                boundary.append(edge_with_j)
                last_edge = edge_with_j

            if edge0[0] == last_edge[1]:
                break

        boundary_lst.append(boundary)
    return boundary_lst


mountain = dir + "mountain.png"
cat = dir + "cat_pic.jpeg"
bird = bds_dir + "train/135069.jpg"
scuba = bds_dir + "train/156079.jpg"
parachute = bds_dir + "train/60079.jpg"
townhouse = bds_dir + "train/232038.jpg"
peppers = bds_dir + "train/25098.jpg"


orig_img = np.asarray(Image.open(mountain))
n, m, _ = orig_img.shape
N = 2000
alpha = 50

'''
PD = {}
start = time.time()
test = Voronize(orig_img, PD, orig_img, 1000)
print(time.time()-start)
tkeys = sorted(list(test.keys()), key=lambda x: area(np.array([np.array(list(p)) for p in x])),reverse=True)
for t in tkeys:
    plt.fill(*zip(*t), color=test[t])
'''

cx = np.random.randint(n, size=(N, 1))
cy = np.random.randint(m, size=(N,1))
c = np.array((cx,cy)).T
c = [tuple(p) for p in c[0].tolist()]

cpd_test = createColorDict(orig_img, c)
for k,v in cpd_test.items():
    if v:
        plt.scatter(*zip(*v), color=np.array([k[0]/255,k[1]/255,k[2]/255]), edgecolors='black')

plt.xlim([0, n]), plt.ylim([0, m])
plt.show()


for k,v in cpd_test.items():
    if len(v) > 3:
        points = np.array(v)
        edges = alpha_shape(points, alpha=alpha, only_outer=True)
        plt.figure()
        plt.axis('equal')
        plt.plot(points[:, 0], points[:, 1], '.')
        for i, j in edges:
            plt.plot(points[[i, j], 0], points[[i, j], 1])
        plt.show()
