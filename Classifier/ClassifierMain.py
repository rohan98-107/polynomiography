from PolyClassifier import *
import matplotlib
from matplotlib.collections import LineCollection
import itertools
import math


def triangle_csc(pts):
    rows, cols = pts.shape

    A = np.bmat([[2 * np.dot(pts, pts.T), np.ones((rows, 1))],
                 [np.ones((1, rows)), np.zeros((1, 1))]])

    b = np.hstack((np.sum(pts * pts, axis=1), np.ones((1))))
    x = np.linalg.solve(A, b)
    bary_coords = x[:-1]
    return np.sum(pts * np.tile(bary_coords.reshape((pts.shape[0], 1)), (1, pts.shape[1])), axis=0)


def voronoi(P):
    delauny = Delaunay(P)
    triangles = delauny.points[delauny.vertices]

    lines = []

    # Triangle vertices
    A = triangles[:, 0]
    B = triangles[:, 1]
    C = triangles[:, 2]
    lines.extend(zip(A, B))
    lines.extend(zip(B, C))
    lines.extend(zip(C, A))
    # lines = matplotlib.collections.LineCollection(lines, color='r')
    # plt.gca().add_collection(lines)

    circum_centers = np.array([triangle_csc(tri) for tri in triangles])

    segments = []
    for i, triangle in enumerate(triangles):
        circum_center = circum_centers[i]
        for j, neighbor in enumerate(delauny.neighbors[i]):
            if neighbor != -1:
                segments.append((circum_center, circum_centers[neighbor]))
            else:
                ps = triangle[(j + 1) % 3] - triangle[(j - 1) % 3]
                ps = np.array((ps[1], -ps[0]))

                middle = (triangle[(j + 1) % 3] + triangle[(j - 1) % 3]) * 0.5
                di = middle - triangle[j]

                ps /= np.linalg.norm(ps)
                di /= np.linalg.norm(di)

                if np.dot(di, ps) < 0.0:
                    ps *= -1000.0
                else:
                    ps *= 1000.0
                segments.append((circum_center, circum_center + ps))
    return segments


def displayVoronoi(points):
    segments = voronoi(points)
    axes = plt.subplot(1, 1, 1)

    lines = matplotlib.collections.LineCollection(segments, color='k')
    axes.add_collection(lines)
    plt.axis([-0.05, 720, -0.05, 720])
    plt.show()


def generateEdgeDict(points, vertices):
    result = dict()
    for indices in vertices:
        ax, ay = tuple(points[indices][0])
        bx, by = tuple(points[indices][1])
        cx, cy = tuple(points[indices][2])

        dab = math.sqrt((by - ay) ** 2 + (bx - ax) ** 2)
        dac = math.sqrt((cy - ay) ** 2 + (cx - ax) ** 2)
        dbc = math.sqrt((cy - by) ** 2 + (cx - bx) ** 2)

        if ((bx, by), (ax, ay)) in set(result.keys()):
            continue
        else:
            result[((ax, ay), (bx, by))] = [(-1, -1)]
            result[((ax, ay), (bx, by))].append(dab)

        if ((cx, cy), (ax, ay)) in set(result.keys()):
            continue
        else:
            result[((ax, ay), (cx, cy))] = [(-1, -1)]
            result[((ax, ay), (cx, cy))].append(dac)

        if ((cx, cy), (bx, by)) in set(result.keys()):
            continue
        else:
            result[((bx, by), (cx, cy))] = [(-1, -1)]
            result[((bx, by), (cx, cy))].append(dbc)

    return result

def generatePointDict(edge_dict):
    result = dict()
    for key, value in edge_dict.items():
        if not key[0] in result.keys():
            result[key[0]] = [(key[1],value[1])]
        else:
            result[key[0]].append((key[1],value[1]))

        if not key[1] in result.keys():
            result[key[1]] = [(key[0],value[1])]
        else:
            result[key[1]].append((key[0],value[1]))

    return result

def densityClasses(d):
    # if isConnected(curr,next) AND (curr.length * 0.8 < next.length < curr.length * 1.2): place next in same...
    # ... density class as curr
    #
    # if, for all edges in a bucket (set), there are no edges not already in the set which satisfy the criteria:...
    # ...that density class (set) is complete, remove all edges from temporary dict, then pick a new edge...
    # ...at random to re-start the loop
    #
    # RECURSIVE TREE IDEA:
    # inputs: d
    # initialize: a dictionary of representatives

    temp_dict = d

    while chooseUnassignedEdge(d) is not None:
        init_edge = chooseUnassignedEdge(temp_dict)
        temp_dict[init_edge][0] = init_edge[0]
        path_stack = [init_edge]
        rep = init_edge[0]

        while path_stack:
            edge = path_stack[-1]
            connected = getNextEdges(temp_dict, edge)
            i = 0
            while not connected:
                if not path_stack:
                    break
                edge = path_stack.pop()
                connected = getNextEdges(temp_dict, edge)
                for elem in connected:
                    if not isAssigned(temp_dict, elem):
                        i = connected.index(elem)
                        break
                    else:
                        connected.remove(elem)

            if not path_stack:
                break

            # path_stack.remove(edge)
            next = connected[i]
            path_stack.append(next)
            for e in connected:
                temp_dict[e][0] = rep
                path_stack.append(e)

    return temp_dict


def getNextEdges(d, edge, assigned=True):
    # temp = set()
    # temp.add(point)
    # nextPoint = (set(edge).symmetric_difference(temp)).pop()

    if assigned:
        nextEdges = [k for k, v in d.items() if k[0] in set(edge)
        and (v[1] * 0.5 <= d[edge][1] <= v[1] * 1.5)
        and v[0] == (-1, -1)]
    else:
        nextEdges = [k for k, v in d.items() if k[0] in set(edge)
        and (v[1] * 0.5 <= d[edge][1] <= v[1] * 1.5)]

    return nextEdges

def isBorder(list, epsilon = 0.2):

    for tupx in list:
        for tupy in list:
            if not tupx[1]*(1-epsilon) <= tupy[1] <= tupx[1]*(1+epsilon):
                return True

    return False

def densityClasses2(points):

    res = []

    for p, value in points.items():
        if onBoundary(points,p):
            continue
        if isBorder(points[p], 0.5):
            dummy = False
            for c in res:
                val = points[p]
                for x in c:
                    if x in [y[0] for y in val]:
                        dummy = True
                        c.append(p)
                        break
                if dummy:
                    break

            if not any(p in c for c in res):
                res.append([p])

    return res

def densityClasses3(points,origin=(0,0),eps = 0.4):

    res = []
    ox, oy = origin
    PI = math.pi

    for p, value in points.items():
        if onBoundary(points,p):
            continue
        if isBorder(points[p], 0.5):
            dummy = False
            for c in res:
                val = points[p]
                for coord in c:
                    cx, cy = coord
                    theta = math.atan2(cy-oy,cx-ox) + 2*PI
                    dist = math.sqrt((cy-oy)**2 + (cx-ox)**2)

                    thetap = math.atan2(p[1]-oy,p[0]-ox) + 2*PI
                    distp = math.sqrt((p[1]-oy)**2 + (p[0]-ox)**2)

                    if coord in [y[0] for y in val]:
                        I1 = math.ceil(eps)
                        I2 = math.ceil(math.sqrt(2))
                        s2 = math.sqrt(2)

                        dummy = True
                        condition1 = I1-eps * thetap <= theta <= (1+eps) * thetap
                        condition2 = I2-s2 * distp <= dist <= s2 * distp
                        if condition1 and condition2:
                            c.append(p)
                            break
                        else:
                            res.append([p])
                            break
                if dummy:
                    break

            if not any(p in c for c in res):
                res.append([p])

    return res

def onBoundary(points_dict,pnt,eps=5):
    S = set(list(sum(points_dict.keys(),())))
    inf = min(S)
    sup = max(S)

    mins = set()
    maxs = set()

    for i in range(0,eps+1):
        mins.add(inf+i)
        maxs.add(sup-i)

    if pnt[0] in mins or pnt[1] in mins:
        return True
    if pnt[0] in maxs or pnt[1] in maxs:
        return True

    return False

def nextEdge(d, edge, eps=0.2, length_sensitive = True):

    for e, l in d.items():
        if e == edge:
            continue
        if length_sensitive:
            if e[0] in set(edge) and l*(1-eps) <= d[edge][1] <= l*(1+eps):
                return e
        else:
            if e[0] in set(edge):
                return e

    return None

def isAssigned(d, edge):
    return (d[edge][0] != (-1, -1))

def isConnected(e1, e2):
    return (len(set(e1) & set(e2)) > 0)

def inClass(point,list):
    for elem in list:
        if isConnected(point,elem):
            return True

    return False

def chooseUnassignedEdge(d):
    for key, value in d.items():
        if d[key][0] == (-1, -1):
            return key
    return None

def chooseLonelyEdge(d):
    for key, value in d.items():
        if d[key][0] == (-1,-1):
            key_neighbors = getNextEdges(d, key)
            key_neighbors_test = getNextEdges(d, key, False)
            if key_neighbors == key_neighbors_test:
                return key

    return None

def assignRepresentative(d, edge, point):
    d[edge][0] = point
    return d
