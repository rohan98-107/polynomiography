from ClassifierMain import *
from numpy import nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform, directed_hausdorff
from functools import reduce
from time import clock
from matplotlib.path import Path
import operator


def getAssignedNeighbors(b,point_dict,buckets):

    if len(b) == 1:
        neighbors = []
        nbuckets = []
        val = point_dict[b[0]]
        connected = [y[0] for y in val]
        for point in connected:
            for bucket in buckets:
                if point in bucket:
                    neighbors.append(point)
                    nbuckets.append(bucket)

        return neighbors, nbuckets

    else:
        return None


def refineSingletons(buckets,point_dict):

    for bucket in buckets:
        if len(bucket) == 1:

            P = bucket[0]
            assigned_neighbors,c_buckets = getAssignedNeighbors(bucket,point_dict,buckets)
            if len(assigned_neighbors) == 0:
                buckets.remove(bucket)
            elif len(assigned_neighbors) == 1:
                buckets.remove(c_buckets[0])
                c_buckets[0].append(P)
                buckets.remove(bucket)
                buckets.append(c_buckets[0])

            elif len(assigned_neighbors) == 2:
                x1,y1 = assigned_neighbors[0]
                x2,y2 = assigned_neighbors[1]

                M = ( (x1+x2)/2, (y1+y2)/2 )
                r = math.sqrt((M[0]-x1)**2 + (M[1]-y1)**2)
                if math.sqrt((M[0]-P[0])**2 + (M[1]-P[1])**2) <= r:
                    buckets.remove(c_buckets[0])
                    buckets.remove(c_buckets[1])
                    buckets.remove(bucket)

                    new_bucket = c_buckets[0]
                    new_bucket.extend(c_buckets[1])
                    new_bucket.append(P)

                    buckets.append(new_bucket)
            else:
                # might replace this with some clever endpoint stuff once that gets done
                min_dist = 10000000
                closest_bucket = []
                for b in c_buckets:

                    for point in b:
                        d = math.sqrt((point[0]-P[0])**2 + (point[1]-P[1])**2)
                        if d <= min_dist:
                            min_dist = d
                            closest_bucket = b

                buckets.remove(closest_bucket)
                buckets.remove(bucket)

                closest_bucket.append(P)
                buckets.append(closest_bucket)

    return buckets

def inNeighborhood(p1,p2,radius):
    return math.sqrt((p1[1]-p2[1])**2 + (p1[0]-p2[0])**2) < radius

def moreraRefinement(points_dict,buckets,radius=10,merge=False):

    bp = [p for c in buckets for p in c]
    res = []

    for c in buckets:
        list_of_sets = []
        for p in c:
            Sp = set()
            val = points_dict[p]
            for coord,dist in val:
                if dist < radius and coord in bp:
                    Sp.add(coord)

            if len(Sp) > 0:
                list_of_sets.append(Sp)

        RC = set().union(*list_of_sets) #refined class
        bad = set(c).difference(RC)

        # we can go two routes here: (1) merging or (2) back-refining

        if merge:
            res.append(RC)

        else:
            #(2) - safer option
            if len(bad) > 0:
                res.append(list(RC.intersection(set(c))))
                for badpoint in bad:
                    res.append([badpoint])
            else:
                res.append(c)

    if merge:
        i = 0
        while len(res) >= 1 and i < len(res)-1:
            if res[i].isdisjoint(res[i+1]):
                i += 1
                continue
            elif len(res[i]) == 0:
                i+= 1
                res.remove(res[i])
            else:
                res.append(res[i].union(res[i+1]))
                res.remove(res[i])
                res.remove(res[i+1])
                i = 0

        # this is kind of dangerous since it could continue infinitely when there is a lot of overlap...
        # ...but who cares lets just try it

    return res

def inRadius(tup1,tup2,R):

    x1,y1 = tup1
    x2,y2 = tup2

    if (y2-y1)**2 + (x2-x1)**2 < R**2:
        return True
    else:
        return False

def radialRefinement(buckets,radius=10):

    bp = [p for c in buckets for p in c]
    res = [set([p]) for p in bp]

    #first pass
    for l in res:

        if len(l) == 1:
            pxy = list(l)[0]
            for p in bp:
                if inRadius(pxy,p,radius) and p != pxy:
                    l.add(p)

            if len(l) == 1:
                res.remove(l)

    # grouping
    for l in res:
        for t in res:
            if l & t and t != l:
                res.remove(t)
                l.update(t)

    for s in res:
        for r in res:
            if r.issubset(s) and r != s:
                res.remove(r)

    return res


def findEndpoints(buckets,eps=math.sqrt(2),origin=(0,0)):

    result = []
    PI = math.pi
    ox, oy = origin
    for bucket in buckets:

        angles = []
        bucket = list(bucket)
        init_point = bucket[0]
        mag = math.sqrt((init_point[0]-ox)**2 + (init_point[1]-oy)**2)

        for point in bucket:
            nm = math.sqrt((point[0]-ox)**2 + (point[1]-oy)**2)
            if mag * (2-eps) <= nm <= mag * eps:
                angles.append(math.atan2(point[1]-oy,point[0]-ox)+2*PI)
                mag = nm
            else:
                angles.append(-2*PI)

        # we could try from the middle of the image, but it will yield the same issue

        # if we try from the middle of the image, then we need to change this as well because
        # ...not everything will be positve then

        mindex = angles.index(min(a for a in angles if a >= 0))
        maxdex = angles.index(max(a for a in angles if a >= 0))

        minpoint = bucket[mindex]
        maxpoint = bucket[maxdex]

        # remove endpoints from bucket?

        result.append((bucket,minpoint,maxpoint))

    return result


def findEndpoints2(buckets):
    result = []
    for bucket in buckets:
        bucket = list(bucket)
        temp = bucket
        map(list,temp)
        if len(bucket) >= 3:
            D = pdist(temp)
            D = squareform(D)
            N, [I_row, I_col] = nanmax(D), unravel_index( argmax(D), D.shape )
            result.append((bucket,bucket[I_row],bucket[I_col]))
        elif len(bucket) == 2:
            result.append((bucket,bucket[0],bucket[1]))
        else:
            result.append((bucket,bucket[0],bucket[0]))

    return result

def naiveMerge(buckets,sensitivity=1):

    for s in buckets:
        temp = [b for b in buckets if b != s]
        for ss in temp:
            i = 0
            if len(set(ss).intersection(set(s))) >= sensitivity:
                buckets.append(set(s).union(set(ss)))
                buckets.remove(ss)
                i += 1

        if i == 0 and len(s) > 1:
            continue
        else:
            buckets.remove(s)


    buckets.sort()
    ret = list(buckets for buckets,_ in itertools.groupby(buckets))
    return ret

def naiveSort(buckets):
    sorted_buckets = []
    for coords in buckets:
        temp = findEndpoints2([coords])
        tup = temp[0]
        points = set(tup[0])
        e1 = tup[1]
        e2 = tup[2]

        curr = e1
        path = [curr]
        points.remove(e1)

        while points:
            curr = min(points,key=(lambda p: (curr[0]-p[0])**2 + (curr[1]-p[1])**2))
            points.remove(curr)
            path.append(curr)

        sorted_buckets.append(path)

    return sorted_buckets

def sequentialMerge(buckets,s_array=None):

    temp = buckets
    complete = False
    indx = 0
    while not complete:
        if not s_array: #crude solution but we can make it nicer later
            temp = naiveMerge(temp)
        else:
            temp = naiveMerge(temp,s_array[indx])
            indx += 1

        L = len(temp)
        i = 0
        for s in temp:
            others = [x for x in temp if x != s]
            if s.isdisjoint(set().union(*others)):
                i += 1

        complete = (L == i)

    return temp

def fillBorder(s_buckets,extend=True):

    res = []
    for c in s_buckets:
        i = 0
        dc = [c[0]]
        while i + 1 < len(c):
            x1,y1 = c[i]
            x2,y2 = c[i+1]

            M = ((x1+x2)/2,(y1+y2)/2)
            dc.append(M)
            dc.append((x2,y2))
            i += 1

        if extend:
            p1,q1 = dc[0]
            a,b = dc[1]

            x1,y1 = (2*p1-a,2*q1-b)
            dc.insert(0,(x1,y1))

            p2,q2 = dc[-1]
            c,d = dc[-2]

            x2,y2 = (2*p2-c,2*q2-d)
            dc.append((x2,y2))

        res.append(dc)

    return res

def findCentroids(buckets, size=False, sig=False, sigsize=5):
    result = []
    for c in buckets:
        if c:
            if not sig:
                K = len(c)
                exes, whys = zip(*c)
                result.append((c, (sum(exes) / K, sum(whys) / K)))
            else:
                if len(c) >= sigsize:
                    K = len(c)
                    exes, whys = zip(*c)
                    result.append((c, (sum(exes) / K, sum(whys) / K)))
                else:
                    if size:
                        result.append(([], None))
        else:
            if size:
                result.append(([], None))

    return result

def createHulls(buckets):
    hulls = []
    temp = [np.array([np.array(x) for x in b]) for b in buckets]

    for t in temp:
        if t.shape[0] <= 5:
            hulls.append(None)
        else:
            hulls.append(ConvexHull(t,'Qg'))

    return hulls

def findRoots_convexHull(buckets_with_centers,dist_eps=5,num_fills=3):

    result = []
    centers = []
    buckets = []
    for b,c in buckets_with_centers:
        centers.append(c)
        buckets.append(b)

    hulls = createHulls(buckets)
    temp = np.array([np.array([np.array(x) for x in b]) for b in buckets])
    i = 0
    H = len(hulls)

    while i < H:
        if hulls[i] is not None:
            approx = []
            for c in centers:
                hull_vertices = temp[i][hulls[i].vertices]
                hull_vertices = hull_vertices.tolist()
                hull_vertices = [tuple(x) for x in hull_vertices]
                filled = [hull_vertices]

                for j in range(0,num_fills):
                    filled = fillBorder(filled,extend=False)

                arr_filled = [np.array([np.array(f) for f in r]) for r in filled][0]
                temp_path = Path(arr_filled)
                orig_path = Path(temp[i])
                dist, _, _ = directed_hausdorff(arr_filled,temp[i])

                if dist <= dist_eps and temp_path.contains_point(np.array(c)):
                    approx.append(c)

            if approx:
                A = len(approx)
                xs, ys = zip(*approx)
                result.append((sum(xs)/A,sum(ys)/A))

        i += 1


    return set(result)

def findRoots_naive(buckets_with_centers,radius):

    result = []
    centers = []
    for _ ,c in buckets_with_centers:
        centers.append(c)

    for c in centers:
        temp = [x for x in centers if x != c]
        temp2 = []
        for t in temp:
            if (c[0]-t[0])**2 + (c[1]-t[1])**2 <= radius**2:
                temp2.append(t)

        temp2.append(c)
        if len(temp2) > 1:
            K = len(temp2)
            exes, whys = zip(*temp2)
            result.append((sum(exes)/K,sum(whys)/K))

    for r in result:
        temp = [y for y in result if y != r]
        for t in temp:
            if (r[0]-t[0])**2 + (r[1]-t[1])**2 <= radius**2:
                result.remove(t)

    return set(result)

def generateEquation(roots):
    return lambda x: np.prod([x - (r[0] + r[1]*1j) for r in roots])
