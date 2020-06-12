from ClassifierMain import *
from numpy import nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
from functools import reduce
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

def radialRefinement(buckets,radius=10): #don't try to do anything clever here, just brute force the whole thing

    bp = [p for c in buckets for p in c]
    res = [[p] for p in bp]

    #first pass
    for l in res:

        if len(l) == 1:
            pxy = l[0]
            temp = [x for x in bp if x != pxy]
            for p in temp:
                if inRadius(pxy,p,radius):
                    l.append(p)

            if len(l) == 1:
                res.remove(l)

    # grouping
    for l in res:

        temp = [b for b in res if b != l]
        for t in temp:
            if len(set(l).intersection(set(t))) > 0:
                l.extend(t)
                res.remove(t)
            # add a stop condition here for safety?
            # else:
            #    break

    result = []
    for l in res:
        s = set(l)
        result.append(s)


    for s in result:
        temp = [y for y in result if y != s]
        for r in temp:
            if r.issubset(s):
                result.remove(r)

    return result

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

def fillBorder(s_buckets):

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

        p1,q1 = dc[0]
        a,b = dc[1]

        x1,y1 = (2*p1-a,2*q1-b)
        dc.insert(0,(x1,y1))

        p2,q2 = dc[-1]
        c,d = dc[-2]

        x2,y2 = (2*p2-c,2*q2-d)
        dc.append((x2,y2))

        res.append(dc)

    #now, make a prediction about the next point assuming that the endpoint
    #is the next midpoint

    return res
#------------------------------------------------------MAIN SCRIPT---------------------------------------------------------

dir = "/Users/rohanrele/Documents/research/Polynomiography/Classifier/images/"
file = "HZOJ5HU6G7.png"
test_file = "D1W68LOJY7.png"
input = dir + test_file


name, tri = triangulation(input, n=500)
p = tri.points
q = tri.vertices
r = generateEdgeDict(p, q)

op = (350,0)
point_test = generatePointDict(r)
#print(point_test)
border_test = densityClasses2(point_test)
#border_test = densityClasses3(point_test,origin=op)

res = radialRefinement(border_test,radius=25)
print("Length:" + str(len(res)))
print(res)

myPlot3(p,res)

s1 = naiveMerge(res)
print("Length Step 1: " + str(len(s1)))
s2 = naiveMerge(s1)
print("Length Step 2: " + str(len(s2)))
s3 = naiveMerge(s2)
print("Length Step 3: " + str(len(s3)))
s4 = naiveMerge(s3)
print("Length Step 4: " + str(len(s4)))
s5 = naiveMerge(s4)
print("Length Step 5: " + str(len(s5)))
s6 = naiveMerge(s5)
print("Length Step 6: " + str(len(s6)))
s7 = naiveMerge(s6)
print("Length Step 7: " + str(len(s7)))
s8 = naiveMerge(s7)
print("Length Step 8: " + str(len(s8)))

myPlot3(p,s8)

sortd = naiveSort(s8)
print(sortd)

brr = []
for t in sortd:
    brr.append((t,t[0],t[-1]))

myPlot2(p,brr)

filled = fillBorder(sortd)

temp = [[item[0],item[1]] for sublist in filled for item in sublist]
new_p = set(tuple(i) for i in p.tolist()).union(set(tuple(t) for t in temp))

myPlot3(new_p,filled)

post_fill = radialRefinement(filled,radius=30)
s1 = naiveMerge(post_fill)
print("Length Step 1: " + str(len(s1)))
s2 = naiveMerge(s1)
print("Length Step 2: " + str(len(s2)))
s3 = naiveMerge(s2)
print("Length Step 3: " + str(len(s3)))
s4 = naiveMerge(s3)
print("Length Step 4: " + str(len(s4)))
s5 = naiveMerge(s4)
print("Length Step 5: " + str(len(s5)))
s6 = naiveMerge(s5)
print("Length Step 6: " + str(len(s6)))
s7 = naiveMerge(s6)
print("Length Step 7: " + str(len(s7)))
s8 = naiveMerge(s7)
print("Length Step 8: " + str(len(s8)))

myPlot3(new_p,s8)

brr = []
s8s = naiveSort(s8)
for t in s8s:
    brr.append((t,t[0],t[-1]))

myPlot2(new_p,brr)
