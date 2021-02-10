def FitFile_To_DF():
    java = r"C:\Program Files\Common Files\Oracle\Java\javapath\java.exe"
    FitCSVTool_jar = r"C:\Users\silas.frantz\Desktop\FitSDKRelease_21.47.00\java\FitCSVTool.jar"
    java_CMD = [java, "-jar", FitCSVTool_jar, "-b", FIT, CSV]
    subprocess.call(java_CMD, stdout=subprocess.DEVNULL) #suppress print messages
    df = pd.read_csv(CSV)
    return df.loc[
        (df['Field 1'] == "timestamp") &
        (df['Field 2'] == "position_lat") &
        (df['Field 3'] == "position_long") &
        (df['Value 1'] != 1)
        ][["Local Number", "Value 1", "Value 2", "Value 3"]]


def EnterCorrectionsToFitFile(TrackWorkout):

    def EditCsv():
        def DeleteUnknowns():
            i_msg = H.index("Message")
            col_Indices = [i_msg] + [i for i, c in enumerate(H) if "Field" in c]
            for i, r in reversed(list(enumerate(ROWS))):
                for i_col in col_Indices:
                    try:
                        if ROWS[i][i_col]=="unknown":
                            del ROWS[i]
                            break
                    except IndexError:
                        break
        R = csv.reader(open(CSV))
        ROWS = list(R)
        H = ROWS[0]
        i_lon = H.index("Value 3")
        i_lat = H.index("Value 2")
        for lon, lat, ts, index in TrackWorkout:
            i = int(index)
            lon = str(round(lon))
            lat = str(round(lat))
            ROWS[i][i_lon] = lon
            ROWS[i][i_lat] = lat
        DeleteUnknowns()
        W = csv.writer(open(CSV, "w", newline=""))
        W.writerows(ROWS)

    def Csv_To_FitFile():
        java = r"C:\Program Files\Common Files\Oracle\Java\javapath\java.exe"
        FitCSVTool_jar = r"C:\Users\silas.frantz\Desktop\FitSDKRelease_21.47.00\java\FitCSVTool.jar"
        java_CMD = [java, "-jar", FitCSVTool_jar, "-c", CSV, FIT.replace(".FIT", "3.FIT")]
        subprocess.call(java_CMD, stdout=subprocess.DEVNULL)

    EditCsv()
    Csv_To_FitFile()


def Semicircles_to_DecimalDegrees(scArray):
    return scArray * (180 / 2 ** 31)

def DecimalDegrees_to_Semicircles(ddArray):
    return ddArray * (2 ** 31 / 180)

def DecimalDegrees_to_Utm(ddArray): # used so the track dims will be in meters
    utmArrays = utm.from_latlon(ddArray[:,1:2], ddArray[:,0:1])
    return np.column_stack(( utmArrays[0], utmArrays[1] )), utmArrays[2:]

def Utm_to_DecimalDegrees(utmArray, zone):
    ddArrays = utm.to_latlon(utmArray[:,0:1], utmArray[:,1:2], zone[0], zone[1])
    return np.column_stack(( ddArrays[1], ddArrays[0] ))


def Dataframe_to_UtmArray():
    lon, lat, ts, I = "Value 3", "Value 2", "Value 1", "I"
    df[I] = df.index + 1
    TS = df[[ ts, I ]].to_numpy()
    XY_sc = df[[ lon, lat ]].to_numpy()
    XY_dd = Semicircles_to_DecimalDegrees(XY_sc)
    XY_utm, UtmZone = DecimalDegrees_to_Utm(XY_dd)
    X, Y = XY_utm[:,0:1], XY_utm[:,1:]
    X_min, Y_min = X.min(), Y.min()
    XY_utm_adjusted = np.hstack(( X-X_min, Y-Y_min, TS))
    return XY_utm_adjusted, X_min, Y_min, UtmZone


def XY_BackTo_Semicircles(utmArray, X_min, Y_min, UtmZone):
    non_spatial_attributes = utmArray[:, 2:]
    X_utm, Y_utm = utmArray[:,0:1]+X_min, utmArray[:,1:2]+Y_min
    XY_utm = np.hstack(( X_utm, Y_utm ))
    XY_dd = Utm_to_DecimalDegrees(XY_utm, UtmZone)
    XY_sc = DecimalDegrees_to_Semicircles(XY_dd)
    return np.hstack(( XY_sc, non_spatial_attributes ))


def Cluster(input_array, min_cluster_size):
    C = hdbscan.HDBSCAN(min_cluster_size).fit(input_array)
    count_unique_clusters = C.labels_.max() - C.labels_.min()
    results = np.column_stack(( C.labels_, C.probabilities_ ))
    results_input_array = np.column_stack(( results, input_array ))
    return count_unique_clusters, results_input_array


def PlotEllipse(x, xMin, xMax, yMin, yMax, ellipseArray, n, n_tot, trackArray):
    # set colors by cluster number
    color_palette = sns.color_palette('deep', n_tot)
    cluster_colors = [color_palette[x] if x >= 0
                    else (0.5, 0.5, 0.5)
                    for x in labels]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                            zip(cluster_colors, probabilities)]
    # plot run points and ellipse of best fit
    plt.scatter(*ellipseArray.T, s=1)
    plt.scatter(*trackArray.T, s=1)
    plt.scatter(*activity_point_array.T, s=1, c=cluster_member_colors, alpha=0.5)
    plt.axis('scaled')
    plt.ylim((yMin, yMax))
    plt.xlim((xMin, xMax))

    # save
    plt.savefig(r"C:\Users\silas.frantz\Desktop\_Strava\track_ellipse_{}.png".format(n))


def GeneratePointsAlongEllipse(coeffs, xMin, xMax):
    # solve for y and returnnp array of pairs for each non-imaginary solution
    # to the equation of the ellipse while x ranges from cluster's xMin to xMax
    a, b, c, d, e = coeffs

    xStep = (xMax - xMin) / 1000
    x = xMin - xStep
    pointsAlongEllipse = []

    while x < xMax - xStep:
        x = x + xStep
        linalgCoeffs = [
            c, # y_squared
            b * x + e, # y
            a * x ** 2 + d * x - 1, # constant
        ]
        solution = np.roots(linalgCoeffs)
        solution = solution[np.isreal(solution)]
        if solution.size>0:
            y1, y2 = solution[0], solution[1]
            pointsAlongEllipse.append([x, y1])
            pointsAlongEllipse.append([x, y2])

    return np.array(pointsAlongEllipse)


def Generate400mTrack(ellipseArray, debugging=False):

    def PointOfMaxDistance(fromPT):
        fromX, fromY = fromPT
        mX, mY, mDist = None, None, 0
        for x, y in ellipseArray:
            distance = ((x - fromX) ** 2 + (y - fromY) ** 2) ** (1/2)
            if distance > mDist:
                mDist, mX, mY = distance, x, y
        return (mX, mY), mDist

    def MinDistance(fromPT):
        fromX, fromY = fromPT
        mDist = float("inf")
        for x, y in ellipseArray:
            distance = ((x - fromX) ** 2 + (y - fromY) ** 2) ** (1/2)
            if distance < mDist:
                mDist = distance
        return mDist

    def GeneratePointsAlongCircle(center, radius, start_theta):
        circle_points = []
        curve_length = round(Pi * radius)
        theta_step = Pi / curve_length
        theta = start_theta
        while theta < start_theta + Pi:#2 * Pi:
            x1 = center[0] + radius * np.cos(theta)
            y1 = center[1] + radius * np.sin(theta)
            circle_points.append([x1, y1])
            theta += theta_step
        return np.array(circle_points)

    def GeneratePointsAlongStraightaway(start, finish, theta):
        straightaway_points = []
        sX, sY = start
        fX, fY = finish
        straightaway_length = round(((fX - sX)**2 + (fY - sY)**2)**(1/2))
        distance = 0
        while distance < straightaway_length:
            sX = sX - np.cos(theta)
            sY = sY - np.sin(theta)
            straightaway_points.append([sX, sY])
            distance += 1
        return np.array(straightaway_points)

    Pi = np.pi
    # Find ellipse length to width ratio
    eX, eY = ellipseArray[:,0:1],ellipseArray[:,1:]
    cent = (eX.mean(), eY.mean())
    end1, eL_half = PointOfMaxDistance(fromPT=cent)
    end2, eL = PointOfMaxDistance(fromPT=end1)
    eW_half = MinDistance(fromPT=cent)
    # empirical adjustment (ratio of track length-to-width to ellipse length-to-width)
    L_to_W = 1.171291534 * eL_half / eW_half
    # below equation derived in merrill lynch notebook
    # parameterizing length_to_width in terms of the track radius (r) = W / 2
    # and known track circumference (C) = 400 = 2 * Pi * r + 2 * (L - 2r)
    # Track radius and distance of 'defining circles' from track center
    r_T = 200 / (2*(L_to_W - 2) + 2 + Pi)
    W = 2 * r_T
    L = 2*W + 200 - 2*r_T - Pi * r_T
    centerDefCircle_DistanceFromTrackCenter = (L / 2) - r_T
    if debugging:
        print(" - track center:  {}\n - width, length: {}, {}\n - defC_R, defCf_offset: {}, {}".format(
            (round(cent[0], 2), round(cent[1], 2)),
            round(W, 2),
            round(L, 2),
            round(r_T, 2),
            round(centerDefCircle_DistanceFromTrackCenter, 2),
            )
        )
    m = (cent[1] - end1[1]) / (cent[0] - end1[0])
    r = centerDefCircle_DistanceFromTrackCenter
    # formula derived in merrill lynch notebook
    # solving [r = sqrt(changeX^2 + changeY^2)] for changeX based on [m = changeY / changeX]
    changeX1 = r * (1 + m ** 2) ** (-1/2)
    changeY1 = m * changeX1
    C1 = (cent[0] + changeX1, cent[1] + changeY1)
    C2 = (cent[0] - changeX1, cent[1] - changeY1)
    if debugging:
        print(" - center def circle 1: {}\n - center def circle 2: {}".format(C1, C2))
    # Build an array of points along track...
    track_angle = np.arctan(m)
    # Curves
    Curve1 = GeneratePointsAlongCircle(center=C1, radius=r_T, start_theta=track_angle-Pi/2)
    Curve2 = GeneratePointsAlongCircle(center=C2, radius=r_T, start_theta=track_angle+Pi/2)
    #Straightaways
    Start_Finish, One_Hundo = Curve1[0], Curve1[-1]
    Two_Hundo, Three_Hundo = Curve2[0], Curve2[-1]
    Back_Straight = GeneratePointsAlongStraightaway(One_Hundo, Two_Hundo, track_angle)
    Home_Straight = GeneratePointsAlongStraightaway(Three_Hundo, Start_Finish, track_angle+Pi)
    # Combine to create track!
    return np.concatenate((Curve1, Back_Straight, Curve2, Home_Straight))


def SnapClusterToTrack(workout, track):

    def SnapToTrackNode():
        # returns index of closest coordinates on track
        distance_to_track = float("inf")
        steps, track_position = 0, 0
        bestTx, bestTy = None, None
        for Tx, Ty in track:
            steps += 1
            distance = ((Wx - Tx) ** 2 + (Wy - Ty) ** 2) ** (1/2)
            if distance < distance_to_track:
                distance_to_track = distance
                track_position = steps
                bestTx, bestTy = Tx, Ty
        return track_position, bestTx, bestTy

    def TrackNodesTraversed():
        # returns number of track nodes traversed
        nodes_traversed = 0
        forward, backward = last_track_position, last_track_position
        while forward != track_position and backward != track_position:
            nodes_traversed+=1
            forward+=1
            backward-=1
            if forward>len(track): forward=1
            if backward<1: backward=len(track)
        return nodes_traversed

    SnappedWorkout = [] #index: (ts, track_Y, track_X)
    splits = {} #lap: (time_s, distance_m, 400m_pace)
    lap_index, lap_dist, lap_time = 0, 0, 0
    track_step = 400 / len(track)
    last_track_position, last_ts = None, None

    for Wx, Wy, ts, I in workout:
        if last_ts is None: last_ts = ts
        time_delta = ts - last_ts#.total_seconds()
        track_position, track_X, track_Y = SnapToTrackNode()
        SnappedWorkout.append([track_X, track_Y, ts, I])
        if last_track_position is None: last_track_position = track_position
        steps_taken = TrackNodesTraversed()
        # record this advance
        if time_delta > 10:
            # this is the start of a new lap
            # save last available record as an intermediate split
            splits[lap_index] = (
                lap_time,
                "{}m".format(round(lap_dist)),
                round(400*lap_time/lap_dist, 1)
            )
            lap_index+=1
            lap_time, lap_dist = 0, 0
        else:
            lap_time += time_delta
            lap_dist += (steps_taken * track_step)
        last_ts = ts 
        last_track_position = track_position
        #print("dist {}, time {}".format(round(lap_dist, 1), round(lap_time, 1)))
        #if lap_dist >= 400:
        #    save this lap
        #    splits[lap_index] = (lap_time, "400m", lap_time)
        #    lap_index+=1
        #    lap_time = 0 # for improved accuracy, figure out some remainder from when the runner crossed the line
        #    lap_dist = lap_dist - 400
    if lap_dist != 0 and lap_time != 0: splits[lap_index] = (
        lap_time,
        "{}m".format(round(lap_dist)),
        round(400*lap_time/lap_dist, 1)
    )
    #JsonPrettyPrint(splits)
    return np.array(SnappedWorkout)

            
def FitTrackToCluster(a):
    def LeastSquaresEllipse():
        # s[0]*x^2 + s[1]*xy + s[2]*y^2 + s[3]*x + s[4]*y - 1 = 0
        A = np.hstack([X**2, X * Y, Y**2, X, Y])
        b = np.ones_like(X)
        return np.linalg.lstsq(A, b, rcond=1)[0].squeeze()
    X = a[:,0:1]
    Y = a[:,1:2]
    s = LeastSquaresEllipse()
    ellipseArray = GeneratePointsAlongEllipse(s, X.min(), X.max())
    return Generate400mTrack(ellipseArray)


def TracklikeCluster(n, clusters):
    ##print("\n{} clusters discovered...".format(n))
    # ahh... just realized you would never go to two tracks...
    # so see if there is a way to conclude this while statement (as a function)
    # returning False if no candidate cluster checks out, and the best cluster if one does
    while n > 0: # -1's are unclassed outliers
        n = n - 1
        cluster = clusters[
            np.where( # current cluster ID and probability .9+
                (clusters[:, 0]==n) * (clusters[:, 1] > 0.89) 
                )
        ][:, 2:]
        # test to see the quality of ellipse fit for the cluster
        #print("\ncluster index {} contains {} points".format(n, len(cluster)))
        if len(cluster)>500:
            return True, cluster 
    return False, None


# MAIN
import time

for timetest in range(10):

    start = time.time()

    import sys, os, subprocess, csv
    import pandas as pd, numpy as np
    import utm, hdbscan
    import seaborn as sns, matplotlib.pyplot as plt
    sys.path.append(r'\\ace-ra-fs1\data\GIS\_Dev\python\apyx')
    from apyx import JsonPrettyPrint

    FIT = r"C:\Users\silas.frantz\Desktop\_Strava\Wkt\FIT_TEST.FIT"
    CSV = FIT.lower().replace(".fit", ".csv")

    df = FitFile_To_DF()
    UtmArray, X_min, Y_min, UtmZone = Dataframe_to_UtmArray()
    n_clusters, clusters = Cluster(UtmArray[:, :2], min_cluster_size=50)
    clusters = np.hstack(( clusters, UtmArray[:, 2:] )) #["X", "Y", "label", "prob", "ts", "index"]
    tracklike_cluster_found, cluster = TracklikeCluster(n_clusters, clusters)
    if not tracklike_cluster_found:
        print("No tracklike XY cluster detected in this activity.")
        #return original fit file
    trackArray = FitTrackToCluster(cluster)
    snappedCluster = SnapClusterToTrack(cluster, trackArray)
    TrackWorkout = XY_BackTo_Semicircles(snappedCluster, X_min, Y_min, UtmZone)

    FIT_Snapped = EnterCorrectionsToFitFile(TrackWorkout)

    print("{}s elapsed".format(round(time.time()-start, 3)))

#return FIT_Snapped


#print(snappedWorkout)

# integrate and return new fit file
# convert this array[:, 1:3] back to...
    # UTM (unadjust, adding back Xmin and Ymin)
    # DD (using the converter)
    # SC (using the formula)
# then use the array to edit the csv
# go through those columns in the csv and try: except 
# looking up cooresponding index values from the semicircle array
# then write the CSV back to fit format
# turns out i only needed the timestamps to truth that this is working...
# plot with both the PlotEllipse function (rename) and the 
#labels, probs = clusters[:, 0], clusters[:, 1]
#PlotEllipse(x, xMin, xMax, yMin, yMax, ellipseArray, n, n_clusters, trackArray)

# RESOURCES
# https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
# https://scikit-learn.org/stable/modules/clustering.html
# https://mathworld.wolfram.com/Ellipse.html

# installed utm: https://github.com/Turbo87/utm 
# installed JDK: https://www.oracle.com/java/technologies/javase-jdk15-downloads.html 
# installed FIT SDK: https://developer.garmin.com/fit/download/

# 2021-02-08 notes to continue...
# try writing a csv back to a fit file
    # first the same one, then the same one with a slight modification (e.g. just one location digit)
# figure out how to enter changes without changing the csv format in any way
# generate a line graph of the workout and pace (like strava does)
# do this from the initial array, then snap to track and look at the final array
# should be apparent immediately if there are big differences
# if not, nice!?, try doing an area under the curve - plotting library?

# finally, if everything checks out, go back and...
    # do a workout and see if you can upload it to strava!
        # claim the win in a post and send out emails to...
        # strava, garmin, apple, nike, other wearable tech folks
    # see if you can test ellipse match strength
    # and get ellipse area, to further weed out clusters
    # (500 points doesnt make sense, happened to work here)