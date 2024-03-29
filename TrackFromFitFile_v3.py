def FitFile_To_DF(FIT):
    java = r"C:\Program Files\Common Files\Oracle\Java\javapath\java.exe"
    java_CMD = [java, "-jar", FitCSVTool_jar, "-b", FIT, CSV]
    subprocess.call(java_CMD, stdout=subprocess.DEVNULL) #suppress print messages
    df = pd.read_csv(CSV)
    return df.loc[
        (df['Field 1'] == "timestamp") &
        (df['Field 2'] == "position_lat") &
        (df['Field 3'] == "position_long") &
        (df['Value 1'] != 1)
        ][["Value 1", "Value 2", "Value 3", "Value 5"]] # ts, lat, lon, m_per_s


def EnterCorrectionsToFitFile(TrackWorkout, Track):

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
        def XY_TrackPosition(p):
            return Track[:, :][p]
        def NeedToInterpolateCurve(): # between this point and next point
            next_ = TrackWorkout[TrackWorkout[:, 0]==index-1]
            position_next = next_[:, 2]
            if position == position_next: return False, None, None, None
            c_s_next, ts_next = next_[:, 3], next_[:, 1]
            if ts - ts_next > 20:
                return False, None, None, None
            if c_s == 0 or c_s_next == 0: # if this point occurs on a curve or next point is on a curve
                if c_s_next.size==0: # next point is not adjacent - means there is some kind of row between
                    n_ = 2
                    #while n_< 5:  # if one of the following 5 rows are good to go that's fine just use them :D
                    #    n_+=1 
                    next__ = TrackWorkout[TrackWorkout[:, 0]==index-n_]
                    c_s_next, ts_next = next__[:, 3], next__[:, 1]
                    if ts-ts_next > 20: # repeat timestamp check for other interp candidates
                        return False, None, None, None
                    if len(next__) > 0:
                        return True, next__, c_s_next, ts_next
                    return False, None, None, None
                return True, next_, c_s_next, ts_next
            return False, None, None, None
        def InterpolateCurve():
            #print("{}: ({})".format(
            #    "Curve" if c_s == 0 else "Straight", NeedToInterpolateCurve()))
            position_next = int(next_[:, 2])
            Steps, Direction = TrackNodesTraversed(position, position_next, len(Track))
            PointsBetween = Steps - 1
            interpolationRow = []
            for data_index, data_val in enumerate(ROWS[i]):
                try:
                    numerical_data_val = float(data_val)
                    numerical_type = len(data_val.split(".")[1]) if "." in data_val else 0
                    next_row_numerical_val = float(ROWS[i-1][data_index])
                    interpolationRow.append(((next_row_numerical_val - numerical_data_val)/Steps, numerical_type))
                except ValueError:
                    interpolationRow.append(data_val) # field label text or whatever, just dont mess with it
            start_row, track_node = ROWS[i], position
            while_loops = 0
            while PointsBetween > 0:
                new_row = []
                for value_index, new_value in enumerate(interpolationRow):
                    if isinstance(new_value, tuple):
                        round_to = new_value[1]
                        new_value = float(start_row[value_index]) + (new_value[0])*while_loops
                        new_value = int(round(new_value)) if round_to == 0 else round(new_value, round_to)
                    new_row.append(str(new_value))
                if Direction=="+": track_node += 1
                else: track_node -= 1
                if track_node < 0: track_node = len(Track)-1
                if track_node > len(Track)-1: track_node = 0
                interp_lon, interp_lat = XY_TrackPosition(track_node)
                new_row[i_lon] = str(round(interp_lon))
                new_row[i_lat] = str(round(interp_lat))
                lonlat = "{}.0{}.0".format(new_row[i_lon], new_row[i_lat])
                if lonlat not in positions.keys(): positions[lonlat] = track_node
                ROWS.insert(i, new_row)
                PointsBetween -= 1

        positions = {}
        R = csv.reader(open(CSV))
        ROWS = list(R)
        H = ROWS[0]
        i_lon = H.index("Value 3")
        i_lat = H.index("Value 2")

        for index, ts, position, c_s in reversed(TrackWorkout):
        # reversed allows inserting without altering indices encountered later
            i = int(index)
            position = int(position)
            lon, lat = XY_TrackPosition(position)
            ROWS[i][i_lon] = str(round(lon))
            ROWS[i][i_lat] = str(round(lat))
            lonlat = "{}.0{}.0".format(ROWS[i][i_lon], ROWS[i][i_lat])
            if lonlat not in positions.keys(): positions[lonlat] = position

            needed, next_, c_s_next, ts_next = NeedToInterpolateCurve()
            if needed: InterpolateCurve()

        DeleteUnknowns()
        W = csv.writer(open(CSV, "w", newline=""))
        W.writerows(ROWS)

        return ROWS, positions

    def Csv_To_FitFile():
        java = r"C:\Program Files\Common Files\Oracle\Java\javapath\java.exe"
        java_CMD = [java, "-jar", FitCSVTool_jar, "-c", CSV, snappedFit]
        subprocess.call(java_CMD, stdout=subprocess.DEVNULL)

    snappedFit = FIT.replace(".FIT", "_snapped.FIT")
    csvRows, allPositions = EditCsv()
    Csv_To_FitFile()

    return csvRows, snappedFit, allPositions


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


def Dataframe_to_UtmArray(df, X_min=None, Y_min=None):
    spatial_cols = ("Value 3", "Value 2")
    lon, lat = spatial_cols#, ts, I, "Value 1", "I"
    I = "I"
    nonspatial_cols = [c for c in df.columns if c not in spatial_cols]
    df[I] = df.index + 1
    nonspatial_attributes = df[ [I] + nonspatial_cols ].to_numpy()
    XY_sc = df[[ lon, lat ]].to_numpy()
    XY_dd = Semicircles_to_DecimalDegrees(XY_sc)
    XY_utm, UtmZone = DecimalDegrees_to_Utm(XY_dd)
    X, Y = XY_utm[:,0:1], XY_utm[:,1:]
    if X_min is None and Y_min is None:
        X_min, Y_min = X.min(), Y.min()
    XY_utm_adjusted = np.hstack(( X-X_min, Y-Y_min, nonspatial_attributes))
    return XY_utm_adjusted, UtmZone, X_min, Y_min


def XY_BackTo_Semicircles(utmArray, X_min, Y_min, UtmZone):
    #non_spatial_attributes = utmArray[:, 2:]
    X_utm, Y_utm = utmArray[:,0:1]+X_min, utmArray[:,1:2]+Y_min
    XY_utm = np.hstack(( X_utm, Y_utm ))
    XY_dd = Utm_to_DecimalDegrees(XY_utm, UtmZone)
    return DecimalDegrees_to_Semicircles(XY_dd)
    #return np.hstack(( XY_sc, non_spatial_attributes ))


def Cluster(input_array, min_cluster_size):
    C = hdbscan.HDBSCAN(min_cluster_size).fit(input_array)
    count_unique_clusters = C.labels_.max() - C.labels_.min()
    results = np.column_stack(( C.labels_, C.probabilities_ ))
    results_input_array = np.column_stack(( results, input_array ))
    return count_unique_clusters, results_input_array


def PlotEllipse(ellipseArray, clusterArray, trackArray): #x, xMin, xMax, yMin, yMax, 
    # clusterProbabilities,
    #cluster labels
    # set colors by cluster number
    #color_palette = sns.color_palette('deep', 2) #ntot
    #cluster_colors = [color_palette[1] if x >= 0
    #                else (0.5, 0.5, 0.5)
    #                for x in clusterProbabilities]
    #cluster_member_colors = [sns.desaturate(x, p) for x, p in
    #                        zip(cluster_colors, clusterProbabilities)]
    # plot run points and ellipse of best fit
    centerArray = np.array(
        [clusterArray[:, 0].mean(), clusterArray[:, 1].mean()]
    )

    qualifyingCluster = clusterArray[
        np.where(
            ((clusterArray[:, 0] - centerArray[0]) ** 2 + (clusterArray[:, 1] - centerArray[1]) ** 2) ** (1/2) > 60
        )
    ]
    #print(qualifyingCluster)
    plt.scatter(*ellipseArray.T)#, s=1)
    plt.scatter(*trackArray.T)#, s=1)
    #plt.scatter(*clusterArray.T)#, s=1)#, c=cluster_member_colors, alpha=0.5)
    plt.scatter(*centerArray.T)
    plt.axis('scaled')
    plt.scatter(*qualifyingCluster.T)
    plt.ylim((clusterArray[:, 1].min(), clusterArray[:, 1].max()))
    plt.xlim((clusterArray[:, 0].min(), clusterArray[:, 0].max()))
    # save
    plt.savefig(r"C:\Users\silas.frantz\Desktop\TrakCat\track_ellipse.png")


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


def Generate400mTrack(ellipseArray):

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
        curve_length = Pi * radius
        theta_step = Pi / curve_length
        theta = start_theta + theta_step / 2
        while theta < start_theta + Pi:
            x1 = center[0] + radius * np.cos(theta)
            y1 = center[1] + radius * np.sin(theta)
            circle_points.append([x1, y1])
            theta += theta_step
        return np.array(circle_points)

    def GeneratePointsAlongStraightaway(start, finish, theta):
        straightaway_points = []
        sX, sY = start
        fX, fY = finish
        straightaway_length = ((fX - sX)**2 + (fY - sY)**2)**(1/2)
        distance = 0.5 # start at half a meter from start of straightaway
        while distance < straightaway_length - 0.5: # end at least half a meter from end of straightaway
            sX = sX - np.cos(theta)
            sY = sY - np.sin(theta)
            straightaway_points.append([sX, sY])
            distance += 1
        return np.array(straightaway_points)

    Pi = np.pi
    # Find ellipse length to width ratio
    eX, eY = ellipseArray[:,0:1],ellipseArray[:,1:2] #0;1, 1:
    cent = (eX.mean(), eY.mean())
    end1, eL_half = PointOfMaxDistance(fromPT=cent)
    end2, eL = PointOfMaxDistance(fromPT=end1)
    eW_half = MinDistance(fromPT=cent)
    # empirical adjustment (ratio of track length-to-width to ellipse length-to-width) 1.171291534
    # new empirical adjustment needed ever since i discovered using the extrema... 2021-02-17 # i guess 1.5 works ok - we'll see
    # using this method - which pretty accurately sets the length - we should probably go with a known full length instead :(
    L_to_W = 1.45 * eL_half / eW_half
    # below equation derived in merrill lynch notebook
    # parameterizing length_to_width in terms of the track radius (r) = W / 2
    # and known track circumference (C) = 400 = 2 * Pi * r + 2 * (L - 2r)
    # Track radius and distance of 'defining circles' from track center
    r_T = 200 / (2*(L_to_W - 2) + 2 + Pi)
    W = 2 * r_T
    L = 2*W + 200 - 2*r_T - Pi * r_T
    centerDefCircle_DistanceFromTrackCenter = (L / 2) - r_T
    m = (cent[1] - end1[1]) / (cent[0] - end1[0])
    r = centerDefCircle_DistanceFromTrackCenter
    # formula derived in merrill lynch notebook
    # solving [r = sqrt(changeX^2 + changeY^2)] for changeX based on [m = changeY / changeX]
    changeX1 = r * (1 + m ** 2) ** (-1/2)
    changeY1 = m * changeX1
    C1 = (cent[0] + changeX1, cent[1] + changeY1)
    C2 = (cent[0] - changeX1, cent[1] - changeY1)
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
    return np.concatenate((Curve1, Back_Straight, Curve2, Home_Straight)), len(Curve1)


def TrackNodesTraversed(ORIG, DEST, MAX):
    nodes_traversed = 0
    forward, backward = ORIG, ORIG
    while forward != DEST and backward != DEST:
        nodes_traversed+=1
        forward+=1
        backward-=1
        if forward>MAX-1: forward=0
        if backward<0: backward=MAX-1
    direction = "+" if forward==DEST else "-"
    return nodes_traversed, direction


def SnapClusterToTrack(workout, track, curveLength):

    def SnapToTrackNode():
        # returns index of closest coordinates on track
        distance_to_track = float("inf")
        steps, track_position = 0, 0
        bestTx, bestTy = None, None
        for Tx, Ty in track:
            distance = ((Wx - Tx) ** 2 + (Wy - Ty) ** 2) ** (1/2)
            if distance < distance_to_track:
                distance_to_track = distance
                track_position = steps
                bestTx, bestTy = Tx, Ty
            steps += 1
        #if distance_to_track > 30: # probably don't want to snap in this case...
        #    print(distance_to_track)
        #    print(track_position)
        curve_straight = 0 if (
            track_position < curveLength or (
            track_position > 200 and track_position < 200+curveLength)
            ) else 1
        return track_position, curve_straight

    SnappedWorkout = [] #index: (ts, track_Y, track_X)
    splits = {} #lap: (time_s, distance_m, 400m_pace)
    lap_index, lap_dist, lap_time = 0, 0, 0
    track_step = 400 / len(track)
    last_track_position, last_ts = None, None

    for Wx, Wy, I, ts, m_s in workout:
        if last_ts is None: last_ts = ts
        time_delta = ts - last_ts#.total_seconds()
        track_position, curve_straight = SnapToTrackNode()
        SnappedWorkout.append([I, ts, track_position, curve_straight])
        if last_track_position is None: last_track_position = track_position
        steps_taken, direction = TrackNodesTraversed(last_track_position, track_position, MAX=len(track))
        # record this advance
        if time_delta > 10:
            # this is the start of a new lap
            # save last available record as an intermediate split
            splits[lap_index] = (
                lap_time,
                "{}m".format(round(lap_dist)),
                round(400*lap_time/lap_dist, 1) if lap_dist != 0 else 0
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
        round(400*lap_time/lap_dist, 1) if lap_dist != 0 else 0
    )
    #JsonPrettyPrint(splits)
    return np.array(SnappedWorkout)

            
def FitTrackToCluster(a):
    def LeastSquaresEllipse():
        # s[0]*x^2 + s[1]*xy + s[2]*y^2 + s[3]*x + s[4]*y - 1 = 0
        A = np.hstack([X**2, X * Y, Y**2, X, Y])
        b = np.ones_like(X)
        return np.linalg.lstsq(A, b, rcond=1)[0].squeeze()

    a_curves = a[
        np.where( # retake a sample of just the track curves to avoid overfitting to either curve
            ((a[:, 0] - a[:, 0].mean()) ** 2 + (a[:, 1] - a[:, 1].mean()) ** 2) ** (1/2) > 60
        )
    ]
    X = a_curves[:,0:1]
    Y = a_curves[:,1:2]
    s = LeastSquaresEllipse()
    ellipseArray = GeneratePointsAlongEllipse(s, X.min(), X.max())
    trackArray, curveLength = Generate400mTrack(ellipseArray)
    return trackArray, curveLength, ellipseArray


def TracklikeCluster(n, clusters):
    ##print("\n{} clusters discovered...".format(n))
    # ahh... just realized you would never go to two tracks...
    # so see if there is a way to conclude this while statement (as a function)
    # returning False if no candidate cluster checks out, and the best cluster if one does
    while n > 0: # -1's are unclassed outliers
        n = n - 1
        cluster = clusters[
            np.where( # current cluster ID and probability .9+, .5
                (clusters[:, 0]==n) * (clusters[:, 1] > probability) 
                )
        ][:, 2:]
        # test to see the quality of ellipse fit for the cluster
        #print("\ncluster index {} contains {} points".format(n, len(cluster)))
        if len(cluster)>500:
            cluster_x_min = cluster[:, 0].min()
            cluster_y_min = cluster[:, 1].min()
            cluster[:, 0] = cluster[:, 0] - cluster_x_min
            cluster[:, 1] = cluster[:, 1] - cluster_y_min
            return True, cluster, cluster_x_min, cluster_y_min
    return False, None, None, None


def StoreResults(InitialArray, TrackArray, FinalArray, InterpolatedArray):
    store = FIT.replace(".FIT", ".npz")
    np.savez(store, InitialArray, TrackArray, FinalArray, InterpolatedArray)



# MAIN
import time
start = time.time()

import sys, os, subprocess, csv
import pandas as pd, numpy as np
import utm, hdbscan
import seaborn as sns, matplotlib.pyplot as plt
sys.path.append(r'\\ace-ra-fs1\data\GIS\_Dev\python\apyx')
from apyx import JsonPrettyPrint

FitCSVTool_jar = r"C:\Users\silas.frantz\Desktop\TrakCat\FitSDKRelease_21.47.00\java\FitCSVTool.jar"

FIT = r"C:\Users\silas.frantz\Desktop\TrakCat\B2FE1101.FIT"
        #B2H90533.FIT" #keira 4x1600
        #
        #B2A84849.FIT"
        #r"C:\Users\silas.frantz\Desktop\_Strava\Wkt\FIT_TEST.FIT"
CSV = FIT.lower().replace(".fit", ".csv")
probability = 0.8

df = FitFile_To_DF(FIT)
UtmArray, UtmZone, X_min, Y_min = Dataframe_to_UtmArray(df)
n_clusters, clusters = Cluster(UtmArray[:, :2], min_cluster_size=50)
clusters = np.hstack(( clusters, UtmArray[:, 2:] )) #["X", "Y", "label", "prob", "ts", "index"]
tracklike_cluster_found, cluster, X_min_track, Y_min_track = TracklikeCluster(n_clusters, clusters)
X_min = X_min + X_min_track
Y_min = Y_min + Y_min_track
if not tracklike_cluster_found:
    print("No tracklike XY cluster detected in this activity.")
    #return original fit file
trackArray, curveLength, ellipse = FitTrackToCluster(cluster)
TrackWorkout = SnapClusterToTrack(cluster, trackArray, curveLength) # index_csv, timestamp, track_position_index, curve_or_straight
trackArray_UTM = XY_BackTo_Semicircles(trackArray, X_min, Y_min, UtmZone)
CSV_Rows, FIT_Snapped, Positions_Dict = EnterCorrectionsToFitFile(TrackWorkout, trackArray_UTM)

# prepare to visualize... need to pickle numpy array of final points w/ interpolations
df_snapped = FitFile_To_DF(FIT_Snapped)

# repair connections to the "position" which will be used in all analysis
if True:
    df_snapped["lonlat"] = df_snapped["Value 3"].astype(str) + df_snapped["Value 2"].astype(str)
    df_positions = pd.DataFrame(list(Positions_Dict.items()), columns=["lonlat", "position"])
    df_snapped = pd.merge(df_snapped, df_positions, how="left", left_on="lonlat", right_on="lonlat")
    del df_snapped["lonlat"]

UtmArray_Snapped, UtmZone, X_min, Y_min = Dataframe_to_UtmArray(df_snapped, X_min, Y_min)

StoreResults(
    InitialArray=cluster, #X, Y, I, ts, m_per_s
    TrackArray=trackArray, #X, Y
    FinalArray=TrackWorkout, #I,  ts, position, c_s
    InterpolatedArray=UtmArray_Snapped, # X, Y, I, ts, m_s
)

print("{}s elapsed".format(round(time.time()-start, 3)))

#labels, probs = clusters[:, 0], clusters[:, 1]
#if False:
PlotEllipse(
    ellipseArray=ellipse,
    clusterArray=np.hstack((cluster[:, 0:1], cluster[:, 1:2])),
    #clusterLabels=cluster[:, 2],
    #clusterProbabilities=cluster[:, 3],
    trackArray=trackArray,
)

# RESOURCES
# https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
# https://scikit-learn.org/stable/modules/clustering.html
# https://mathworld.wolfram.com/Ellipse.html

# installed utm: https://github.com/Turbo87/utm 
# installed JDK: https://www.oracle.com/java/technologies/javase-jdk15-downloads.html 
# installed FIT SDK: https://developer.garmin.com/fit/download/

# 2021-02-08 notes to continue...
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