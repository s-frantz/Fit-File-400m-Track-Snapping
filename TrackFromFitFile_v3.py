def DecodeFitData(fit_file):

    # https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
    # https://scikit-learn.org/stable/modules/clustering.html
    # https://mathworld.wolfram.com/Ellipse.html

    def RecordFitDataMessage(f):
        try: f = f[0]
        except IndexError: return
        field_metadata = [
            f.base_type,
            f.def_num,
            f.field,
            f.field_def,
            f.is_base_type,
            f.is_expanded,
            f.is_named,
            None if f.parent_field is None else f.parent_field.name,
            f.type,
            f.units,
        ]
        if f.name.endswith("_lat") or f.name.endswith("_long"):
            f.value = f.value * (180 / 2 ** 31) # semicircles to degrees
        try:
            fields[f.name][0] += 1
            fields[f.name][-1][FDM_i] = f.value
        except KeyError:
            fields[f.name] = [1] + field_metadata + [{FDM_i: f.value}]

    field_properties = [
        'occurences',
        'base_type',
        'def_num',
        'field',
        'field_def',
        'field_type',
        'is_base_type',
        'is_expanded',
        'is_named',
        'parent_field',
        'type',
        'units',
        'value',
        ]
    fields = {
        'name': field_properties
        }

    with fitdecode.FitReader(fit_file) as fit:
        FDMs = [f for f in fit if isinstance(f, fitdecode.FitDataMessage)]
        for FDM_i, FDM in enumerate(FDMs):
            field_index = 0
            f = list(FDM.get_fields(field_index))
            while field_index < 300: #f
                RecordFitDataMessage(f)
                field_index+=1
                f = list(FDM.get_fields(field_index))
                
    return fields


def FitDataToArray(
    fit_data,
    CSV=False,#r"C:\Users\silas.frantz\Desktop\_Strava\Wkt\B1UB0939.csv"
    ):
    lon, lat, ts = "position_long", "position_lat", "timestamp"
    activity_stream = {
        i : [
            v,
            fit_data[lat][-1][i],
            fit_data[ts][-1][i]
            ]
        for i, v in fit_data[lon][-1].items()
    }
    df = pd.DataFrame.from_dict(
        activity_stream,
        orient='index',
        columns=[lon,lat,ts],
    )
    if CSV:
        df.to_csv(
            CSV,
            index=False
        )
    XY_dd = df[[lon, lat]].to_numpy()
    XY_m = DD_to_Meters(XY_dd)
    X_m, Y_m = XY_m[:,0:1], XY_m[:,1:]
    XY_m_adj = np.hstack(( X_m - X_m.min(), Y_m - Y_m.min() ))
    return XY_m_adj, X_m.min(), Y_m.min()


def Cluster(
    array,
    min_cluster_size,
    CSV=False,#r"C:\Users\silas.frantz\Desktop\_Strava\Wkt\cluster.csv",
    ):
    clusterer = hdbscan.HDBSCAN(min_cluster_size).fit(array)
    cluster_count = clusterer.labels_.max() - clusterer.labels_.min()
    results_array = np.column_stack((clusterer.labels_, clusterer.probabilities_))
    points_with_cluster_results = np.column_stack((array, results_array))
    #headings = ["lon", "lat", "labels", "probabilities"]
    if CSV: # save a table of run points
        np.savetxt(
            CSV,
            points_with_cluster_results,
            delimiter=",")
    return cluster_count, points_with_cluster_results, clusterer.labels_, clusterer.probabilities_


def PlotEllipse(x, xMin, xMax, yMin, yMax, ellipseArray, n, n_tot, trackCircles):

    #plt.figure(figsize=(50, 40), dpi=150)
   
    if False:
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)

        # plot ellipse line
        x_coord = np.linspace(xMin,xMax,300)
        y_coord = np.linspace(yMin,yMax,300)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
        plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)

    # set colors by cluster number
    color_palette = sns.color_palette('deep', n_tot)
    cluster_colors = [color_palette[x] if x >= 0
                    else (0.5, 0.5, 0.5)
                    for x in labels]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                            zip(cluster_colors, probabilities)]

    # plot run points and ellipse of best fit
    plt.scatter(*ellipseArray.T, s=1)
    plt.scatter(*trackCircles[0].T, s=1)
    plt.scatter(*trackCircles[1].T, s=1)
    plt.scatter(*trackCircles[2].T, s=1)
    plt.scatter(*trackCircles[3].T, s=1)
    plt.scatter(*activity_point_array.T, s=1, c=cluster_member_colors, alpha=0.5)
    plt.axis('scaled')
    plt.ylim((yMin, yMax))
    plt.xlim((xMin, xMax))

    # save
    plt.savefig(r"C:\Users\silas.frantz\Desktop\_Strava\track_ellipse_{}.png".format(n))


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

    eX, eY = ellipseArray[:,0:1],ellipseArray[:,1:]
    cent = (eX.mean(), eY.mean())
    end1, eL_half = PointOfMaxDistance(fromPT=cent)
    end2, eL = PointOfMaxDistance(fromPT=end1)
    eW_half = MinDistance(fromPT=cent)

    Pi = np.pi
    # empirical adjustment (ratio of track length-to-width to ellipse length-to-width)
    L_to_W = 1.171291534 * eL_half / eW_half
    # below equation derived by hand in merrill lynch notebook
    # parameterizing length_to_width in terms of the track radius (r) = W / 2
    # using known track circumference (C) = 400 = 2 * Pi * r + 2 * (L - 2r)
    r_T = 200 / (2*(L_to_W - 2) + 2 + Pi)
    W = 2 * r_T
    L = 2*W + 200 - 2*r_T - Pi * r_T
    centerDefCircle_DistanceFromTrackCenter = (L / 2) - r_T
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
    centC1 = (cent[0] + changeX1, cent[1] + changeY1)
    centC2 = (cent[0] - changeX1, cent[1] - changeY1)

    print(" - center C1: {}\n - center C2: {}".format(centC1, centC2))

    def GeneratePointsAlongCircle(center, radius, start_theta):
        circle_points = []
        theta_step = Pi / 100 # 200 points or roughly 1 per meter for avg track
        theta = start_theta # 0 # need to ensure angle is positive
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

    # Build an array of points along track...
    track_angle = np.arctan(m)
    # Curves
    Curve1 = GeneratePointsAlongCircle(center=centC1, radius=r_T, start_theta=track_angle-Pi/2)
    Curve2 = GeneratePointsAlongCircle(center=centC2, radius=r_T, start_theta=track_angle+Pi/2)

    #Straightaways
    Start_Finish, One_Hundo = Curve1[0], Curve1[-1]
    Two_Hundo, Three_Hundo = Curve2[0], Curve2[-1]
    Back_Straight = GeneratePointsAlongStraightaway(One_Hundo, Two_Hundo, track_angle)
    Home_Straight = GeneratePointsAlongStraightaway(Three_Hundo, Start_Finish, track_angle+Pi)

    #print(Start_Finish)
    #print("Length c1 = {}".format(len(Curve1)))

    return (Curve1, Back_Straight, Curve2, Home_Straight)
    
    def GenerateTrack(c1, c2):
        #convex hull of both circle arrays will generate the track oval
        C = np.append(c1, c2)
        from scipy.spatial import ConvexHull
        #need to add scipy to requirements
        return ConvexHull(C)

    # finally match the original run points up to the closest location on the convex hull... maybe to the meter!
    # so really what we need out of the convex hull object is a np array of points
    # at every meter along the edge of the track. might be a faster way to do this without calling in scipy

    return GenerateTrack(defCircle1, defCircle2)


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


def DD_to_Meters(ellipseArray):
    # https://github.com/Turbo87/utm
    utmArrays = utm.from_latlon(ellipseArray[:,1:], ellipseArray[:,0:1])
    return np.column_stack((utmArrays[0],utmArrays[1]))


def FitEllipse(a, n, n_tot):

    X = a[:,0:1]
    Y = a[:,1:]

    a_plot = np.hstack((X, Y))

    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b, rcond=1)[0].squeeze()

    print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(x[0],x[1],x[2],x[3],x[4]))
    xMin, xMax = a[:,0:1].min(), a[:,0:1].max()
    yMin, yMax = a[:,1:].min(), a[:,1:].max()

    # Plot the least squares ellipse
    if False:
        x_coord = np.linspace(xMin,xMax,300)
        y_coord = np.linspace(yMin,yMax,300)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
        plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)
        plt.scatter(X, Y, label='Data Points')#plt.scatter(*a.T, s=1)#, c=cluster_member_colors, alpha=0.5)
        plt.axis('scaled')
        plt.ylim((yMin, yMax))
        plt.xlim((xMin, xMax))
        plt.savefig(r"C:\Users\silas.frantz\Desktop\_Strava\Wkt\ellipse_Test_sideWAYS.png")
        import sys
        sys.exit()

    ellipseArray = GeneratePointsAlongEllipse(x, xMin, xMax)
    trackArray = Generate400mTrack(ellipseArray)
    PlotEllipse(x, xMin, xMax, yMin, yMax, ellipseArray, n, n_tot, trackArray)


# MAIN
import sys, fitdecode, hdbscan, utm
import pandas as pd, numpy as np
import seaborn as sns, matplotlib.pyplot as plt
sys.path.append(r'\\ace-ra-fs1\data\GIS\_Dev\python\apyx')
from apyx import JsonPrettyPrint

fit_data = DecodeFitData(fit_file=r"C:\Users\silas.frantz\Desktop\_Strava\Wkt\B1UB0939.FIT")
activity_point_array, X_adjust, Y_adjust = FitDataToArray(fit_data)
num_clusters, clusters, labels, probabilities = Cluster(activity_point_array, min_cluster_size=50)

print("\n{} clusters discovered...".format(num_clusters))
n = num_clusters
while n > 0: # -1's are unclassed outliers
    n = n - 1
    cluster_slice = clusters[ # potential track
        np.where(
            (clusters[:,2]==n) * (clusters[:,3] > 0.89) 
            ) # SQL tested in arcmap, where: cluster = 0 ANd probability > 0.89
        ][:, :2]
    # test to see the quality of ellipse fitting to this cluster
    print("\ncluster index {} contains {} points".format(n, len(cluster_slice)))
    #print("\ncluster {}:\n{}".format(n, cluster_slice))
    if len(cluster_slice)>500: FitEllipse(cluster_slice, n, num_clusters)