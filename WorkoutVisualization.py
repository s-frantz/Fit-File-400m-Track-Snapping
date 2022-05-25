def TrackNodesTraversed(ORIG, DEST, MAX):
    nodes_traversed = 0
    forward, backward = ORIG, ORIG
    while forward != DEST and backward != DEST:
        nodes_traversed+=1
        forward+=1
        backward-=1
        if forward>MAX: forward=0
        if backward<0: backward=MAX
    direction = 1 if forward==DEST else -1
    return nodes_traversed, direction

def CorrectPaces(a):
    def CalculatePaces(a_, dist_, splits_):
        def GetLastPace():
            i_ = len(correct_paces)-1
            while i_ >= 0:
                if correct_paces[i_]!= null_pace:
                    return correct_paces[i_]
                i_ -= 1
        null_pace = 1000
        # key: X, Y, I, ts, m_s, position
        correct_paces, split_completion = [null_pace], [0]
        distance_covered, direction_taken = [0], [0]
        prev_position, prev_time = a_["position"][0], a_["timestamp"][0]
        max_position = a_["position"].max() # should be 399 - TEST THIS
        split_crossings, total_distance = 0, 0
        time_since_last_split, distance_since_last_split = 0, 0
        for i, r in a_[1:].iterrows():
            next_time, next_position = r["timestamp"], r["position"]
            time_delta = next_time - prev_time
            try:
                distance, direction = TrackNodesTraversed(
                    ORIG=int(prev_position),
                    DEST=int(next_position),
                    MAX=max_position
                )
            except:
                distance, direction = 0, 0
            distance_since_last_split += distance
            time_since_last_split += time_delta
            if next_position in splits_:
                split_crossings+=1
                if distance_since_last_split != dist_: # was 400
                    correct_paces.append(time_since_last_split * 400 / distance_since_last_split)
                    # i THINK risking div 0 error here
                else:
                    correct_paces.append(time_since_last_split * 400 / dist_) # remove *4 for 400s 
                time_since_last_split, distance_since_last_split = 0, 0
            else:
                correct_paces.append(null_pace)
            direction_taken.append(direction)
            distance_covered.append(total_distance)
            split_completion.append(split_crossings)
            prev_position, prev_time = next_position, next_time

        last_pace = GetLastPace()
        
        for i, p in reversed(list(enumerate(correct_paces))):
            if p!=null_pace: last_pace = p
            correct_paces[i] = last_pace

        a_["speed_400m"] = correct_paces
        a_["splits"] = split_completion
        a_["direction"] = direction_taken

        return a_

    a_100 = CalculatePaces(a.copy(), 100, (0, 101, 201, 301))
    a_200 = CalculatePaces(a.copy(), 200, (0, 201))
    a_400 = CalculatePaces(a.copy(), 400, (0,))

    return a_100, a_200, a_400


import sys, os, numpy as np, pandas as pd
import dash
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd

FIT_DIR = r"C:\_GitHub\TrackCat\_W"

FIT_ID = "C5IH3444.FIT" # 10x300 @ Cuesta
    #"B49A3004.FIT" # 3200 - 3200 - 1600 + fast 400 cutback w/ Keira
    #B3VA5900.FIT" # 2 x DMR (solo)
    #B3KB1127.FIT" # 5 x 1K on 1K off
    #B3DB0907.FIT" # 11 x 800 w/ Will and Cleo
    #B2FE1101.FIT" #solo 2x(3x1k - 1600)
    #B2H90533.FIT" #keira 4x1600
    #B2FE1101.FIT" #solo 2x(3x1k - 1600)
    #B2A84849.FIT"

FIT = os.path.join(FIT_DIR, FIT_ID)

npz = np.load(FIT.replace(".FIT", ".npz"))

init_array = npz["arr_0"]   #[X, Y, ts, I] (I = index of row in original csv)
                            #X, Y, I, ts, m_per_s
track_array = npz["arr_1"] #[X, Y] index = P-index
final_array = npz["arr_2"] #[I, ts, P_index, curve_straight]
interpolated_array = npz["arr_3"] #[X, Y, ts, I, m_per_s] (I = index of row in original csv)
                                    # X, Y, I, ts, m_s, position # but this is the I after interpolation

ts_min = init_array[:, 3].min()

init_array[:, 3] = init_array[:, 3]-ts_min
final_array[:, 1] = final_array[:, 1]-ts_min
interpolated_array[:, 3] = interpolated_array[:, 3]-ts_min

ts_max = init_array[:, 3].max()
x_max = init_array[:, 0].max()
y_max = init_array[:, 1].max()

x_min = init_array[:, 0].min()
y_min = init_array[:, 1].min()

if False:
    print(track_array[:, 0].max())
    print(init_array[:, 0].max())
    print(interpolated_array[:, 0].max())

    print(track_array[:, 0].min())
    print(init_array[:, 0].min())
    print(interpolated_array[:, 0].min())
    print(" ")
    print(track_array[:, 1].max())
    print(init_array[:, 1].max())
    print(interpolated_array[:, 1].max())

    print(track_array[:, 1].min())
    print(init_array[:, 1].min())
    print(interpolated_array[:, 1].min())


interpolated_array = interpolated_array[
    np.logical_and(interpolated_array[:, 3] > 0, interpolated_array[:, 3] < ts_max)
    ]

xy_max = x_max if x_max > y_max else y_max

init_labeled = np.hstack((init_array[:, :], np.zeros((len(init_array), 1))))
interpolated_labeled = np.hstack((interpolated_array[:, :], np.zeros((len(interpolated_array), 1))))

df_track = pd.DataFrame(
    data=track_array,
    index=[i for i in range(len(track_array))],
    columns=["lon", "lat"]
)

df_final = pd.DataFrame(
    data=final_array[:, [2, 1, 0]],
    columns=["p_index", "timestamp", "I"]
)

df_final_joined = pd.merge(df_track, df_final, right_on="p_index", left_on=df_track.index)
df_final_joined["source"] = 1
del df_final_joined["p_index"]

df_init = pd.DataFrame(
    data=init_labeled,
    columns=["lon", "lat", "I", "timestamp", "speed", "source"]
)
df_interpolated = pd.DataFrame(
    data=interpolated_labeled,
    columns=["lon", "lat", "I", "timestamp", "speed", "position", "source"] # "position"
)

df_100, df_interpolated, df_400 = CorrectPaces(a=df_interpolated)

if False:
    print(df_interpolated)
    df_interpolated = df_interpolated.sort_values(df_interpolated.index)
    print(df_interpolated)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df_final_joined.sort_values(by='timestamp', ascending=True, inplace=True)
df_final_joined["timestamp"] /= 60
df_init["timestamp"] /= 60
df_interpolated["timestamp"]/=60
x, y, z = df_final_joined["lon"], df_final_joined["lat"], df_final_joined["timestamp"]
x1, y1, z1 = df_interpolated["lon"], df_interpolated["lat"], df_interpolated["timestamp"]
speed1, position1 = round(df_interpolated["speed_400m"], 1), df_interpolated["position"]
splits = df_interpolated["splits"]
#speed1 = 400 * 1 / (speed1 + .0001) # to ensure no division by 0
x_init, y_init, z_init = df_init["lon"], df_init["lat"], df_init["timestamp"]
df_snap = pd.DataFrame(
    [
        t for I in df_final_joined["I"] for t in [
            [   float(df_init['lon'].loc[df_init['I']==I]),
                float(df_init['lat'].loc[df_init['I']==I]),
                float(df_init['timestamp'].loc[df_init['I']==I])
            ], 
            [   float(df_final_joined['lon'].loc[df_final_joined['I']==I]),
                float(df_final_joined['lat'].loc[df_final_joined['I']==I]),
                float(df_final_joined['timestamp'].loc[df_final_joined['I']==I])
            ],
            [   None, None, None
            ],
        ]
    ],
    columns=["X", "Y", "Z"]
)

fig = go.Figure(
    data=[
        go.Scatter3d(
            x=x[:1], y=y[:1], z=z[:1],
            marker=dict(size=3),
            name="Animation",
        ),
        go.Scatter3d(
            x=df_snap["X"], y=df_snap["Y"], z=df_snap["Z"],
            marker=dict(size=1.5, color='gray'),
            line=dict(color='#303030',width=.5),
            mode='markers+lines',
            visible='legendonly',
            name="Original GPS nodes"
        ),
        go.Scatter3d(
            x=x, y=y, z=z,
            marker=dict(size=2, color='black'),# color=z, colorscale='Portland'),
            mode="markers",
            visible='legendonly',
            name="Snapped GPS nodes",
        ),
        go.Scatter3d(
            x=x1, y=y1, z=z1,
            marker=dict(
                # colorbar=dict(
                #    thickness=10,
                #    lenmode="fraction",
                #    len=.6,
                # ),
               size=2,
               color=speed1,
               colorscale='Portland',
               cmax=90,
               cmin=64,
               showscale=False,#True,
               reversescale=True,
            ),
            #color was z1 # size was 1.1
            line=dict(
                colorbar=dict(
                    thickness=20,
                    lenmode="fraction",
                    len=.6,
                ),
                width=1.5,
                color=speed1,
                colorscale='Portland',
                cmax=90,
                cmin=64,
                showscale=True,
                reversescale=True
            ),
            #color=speed1, colorscale='Portland', width=6.5, cmax=80, cmin=65, showscale=True, reversescale=True),
            #'#303030',width=1.5),
            text=splits.astype(str) + " - " + speed1.astype(str), #speed1,#position1,
            hoverinfo="text",
            mode="markers+lines",
            name="Interpolated Track",
        ),
    ],
    frames=[
        go.Frame(
            data=go.Scatter3d(
                x=x[k:k+1], y=y[k:k+1], z=z[k:k+1],
                marker=dict(size=10)
            )
        ) for k in range(len(x1))
    ]
)

fig.update_layout(
    #hovermode='x',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01),
    updatemenus=[dict(  type="buttons",
                        buttons=[dict(label="Play",
                                method="animate",
                                #mode="immediate",
                                #args = [None])
                                args = [
                                    None, {
                                        "mode": "immediate",
                                        "frame": {"duration": 0},
                                        "transition": {"duration": 0}
                                        }]),
                                {"args": [[None], {"frame": {"duration": 0},
                                "mode": "immediate",
                                "transition": {"duration": 0}}],
                                "label": "Pause",
                                "method": "animate"}
            
                                ])],
    margin=dict(l=0, r=0, t=0, b=0),
    width=1000,
    height=850,
    autosize=False,
    scene=dict(
        xaxis = dict(nticks=4, range=[0,xy_max], title="X position (m)", showspikes=False),
        yaxis = dict(nticks=4, range=[0,xy_max], title="Y position (m)", showspikes=False),
        zaxis = dict(nticks=4, range=[0,ts_max/60], title="Time (mins)", showspikes=False),
        #camera=dict(
            #up=dict(x=.5,y=.5,z=.5),
            #eye=dict(x=1,y=1,z=1,)
            #eye=dict(x=.6,y=.6,z=.6,)
        #),
        aspectratio = dict( x=1, y=1, z=.5 ),
        aspectmode = 'manual'
    ),
)

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')

app.layout = html.Div([
    # html.Div([
        dcc.Graph(
            id='tracks',
            figure=fig,
            #shapes=shapes,
            config={  'displaylogo': False} #'hovermode':'closest'}#'hoverClosest3d': False}
        ),
        # html.Div(
        #             dt.DataTable(
        #     data=df.to_dict('rows'),
        #     columns=[{'id': c, 'name': c} for c in df.columns[0:1]],
        #     style_cell_conditional=[
        #         {'if': {'column_id': 'Version'},
        #          'width': '10px'},
        #         {'if': {'column_id': 'Uptime'},
        #          'width': '10px'},
        #     ]
        # )
        #     , className="col s6"),

        #"<h1>I am the text boom boom boom </h1>",
        #html.Div('interesting stuff', style={'display': 'inline-block'}),
        #html.Div('More intersting stuff', style={'display': 'inline-block'})
    ],
)

#     className="row",
#     ),
# className="container"
# )

#, style={'display': 'inline-block'})

# custom color ramp where the various easy paces look ok but the reallly fast stuff also looks really fast

# html.Div(
#     html.Div(
#         [
#             html.Div(table1, className="col s6"),
#             html.Div(table2, className="col s6"),
#         ],
#         className="row",
#     ),
#     className="container",
# )

if __name__ == '__main__':
    app.run_server(debug=False)