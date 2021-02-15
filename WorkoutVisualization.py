import sys, numpy as np, pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd

FIT = r"C:\Users\silas.frantz\Desktop\B2A84849.FIT"
npz = np.load(FIT.replace(".FIT", ".npz"))

init_array = npz["arr_0"] #[X, Y, ts, I] (I = index of row in original csv)
track_array = npz["arr_1"] #[X, Y] index = P-index
final_array = npz["arr_2"] #[I, ts, P_index, curve_straight]
interpolated_array = npz["arr_3"] #[X, Y, ts, I] (I = index of row in original csv)

ts_min = init_array[:, 2].min()
init_array[:, 2] = init_array[:, 2]-ts_min
final_array[:, 1] = final_array[:, 1]-ts_min
interpolated_array[:, 2] = interpolated_array[:, 2]-ts_min

ts_max = init_array[:, 2].max()
x_max = init_array[:, 0].max()
y_max = init_array[:, 1].max()

interpolated_array = interpolated_array[
    np.logical_and(interpolated_array[:, 2] > 0, interpolated_array[:, 2] < ts_max)
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
    columns=["lon", "lat", "timestamp", "I", "source"]
)
df_interpolated = pd.DataFrame(
    data=interpolated_labeled,
    columns=["lon", "lat", "timestamp", "I", "source"]
)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df_final_joined.sort_values(by='timestamp', ascending=True, inplace=True)
df_final_joined["timestamp"] /= 60
df_init["timestamp"] /= 60
df_interpolated["timestamp"]/=60
x, y, z = df_final_joined["lon"], df_final_joined["lat"], df_final_joined["timestamp"]
x1, y1, z1 = df_interpolated["lon"], df_interpolated["lat"], df_interpolated["timestamp"]
x_init, y_init, z_init = df_init["lon"], df_init["lat"], df_init["timestamp"]

df_snap = pd.DataFrame( [t for I in df_final_joined["I"] for t in [
            [   float(df_init['lon'].loc[df_init['I']==I]),
                float(df_init['lat'].loc[df_init['I']==I]),
                float(df_init['timestamp'].loc[df_init['I']==I])], 
            [   float(df_final_joined['lon'].loc[df_final_joined['I']==I]),
                float(df_final_joined['lat'].loc[df_final_joined['I']==I]),
                float(df_final_joined['timestamp'].loc[df_final_joined['I']==I])],
            [   None, None, None]
            ]],
            columns=["X", "Y", "Z"])

fig = go.Figure(data=[
    
    go.Scatter3d(
            x=x[:1], y=y[:1], z=z[:1],
            marker=dict(
                size=3,
            )),
    
    go.Scatter3d(
    x=x, y=y, z=z,
    marker=dict(
        size=2.75,
        color=z,
        colorscale='Portland',#'Viridis',
    ),
    mode="markers",
    #line=dict(
    #    color='#432E49',#'#607EA0',#'lightblue',#'#303030', #'#4B312C',
    #    width=1.5
    #    )
    ),
    go.Scatter3d(
    x=x1, y=y1, z=z1,
    marker=dict(
        size=1,
        color=z1,
        colorscale='Portland',#'Viridis',
    ),
    line=dict(
        color='#303030',#'#432E49',#'#607EA0',#'lightblue',#'#303030', #'#4B312C',
        width=1.5
        )
    ),
    #go.Scatter3d(
    #x=x_init, y=y_init, z=z_init,
    #marker=dict(
    #    size=1.5,
    #    color='#606060',
    #    ),
    #mode='markers',
    #),
    go.Scatter3d(
    x=df_snap["X"], y=df_snap["Y"], z=df_snap["Z"],
    marker=dict(
        size=1.5,
        color='gray',
        ),
    line=dict(
        color='#303030', #'#4B312C',
        width=.5
        ),
    mode='markers+lines',
    )
    ],
    frames=[go.Frame(data=go.Scatter3d(
            x=x[k:k+1], y=y[k:k+1], z=z[k:k+1],
            marker=dict(
                size=10,
            ))) for k in range(len(x1))]
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
    width=900,
    height=900,
    autosize=False,
    scene=dict(
        xaxis = dict(nticks=4, range=[0,xy_max], title="X position (m)", showspikes=False),
        yaxis = dict(nticks=4, range=[0,xy_max], title="Y position (m)", showspikes=False),
        zaxis = dict(nticks=4, range=[0,ts_max/60], title="Time (mins)", showspikes=False),
        #camera=dict(
        #    up=dict(
        #        x=0,
        #        y=0,
        #        z=1
        #    ),
        #    eye=dict(
        #        x=0,
        #        y=1.0707,
        #        z=1,
        #    )
        #),
        aspectratio = dict( x=1, y=1, z=0.7 ),
        aspectmode = 'manual'
    ),
)

config = {  'displaylogo': False}#, 'hovermode':'closest'}#'hoverClosest3d': False}

app.layout = html.Div([
    dcc.Graph(
        id='tracks',
        figure=fig,
        #shapes=shapes,
        config=config
    )
])

if __name__ == '__main__':
    app.run_server(debug=False)