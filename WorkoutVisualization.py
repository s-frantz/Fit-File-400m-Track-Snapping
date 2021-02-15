import sys, numpy as np, pandas as pd
FIT = r"C:\Users\silas.frantz\Desktop\B2A84849.FIT"

npz = np.load(FIT.replace(".FIT", ".npz"))

init_array = npz["arr_0"] #[X, Y, ts, I] (i = index of row in original csv)
track_array = npz["arr_1"] #[X, Y] index = P-index
final_array = npz["arr_2"] #[I, ts, P_index, curve_straight]

ts_min = init_array[:, 2].min()
init_array[:, 2] = init_array[:, 2]-ts_min
final_array[:, 1] = final_array[:, 1]-ts_min

ts_max = init_array[:, 2].max()
x_max = init_array[:, 0].max()
y_max = init_array[:, 1].max()

xy_max = x_max if x_max > y_max else y_max


init_labeled = np.hstack((init_array[:, :], np.zeros((len(init_array), 1))))
#track_labeled = np.hstack((track_array, np.zeros((len(track_array), 1))))

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

df_init = pd.DataFrame(data=init_labeled, columns=["lon", "lat", "timestamp", "I", "source"])

#df = pd.concat([df_init, df_final_joined])


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

if False:



    fig = px.line_3d(df_final_joined, x="lon", y="lat", z="timestamp",#)
                    #color="source", #size="population", color="continent", hover_name="country",
                    width=1100, height=800)#, size_max=60)#, log_x=True)

#fig.update_traces(marker=dict(size=1,
#                              line=dict(width=2,
#                                        color='DarkSlateGrey')),
#                  selector=dict(mode='markers'))

import plotly.graph_objects as go

df_final_joined.sort_values(by='timestamp', ascending=True, inplace=True)
df_final_joined["timestamp"] /= 60
df_init["timestamp"] /= 60
x, y, z = df_final_joined["lon"], df_final_joined["lat"], df_final_joined["timestamp"]
x_init, y_init, z_init = df_init["lon"], df_init["lat"], df_init["timestamp"]

#print(pd.DataFrame([[df_init["lon"].loc[1], df_final_joined["lon"].loc[1]]]))

#print(df_final_joined["I"][0])

#df.loc[df['column_name'] == some_value]

#print(
#    df_init['timestamp'].loc[df_init['I'] == 475]
#)
#sys.exit()
#print(
#    pd.DataFrame([[df_init['timestamp'].loc[i]], [df_final_joined['timestamp'].loc[i]]])
#)
#sys.exit()

I = 475

x11 = float(df_init['lon'].loc[df_init['I']==I])
x22 = float(df_final_joined['lon'].loc[df_final_joined['I']==I])

#print(pd.DataFrame([[x11], [x22]], columns=['lon']))
#print(pd.DataFrame([df_init['lat'].loc[df_init['I']==I]], [df_final_joined['lat'].loc[df_final_joined['I']==I]]))
#print(pd.DataFrame([df_init['timestamp'].loc[df_init['I']==I]], [df_final_joined['timestamp'].loc[df_final_joined['I']==I]]))

I = 475

xo=pd.DataFrame([
    [float(df_init['lon'].loc[df_init['I']==I])],
    [float(df_final_joined['lon'].loc[df_final_joined['I']==I])]
    ],
    columns=["lon"]
    )
yo=pd.DataFrame([
    [float(df_init['lat'].loc[df_init['I']==I])],
    [float(df_final_joined['lat'].loc[df_final_joined['I']==I])]
    ],
    columns=["lat"]
    )
zo=pd.DataFrame([
    [float(df_init['timestamp'].loc[df_init['I']==I])],
    [float(df_final_joined['timestamp'].loc[df_final_joined['I']==I])]
    ],
    columns=["timestamp"]
    )

print(xo)
print(yo)
print(zo)

df_snap = pd.DataFrame(dict(
    X=[0,1,2,3, 1,2,3,4], 
    Y=[0,2,3,1, 1,3,4,2], 
    Z=[0,3,1,2, 1,4,2,3],
    color=["a", "a", "a", "a", "b", "b", "b", "b"]
))


snap = [t for x in range(2) for t in [[1, 2, 4], [0,5,6]]]


for I in df_final_joined["I"]:

    snap = [
            [   float(df_init['lon'].loc[df_init['I']==I]),
                float(df_init['lat'].loc[df_init['I']==I]),
                float(df_init['timestamp'].loc[df_init['I']==I])], 
            [   float(df_final_joined['lon'].loc[df_final_joined['I']==I]),
                float(df_final_joined['lat'].loc[df_final_joined['I']==I]),
                float(df_final_joined['timestamp'].loc[df_final_joined['I']==I])],
            [   None, None, None]
            ]
    print(snap)
    print(" ")
    #    columns=["X", "Y", "Z"]
    #)
        
print(snap)

#print(df_snap1)

sys.exit()

fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z,
    marker=dict(
        size=3,
        color=z,
        colorscale='Portland',#'Viridis',
    ),
    line=dict(
        color='black', #'#4B312C',
        width=2
        )
    ), go.Scatter3d(
    x=x_init, y=y_init, z=z_init,
    marker=dict(
        size=2.5,
        color='darkgray',
        ),
    mode='markers',
    )
    ] #+ [go.Scatter3d(
      #  x=pd.DataFrame([
      #      [float(df_init['lon'].loc[df_init['I']==I])],
      #      [float(df_final_joined['lon'].loc[df_final_joined['I']==I])]
      #      ],
      #      columns=["lon"]
      #      ),
      #  y=pd.DataFrame([
      #      [float(df_init['lat'].loc[df_init['I']==I])],
      #      [float(df_final_joined['lat'].loc[df_final_joined['I']==I])]
      #      ],
      #      columns=["lat"]
      #      ),
      #  z=pd.DataFrame([
      #      [float(df_init['timestamp'].loc[df_init['I']==I])],
      #      [float(df_final_joined['timestamp'].loc[df_final_joined['I']==I])]
      #      ],
      #      columns=["timestamp"]
      #      ),
      #  marker=dict(
      #  size=3,
      #  color=z,
      #  colorscale='Portland',#'Viridis',
      #  ),
      #  line=dict(
      #      color='black', #'#4B312C',
      #      width=1
      #      ))
      #  for I in df_final_joined["I"][:10]
      #  ]
)



#shapes=[dict(
#        type='line',
#        x0 = df_init['lon'].loc[i],
#        y0 = df_init['lat'].loc[i],
#        x1 = df_final_joined['lon'].loc[i],
#        y1 = df_final_joined['lat'].loc[i],
#        line = dict(
#            color = 'grey',
#            width = 2
#        )
#    ) for i in range(len(df_init['lon']))]



fig.update_layout(
    #hovermode='x',
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