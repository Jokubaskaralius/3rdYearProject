#https://plotly.com/python/visualizing-mri-volume-slices/
# Import data
import time
import numpy as np

from skimage import io

vol = io.imread(
    "https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif"
)
volume = vol.T
r, c = volume[0].shape

# Define frames
import plotly.graph_objects as go
nb_frames = 68

fig = go.Figure(frames=[
    go.Frame(
        data=go.Surface(z=(6.7 - k * 0.1) * np.ones((r, c)),
                        surfacecolor=np.flipud(volume[67 - k]),
                        cmin=0,
                        cmax=200),
        name=str(
            k
        )  # you need to name the frame for the animation to behave properly
    ) for k in range(nb_frames)
])

# Add data to be displayed before animation starts
fig.add_trace(
    go.Surface(z=6.7 * np.ones((r, c)),
               surfacecolor=np.flipud(volume[67]),
               colorscale='Gray',
               cmin=0,
               cmax=200,
               colorbar=dict(thickness=20, ticklen=4)))


def frame_args(duration):
    return {
        "frame": {
            "duration": duration
        },
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {
            "duration": duration,
            "easing": "linear"
        },
    }


sliders = [{
    "pad": {
        "b": 10,
        "t": 60
    },
    "len":
    0.9,
    "x":
    0.1,
    "y":
    0,
    "steps": [{
        "args": [[f.name], frame_args(0)],
        "label": str(k),
        "method": "animate",
    } for k, f in enumerate(fig.frames)],
}]

# Layout
fig.update_layout(
    title='Slices in volumetric data',
    width=600,
    height=600,
    scene=dict(
        zaxis=dict(range=[-0.1, 6.8], autorange=False),
        aspectratio=dict(x=1, y=1, z=1),
    ),
    updatemenus=[{
        "buttons": [
            {
                "args": [None, frame_args(50)],
                "label": "&#9654;",  # play symbol
                "method": "animate",
            },
            {
                "args": [[None], frame_args(0)],
                "label": "&#9724;",  # pause symbol
                "method": "animate",
            },
        ],
        "direction":
        "left",
        "pad": {
            "r": 10,
            "t": 70
        },
        "type":
        "buttons",
        "x":
        0.1,
        "y":
        0,
    }],
    sliders=sliders)

fig.show()


############################
############################
############################
############################
#https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[2] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()


def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])