import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl


def get_data_from_file(path):
    server_n = []
    gpu_n = []
    data = []
    error = []
    with open(path, 'r') as file:
        while line := file.readline():
            # print(line.rstrip())
            line_list = line.rstrip().split(" ")
            s_n = int(line_list[0])
            g_n = int(line_list[1])
            d = float(line_list[2])
            err = float(line_list[3])

            if s_n not in server_n:
                data.append([])
                error.append([])
                server_n.append(s_n)
            cur_data_list = data[-1]
            cur_data_list.append(d)
            cur_error_list = error[-1]
            cur_error_list.append(err)

            if g_n not in gpu_n:
                gpu_n.append(g_n)
            
    return server_n, gpu_n, data, error



def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, aspect=40, ax=ax, shrink=0.25, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_xlabel('Server number')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('GPU number')


    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)


    ax.set_aspect('equal', adjustable='box')
    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts



# Speedup
server_n, gpu_n, data, error = get_data_from_file("../benchmark/speedup_server_gpu_number.txt")
fig, ax = plt.subplots(sharex=True, sharey=True,  figsize = (20,16))

data_2d = np.array(data)
data_2d = np.transpose(data_2d)


im, cbar = heatmap(data_2d, gpu_n, server_n, ax=ax,
                   cmap="RdPu", cbarlabel="speedup")

texts = annotate_heatmap(im, valfmt="{x:.1f}")

fig.tight_layout()
plt.savefig('speedup_server_gpu_number.png', bbox_inches='tight')


# Algorithm Time
server_n, gpu_n, data, error = get_data_from_file("../benchmark/time_server_gpu_number.txt")
fig, ax = plt.subplots(sharex=True, sharey=True,  figsize = (20,16))

data_2d = np.array(data)
data_2d = np.transpose(data_2d)
data_2d = np.divide(data_2d, 1000)



im, cbar = heatmap(data_2d, gpu_n, server_n, ax=ax,
                   cmap="RdPu", cbarlabel="Algorithm time (ms)")

texts = annotate_heatmap(im, valfmt="{x:.1f}")

fig.tight_layout()
plt.savefig('time_server_gpu_number.png', bbox_inches='tight')