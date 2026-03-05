import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl
from itertools import cycle



def get_data_from_file(path, dtype=[True,True,False,False]):  # True means integer, False means float
    server_n = []
    intra_speed = []
    data = []
    error = []
    with open(path, 'r') as file:
        while line := file.readline():
            # print(line.rstrip())
            line_list = line.rstrip().split(" ")
            s_n = int(line_list[0]) if dtype[0] else float(line_list[0])
            speed = int(line_list[1]) if dtype[1] else float(line_list[1])
            d = int(line_list[2]) if dtype[2] else float(line_list[2])
            err = int(line_list[3]) if dtype[3] else float(line_list[3])

            if s_n not in server_n:
                data.append([])
                error.append([])
                server_n.append(s_n)
            cur_data_list = data[-1]
            cur_data_list.append(d)
            cur_error_list = error[-1]
            cur_error_list.append(err)

            if speed not in intra_speed:
                intra_speed.append(speed)
            
    return server_n, intra_speed, data, error


def line_graph(server_n, speed, data, error, xlabel, ylabel, img_name, legend_loc = "lower right", log_x = 0):

    plt.clf()
    line_style = ["-","--","-.",":"]
    linecycler = cycle(line_style)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set1.colors)

    idx = 0
    for d in data:
        x = np.array(speed)
        y = np.array(d)
        label = server_n[idx]
        if isinstance(server_n[0], int):
            plt.plot(x, y, next(linecycler), label="server# = " + str(label))
        else:
            plt.plot(x, y, next(linecycler), label=label)
        idx += 1

    plt.legend(loc=legend_loc)
    plt.xlabel(xlabel)
    if log_x != 0:
        plt.xscale('log',base=log_x) 
    plt.ylabel(ylabel)
    plt.style.use('tableau-colorblind10')
    plt.tight_layout()
    plt.savefig(img_name, bbox_inches='tight')
    return

# speedup vs intra-link speed
server_n, speed, data, error = get_data_from_file("../benchmark/speedup_intra_speed.txt")
line_graph(server_n, speed, data, error, "Intra-server link speed (Gbps)", "Speedup", "speedup_intra_speed.png")

# ratio vs intra-link speed
server_n, speed, data, error = get_data_from_file("../benchmark/intra_ratio_intra_speed.txt")
line_graph(server_n, speed, data, error, "Intra-server link speed (Gbps)", "Ratio of intra all-to-all to total communication", "intra_ratio_intra_speed.png", "upper right")

# speedup vs skewness
server_n, skewness, data, error = get_data_from_file("../benchmark/speedup_server_number_skewness.txt", [True, False, False, False])
line_graph(server_n, skewness, data, error, "skewness factor", "Speedup", "speedup_skewness.png", "upper left")

# speedup vs transfer size
dt = []
ts = []
er = []
for i in range(0, 3):
    server_n, transfer, data, error = get_data_from_file("../benchmark/speedup_server_number_transfer" + str(i) + ".txt", [True, False, False, False])
    id = server_n.index(64)
    dt.append(data[id])
    er.append(error[id])
    ts = transfer

labels = ["no scale", "balanced scale", "always scale"]
line_graph(labels, ts, dt, er, "transfer size (MB)", "Speedup", "speedup_transfer.png", "lower right", 10)
