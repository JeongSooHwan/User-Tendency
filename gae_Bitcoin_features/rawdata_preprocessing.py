import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import time

from PIL import Image
import cv2
import imageio
import os
import natsort
import matplotlib.pyplot as plt


def create_graph(dataset, debug=False, save=False):
    df = pd.read_csv("../gae_Bitcoin/data/soc-sign-bitcoin{}.csv".format(dataset), header=None)
    columns = ["source", "target", "rating", "raw_time"]
    df.columns = columns

    df = df.sort_values(by="raw_time")
    df = df.reset_index(drop=True)

    if debug:
        print("-------------------")
        print("|| Original data ||")
        print("-------------------")
        print(df)
        print()

    def standardization(x):
        if len(x) < 2:
            return "0" + x
        else:
            return x

    real_time_list = list()

    for t in list(df["raw_time"]):
        real_time = list(time.gmtime(t))
        y = str(real_time[0])
        m = standardization(str(real_time[1]))
        d = standardization(str(real_time[2]))

        real_time_list.append(y[2:] + m + d)

    df["time"] = real_time_list
    del df["raw_time"]

    if save:
        df.to_csv("../gae_Bitcoin/data/bitcoin{}.csv".format(dataset))

    if debug:
        print("-------------------------")
        print("|| Convert time format ||")
        print("-------------------------")
        print(df)
        print()

    source_node_list = df["source"].to_list()
    target_node_list = df["target"].to_list()

    node_list = source_node_list + target_node_list
    node_list = list(set(node_list))

    G = nx.DiGraph()
    G.add_nodes_from(node_list)

    for src, trg, rat, tm in df.to_numpy():
        G.add_edge(src, trg, rating=rat, time=tm)
    if save:
        df.to_csv("../gae_Bitcoin/data/bitcoin{}.csv".format(dataset))
        nx.write_gml(G, "../gae_Bitcoin/data/bitcoin{}.gml".format(dataset))
    np_array = df.to_numpy()

    return G, df


if __name__ == "__main__":
    G, np_array = create_graph("otc", save=True)
    print(list(G.edges(data=True))[:10])
    print(np_array)
    print((113, 54) in G.edges())
