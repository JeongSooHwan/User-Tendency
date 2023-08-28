from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing

import numpy as np

from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph, StellarDiGraph

from gensim.models import Word2Vec

import warnings
import collections
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def get_dataframe_with_weight(dataname, norm=False):
    if norm:
        nx_G = nx.read_gml("../gae_Bitcoin/data/bitcoin{}norm2.gml".format(dataname), label="id")

        src_list = list()
        trg_list = list()
        rating_list = list()

        for (u, v), normrating in nx.get_edge_attributes(nx_G, "normrating").items():
            src_list.append(str(u))
            trg_list.append(str(v))
            rating_list.append(normrating)

        df_dict = {
            "source": src_list,
            "target": trg_list,
            "weight": rating_list
        }
        edges = pd.DataFrame(df_dict)
        edges["weight"] = edges["weight"] + 15
        edges["weight"] = edges["weight"].astype(float)

        return edges

    else:
        nx_G = nx.read_gml("../gae_Bitcoin/data/bitcoin{}.gml".format(dataname), label="id")

        src_list = list()
        trg_list = list()
        rating_list = list()

        for (u, v), rating in nx.get_edge_attributes(nx_G, "rating").items():
            src_list.append(str(u))
            trg_list.append(str(v))
            rating_list.append(rating)

        df_dict = {
            "source": src_list,
            "target": trg_list,
            "weight": rating_list
        }
        edges = pd.DataFrame(df_dict)
        edges["weight"] = edges["weight"] + 10
        edges["weight"] = edges["weight"].astype(float)

        return edges


def get_stellargraph(df_edges):
    G = StellarDiGraph(edges=df_edges)
    print(G.info())

    return G


def save_embeddings_with_node2vec(SG, dim, dataname, normmode=False):
    rw = BiasedRandomWalk(SG)
    weighted_walks = rw.run(
        nodes=SG.nodes(),  # root nodes
        length=10,  # maximum length of a random walk
        n=10,  # number of random walks per root node
        p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=True,  # for weighted random walks
        seed=42,  # random seed fixed for reproducibility
    )

    weighted_model = Word2Vec(
        weighted_walks, size=dim, window=5, min_count=0, sg=1, workers=1, iter=1
    )

    sort_embeddings = list()
    for i in range(weighted_model.wv.vectors.shape[0]):
        sort_embeddings.append(weighted_model.wv[str(i)])

    sort_embeddings = np.array(sort_embeddings)

    emb_df = pd.DataFrame(sort_embeddings)
    if normmode:
        emb_df.to_csv("../gae_Bitcoin/data/newdata/bitcoin{}norm2_node2vec_embeddings.csv".format(dataname))
    else:
        emb_df.to_csv("../gae_Bitcoin/data/newdata/bitcoin{}_node2vec_embeddings.csv".format(dataname))


if __name__ == "__main__":
    input_dataname = input("Input data name: ")
    input_norm_mode = input("Norm mode(y/n): ")
    if input_norm_mode == "y":
        edges = get_dataframe_with_weight(input_dataname, True)
    else:
        edges = get_dataframe_with_weight(input_dataname)
    SG = get_stellargraph(edges)
    if input_norm_mode == "y":
        save_embeddings_with_node2vec(SG, 32, input_dataname, True)
    else:
        save_embeddings_with_node2vec(SG, 32, input_dataname)
