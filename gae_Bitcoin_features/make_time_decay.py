import networkx as nx
import numpy as np
import pandas as pd

from rawdata_preprocessing import create_graph
from collections import defaultdict
from datetime import datetime, timedelta


def make_time_series_dict(dataset):
    G, df = create_graph(dataset)

    transaction_src_dict = defaultdict(list)
    transaction_time_dict = defaultdict(list)
    for i in range(len(df)):
        src, trg, rating, time = df.iloc[i]

        str_year = time[:2]
        str_month = time[2:4]
        str_day = time[4:]

        year = int("20" + str_year)
        month = int(str_month)
        day = int(str_day)

        date_time_form = datetime(year, month, day)

        transaction_src_dict[src].append([trg, rating, date_time_form])
        transaction_time_dict[date_time_form].append([src, trg, rating])

    time_series_dict = defaultdict(list)
    for date_time, transactions in transaction_time_dict.items():
        for trans in transactions:
            time_series_dict[trans[0]].append([trans[1], trans[2], date_time])

    return time_series_dict


# def scale_dif_days(date1, date2, days):
#     dif_days = np.abs((date2 - date1).days)
#     mok = dif_days // days
#     return (1 / 2) ** mok


def cal_dif_days(date1, date2, days):
    dif_days = np.abs((date2 - date1).days)
    mok = dif_days // days
    return mok


def make_time_decay_graph(ts_data):
    my_edge_list = list()

    for k, transactions in ts_data.items():
        k_th_trans_list = transactions.copy()

        for i in range(len(k_th_trans_list)):
            semi_result = defaultdict(list)
            for j in range(i + 1):
                if i == 0:
                    my_edge_list.append((k, k_th_trans_list[i][0], k_th_trans_list[i][1]))
                    continue
                if i == j:
                    continue

                idx = cal_dif_days(k_th_trans_list[i][2], k_th_trans_list[j][2], 30)
                semi_result[idx].append(k_th_trans_list[j][1])

            if len(semi_result) != 0:
                c_sum = 0
                avg_sum = 0
                for sc, ratings in semi_result.items():
                    c = (1 / 2) ** sc
                    c_sum += c
                    avg_sum += np.mean(ratings) * c
                mu = avg_sum / c_sum
                r_uv = ((k_th_trans_list[i][1] - mu) / 4) + k_th_trans_list[i][1]
                my_edge_list.append((k, k_th_trans_list[i][0], r_uv))

    print(my_edge_list)
    G = nx.DiGraph()
    for u, v, rating in my_edge_list:
        G.add_edge(int(u), int(v), normrating=float(rating))

    nx.write_gml(G, "../gae_Bitcoin/data/newdata/bitcoinalphanorm2.gml")


if __name__ == "__main__":
    time_series_data = make_time_series_dict("alpha")
    make_time_decay_graph(time_series_data)
