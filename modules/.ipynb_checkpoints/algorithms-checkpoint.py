from sklearn.cluster import KMeans
import numpy as np
from .simulation import compute_distances

def heuristic_rebalance(users, bs):
    users = users.copy()
    distances = compute_distances(users, bs)
    changed = True
    iteration = 0

    while changed and iteration < 5:
        iteration += 1
        changed = False
        load_counts = users['Connected_BS'].value_counts().reindex(bs['BS_ID']).fillna(0).astype(int)
        spare = (bs['capacity'].values - load_counts.values).astype(int)
        overloaded_bss = np.where(load_counts.values > bs['capacity'].values)[0]
        if len(overloaded_bss) == 0:
            break

        for ob in overloaded_bss:
            user_idx = users[users['Connected_BS'] == ob].index
            order = np.argsort(-distances[user_idx, ob])
            for idx in user_idx[order]:
                candidate_order = np.argsort(distances[idx])
                for cand in candidate_order:
                    if cand == ob:
                        continue
                    if spare[cand] > 0:
                        users.at[idx, 'Connected_BS'] = cand
                        spare[cand] -= 1
                        spare[ob] += 1
                        changed = True
                        break
    return users

def kmeans_reassign(users, bs):
    users = users.copy()
    k = len(bs)
    coords = users[['x', 'y']].values
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(coords)
    cluster_centers = kmeans.cluster_centers_
    d_cb = np.sqrt(((cluster_centers[:, None, :] - bs[['x', 'y']].values[None, :, :]) ** 2).sum(axis=2))
    cluster_to_bs = np.argmin(d_cb, axis=1)
    users['Cluster'] = kmeans.labels_
    users['Connected_BS'] = users['Cluster'].map(lambda c: cluster_to_bs[c])
    users = users.drop(columns=['Cluster'])
    return users
