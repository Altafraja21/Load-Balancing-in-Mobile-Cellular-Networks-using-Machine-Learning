from sklearn.cluster import KMeans
import numpy as np
from .simulation import compute_distances

# ─────────────── Heuristic Rebalance ───────────────
def heuristic_rebalance(users, bs):
    users = users.copy()
    distances = compute_distances(users, bs)
    for _ in range(5):
        load = users["Connected_BS"].value_counts().reindex(bs["BS_ID"]).fillna(0).astype(int)
        spare = (bs["capacity"].values - load.values).astype(int)
        overloaded = np.where(load.values > bs["capacity"].values)[0]
        if not len(overloaded): break
        for ob in overloaded:
            idxs = users[users["Connected_BS"] == ob].index
            for i in idxs:
                for cand in np.argsort(distances[i]):
                    if spare[cand] > 0 and cand != ob:
                        users.at[i, "Connected_BS"] = cand
                        spare[cand] -= 1
                        spare[ob] += 1
                        break
    return users

# ─────────────── KMeans Reassignment ───────────────
def kmeans_reassign(users, bs):
    users = users.copy()
    kmeans = KMeans(n_clusters=len(bs), random_state=42, n_init=10)
    kmeans.fit(users[["x", "y"]])
    cluster_centers = kmeans.cluster_centers_
    d = np.sqrt(((cluster_centers[:, None, :] - bs[["x", "y"]].values[None, :, :]) ** 2).sum(axis=2))
    cluster_to_bs = np.argmin(d, axis=1)
    users["Connected_BS"] = [cluster_to_bs[c] for c in kmeans.labels_]
    return users
