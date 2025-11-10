import numpy as np

def jain_index(l):
    l = np.array(l, float)
    return 1.0 if l.sum() == 0 else (l.sum()**2) / (len(l) * (l**2).sum())

def throughput_proxy(l, c):
    l, c = np.array(l), np.array(c)
    s = np.minimum(l, c)
    return s.sum(), (s / c).mean()

def compute_loads(users, bs):
    """
    Count users per BS and return a numpy array aligned with bs['BS_ID'] order.
    Handles missing Connected_BS column and empty users gracefully.
    """
    if users is None or users.empty:
        return np.zeros(len(bs), dtype=int)

    if "Connected_BS" not in users.columns:
        return np.zeros(len(bs), dtype=int)

    # value_counts -> map by BS_ID to ensure alignment even if some IDs are missing
    counts = users["Connected_BS"].value_counts().to_dict()  # {bs_id: count}
    # Use map to align counts with bs['BS_ID']
    loads = bs["BS_ID"].map(lambda i: int(counts.get(i, 0))).values
    return loads.astype(int)
