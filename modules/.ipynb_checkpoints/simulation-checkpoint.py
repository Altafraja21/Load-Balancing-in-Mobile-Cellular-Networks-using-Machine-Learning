import numpy as np
import pandas as pd

np.random.seed(42)

def simulate_environment(num_bs=5, num_users=200, area_size=100, cap_low=25, cap_high=45):
    bs = pd.DataFrame({
        'BS_ID': np.arange(num_bs),
        'x': np.random.uniform(0, area_size, num_bs),
        'y': np.random.uniform(0, area_size, num_bs),
        'capacity': np.random.randint(cap_low, cap_high + 1, num_bs)
    })
    users = pd.DataFrame({
        'User_ID': np.arange(num_users),
        'x': np.random.uniform(0, area_size, num_users),
        'y': np.random.uniform(0, area_size, num_users),
    })
    return bs, users

def compute_distances(users, bs):
    u_coords = users[['x', 'y']].values
    b_coords = bs[['x', 'y']].values
    d = np.sqrt(((u_coords[:, None, :] - b_coords[None, :, :]) ** 2).sum(axis=2))
    return d

def assign_nearest_bs(users, bs):
    d = compute_distances(users, bs)
    users = users.copy()
    users['Connected_BS'] = np.argmin(d, axis=1)
    return users
