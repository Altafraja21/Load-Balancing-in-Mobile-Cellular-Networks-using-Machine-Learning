# modules/simulation.py
import numpy as np
import pandas as pd
import requests

def fetch_realtime_towers_mozilla(lat=26.85, lon=80.95, radius=5000):
    url="https://location.services.mozilla.com/v1/search?key=test&type=cell"
    payload={"radio":"lte","cellTowers":[],"fallBacks":True,
             "latitude":lat,"longitude":lon,"accuracy":radius}
    try:
        r=requests.post(url,json=payload,timeout=8)
        if r.status_code==200:
            data=r.json()
            towers=[{"BS_ID":i,"x":lat+np.random.uniform(-0.02,0.02),
                     "y":lon+np.random.uniform(-0.02,0.02),
                     "capacity":np.random.randint(80,120)}
                    for i,_ in enumerate(data.get("cellTowers",[]))]
            if towers:
                df=pd.DataFrame(towers); df.to_csv("data/telecom_real.csv",index=False)
                return df
    except Exception:
        pass
    return None

def _random_bs(n,a):
    return pd.DataFrame({"BS_ID":range(n),"x":np.random.uniform(0,a,n),
                         "y":np.random.uniform(0,a,n),"capacity":np.random.randint(30,60,n)})

def simulate_environment(num_bs=6, num_users=300, area=100, use_real_data=False, live_move=False):
    """
    Returns (bs, users)
    users: has columns User_ID, x, y, vx, vy, Connected_BS (primary), Secondary_BS (maybe NaN)
    """
    if use_real_data:
        try:
            bs=pd.read_csv("data/telecom_real.csv")
            if bs.empty: raise Exception()
            # auto-detect column names
            col_map = {}
            for col in bs.columns:
                lower = col.lower()
                if "lat" in lower: col_map[col] = "x"
                elif "lon" in lower or "lng" in lower: col_map[col] = "y"
                elif "id" in lower: col_map[col] = "BS_ID"
                elif "cap" in lower: col_map[col] = "capacity"
            if col_map:
                bs.rename(columns=col_map, inplace=True)
        except Exception:
            bs = fetch_realtime_towers_mozilla() or _random_bs(num_bs, area)
    else:
        bs = _random_bs(num_bs, area)

    # users: random initial positions + velocities
    users = pd.DataFrame({
        "User_ID": np.arange(num_users),
        "x": np.random.uniform(0, area, num_users),
        "y": np.random.uniform(0, area, num_users),
        # velocity magnitude and direction -> vx, vy
        "vx": np.random.uniform(-1.5, 1.5, num_users),
        "vy": np.random.uniform(-1.5, 1.5, num_users),
    })

    # small initial values
    users["Connected_BS"] = -1
    users["Secondary_BS"] = -1

    # If live_move: nudge positions slightly (will be moved each refresh in app)
    if live_move:
        users["x"] = np.clip(users["x"] + users["vx"], 0, area)
        users["y"] = np.clip(users["y"] + users["vy"], 0, area)

    return bs.reset_index(drop=True), users.reset_index(drop=True)

def step_users(users, area=100, time_step=1.0):
    """
    Move users by vx,vy scaled by time_step and keep inside area.
    Returns updated users DataFrame (modifies positions in place).
    """
    users["x"] = users["x"] + users["vx"] * time_step
    users["y"] = users["y"] + users["vy"] * time_step
    users["x"] = np.clip(users["x"], 0, area)
    users["y"] = np.clip(users["y"], 0, area)
    return users

def compute_distances(users, bs):
    u = users[["x","y"]].values
    b = bs[["x","y"]].values
    return np.sqrt(((u[:,None,:]-b[None,:,:])**2).sum(axis=2))

def assign_nearest_bs(users, bs):
    d = compute_distances(users, bs)
    users = users.copy()
    users["Connected_BS"] = np.argmin(d, axis=1)
    # leave Secondary_BS as -1 for now
    return users
