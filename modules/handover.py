# modules/handover.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .simulation import compute_distances

# Train a simple RandomForest to predict next BS given current state
def generate_training_data(bs, num_users=1000, steps=10, area=100):
    """
    Simulate many short trajectories and collect (x,y,vx,vy,current_bs) -> next_bs
    """
    rows = []
    for _ in range(num_users):
        x = np.random.uniform(0, area)
        y = np.random.uniform(0, area)
        vx = np.random.uniform(-1.5, 1.5)
        vy = np.random.uniform(-1.5, 1.5)
        for _t in range(steps):
            # compute current connected bs
            dists = ((bs[["x","y"]].values - np.array([x,y]))**2).sum(axis=1)**0.5
            cur = int(np.argmin(dists))
            # move a bit
            x_next = np.clip(x + vx, 0, area)
            y_next = np.clip(y + vy, 0, area)
            dists_next = ((bs[["x","y"]].values - np.array([x_next,y_next]))**2).sum(axis=1)**0.5
            nxt = int(np.argmin(dists_next))
            rows.append([x,y,vx,vy,cur,nxt])
            x,y = x_next, y_next
    df = pd.DataFrame(rows, columns=["x","y","vx","vy","cur","nxt"])
    return df

def train_predictor(bs):
    """
    Train and return a RandomForest classifier that predicts next BS index.
    """
    df = generate_training_data(bs, num_users=600, steps=8, area=100)
    X = df[["x","y","vx","vy","cur"]].values
    y = df["nxt"].values
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

def predict_next_bs(clf, user_row, bs):
    """
    Predict next BS index for a single user Row (Series-like).
    user_row must have x,y,vx,vy,Connected_BS
    """
    X = np.array([[user_row["x"], user_row["y"], user_row["vx"], user_row["vy"], int(user_row["Connected_BS"])]])
    pred = clf.predict(X)
    return int(pred[0])

def detect_handover_event(user_row, bs, margin=0.85):
    """
    Simple rule: if another BS distance is < margin * current_distance -> handover suggested.
    Returns new_bs (int) or None.
    """
    dists = compute_distances(user_row.to_frame().T, bs)[0]
    cur = int(user_row["Connected_BS"])
    cur_dist = dists[cur]
    candidate = int(np.argmin(dists))
    if candidate != cur and dists[candidate] < cur_dist * margin:
        return candidate
    return None
