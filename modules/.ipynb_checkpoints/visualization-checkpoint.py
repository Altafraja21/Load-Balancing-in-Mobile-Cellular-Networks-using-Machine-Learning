import matplotlib.pyplot as plt

def plot_assignments(users, bs, title, ax=None):
    if ax is None:
        plt.figure(figsize=(6,6))
        ax = plt.gca()
    scatter = ax.scatter(users['x'], users['y'], c=users['Connected_BS'], cmap='tab10', s=20, alpha=0.7)
    ax.scatter(bs['x'], bs['y'], c='red', marker='^', s=180, label='Base Stations')
    for _, row in bs.iterrows():
        ax.annotate(f"BS{int(row.BS_ID)}\nC={int(row.capacity)}", (row.x+1, row.y+1), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return ax
