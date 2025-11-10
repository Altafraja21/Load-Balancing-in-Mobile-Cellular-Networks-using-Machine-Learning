import matplotlib.pyplot as plt
def plot_assignments(users, bs, title):
    plt.scatter(users["x"], users["y"], c=users["Connected_BS"], cmap="tab10", s=20)
    plt.scatter(bs["x"], bs["y"], c="red", marker="^", s=150)
    plt.title(title); plt.xlabel("X"); plt.ylabel("Y"); plt.show()
