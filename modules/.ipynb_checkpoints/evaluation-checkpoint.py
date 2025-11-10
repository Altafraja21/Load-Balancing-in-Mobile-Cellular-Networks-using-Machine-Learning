import numpy as np

def jain_index(loads):
    loads = np.array(loads, dtype=float)
    if loads.sum() == 0:
        return 1.0
    return (loads.sum() ** 2) / (len(loads) * (loads ** 2).sum())

def throughput_proxy(loads, capacities):
    loads = np.array(loads)
    capacities = np.array(capacities)
    served = np.minimum(loads, capacities)
    total_throughput = served.sum()
    avg_utilization = (served / capacities).mean()
    return total_throughput, avg_utilization

def compute_loads(users, bs):
    load_counts = users['Connected_BS'].value_counts().reindex(bs['BS_ID']).fillna(0).astype(int)
    return load_counts.values

def evaluate_and_print(title, users, bs):
    loads = compute_loads(users, bs)
    capacities = bs['capacity'].values
    overloads = (loads > capacities).sum()
    jain = jain_index(loads)
    total_throughput, avg_util = throughput_proxy(loads, capacities)

    print(f"\n--- {title} ---")
    print(f"Loads per BS: {loads.tolist()}")
    print(f"Capacities: {capacities.tolist()}")
    print(f"Overloaded BSs: {overloads}")
    print(f"Jain's Fairness Index: {jain:.4f}")
    print(f"Total Throughput (proxy): {total_throughput}")
    print(f"Average Utilization: {avg_util:.3f}")
