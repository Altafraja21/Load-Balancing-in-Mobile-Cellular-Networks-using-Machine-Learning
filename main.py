from modules.simulation import simulate_environment, assign_nearest_bs
from modules.algorithms import heuristic_rebalance, kmeans_reassign
from modules.evaluation import evaluate_and_print, compute_loads, throughput_proxy, jain_index
from modules.visualization import plot_assignments
import pandas as pd
import matplotlib.pyplot as plt

def main():
    NUM_BS = 6
    NUM_USERS = 300
    AREA = 100

    # Simulate environment
    bs, users = simulate_environment(NUM_BS, NUM_USERS, AREA)
    users_baseline = assign_nearest_bs(users, bs)

    # Evaluate baseline
    evaluate_and_print("Baseline (Nearest BS)", users_baseline, bs)

    # Heuristic balancing
    users_heuristic = heuristic_rebalance(users_baseline, bs)
    evaluate_and_print("Heuristic Rebalanced", users_heuristic, bs)

    # KMeans balancing
    users_kmeans = kmeans_reassign(users, bs)
    evaluate_and_print("KMeans Reassigned", users_kmeans, bs)

    # Visual comparison
    fig, axes = plt.subplots(1, 3, figsize=(18,6))
    plot_assignments(users_baseline, bs, "Baseline", ax=axes[0])
    plot_assignments(users_heuristic, bs, "Heuristic", ax=axes[1])
    plot_assignments(users_kmeans, bs, "KMeans", ax=axes[2])
    plt.tight_layout()
    plt.show()

    # Save outputs
    summary = []
    for label, u in [("Baseline", users_baseline), ("Heuristic", users_heuristic), ("KMeans", users_kmeans)]:
        loads = compute_loads(u, bs)
        capacities = bs['capacity'].values
        overload_count = int((loads > capacities).sum())
        jain = jain_index(loads)
        total_throughput, avg_util = throughput_proxy(loads, capacities)
        summary.append({
            'Method': label,
            'Overloaded_BS': overload_count,
            'Jain_Index': round(jain, 4),
            'Throughput': int(total_throughput),
            'Avg_Utilization': round(avg_util, 3)
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("data/lb_summary.csv", index=False)
    print("\n✅ Results saved in data/lb_summary.csv")

if __name__ == "__main__":
    main()
