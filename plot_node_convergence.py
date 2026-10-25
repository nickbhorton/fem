import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "axes.grid": False,
        "axes.edgecolor": "black",
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "font.size": 12,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.labelsize": 14,
        "legend.frameon": False,
        "lines.linewidth": 2.5,
    }
)


data = np.load("node_convergence_abc1.npz")
nodes_pec = data["nodes_pec"]
nodes_die = data["nodes_die"]
error_pec_tmz = data["error_pec_tmz"]
error_pec_tez = data["error_pec_tez"]
error_die_tmz = data["error_die_tmz"]
error_die_tez = data["error_die_tez"]

fig, ax = plt.subplots()

ax.plot(nodes_pec, error_pec_tmz, label="pec_tmz", color="black")
ax.plot(nodes_pec, error_pec_tez, label="pec_tez", color="black", ls="--")
ax.plot(nodes_die, error_die_tmz, label="die_tmz", color="blue")
ax.plot(nodes_die, error_die_tez, label="die_tez", color="blue", ls="--")

ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
ax.set_xlabel("Node Count")
ax.set_ylabel("RMSE")
plt.show()
