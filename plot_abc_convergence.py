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


data1 = np.load("node_convergence_abc1.npz")
data2 = np.load("node_convergence_abc2.npz")

nodes_pec1 = data1["nodes_pec"]
nodes_die1 = data1["nodes_die"]
error_pec_tmz1 = data1["error_pec_tmz"]
error_pec_tez1 = data1["error_pec_tez"]
error_die_tmz1 = data1["error_die_tmz"]
error_die_tez1 = data1["error_die_tez"]

nodes_pec2 = data2["nodes_pec"]
nodes_die2 = data2["nodes_die"]
error_pec_tmz2 = data2["error_pec_tmz"]
error_pec_tez2 = data2["error_pec_tez"]
error_die_tmz2 = data2["error_die_tmz"]
error_die_tez2 = data2["error_die_tez"]

fig, ax = plt.subplots(2, 2, layout="constrained")

ax[0][0].plot(nodes_pec1, error_pec_tmz1, color="black")
ax[0][0].plot(nodes_pec2, error_pec_tmz2, color="blue")

ax[0][1].plot(nodes_pec1, error_pec_tez1, color="black")
ax[0][1].plot(nodes_pec2, error_pec_tez2, color="blue")

ax[1][0].plot(nodes_die1, error_die_tmz1, color="black")
ax[1][0].plot(nodes_die2, error_die_tmz2, color="blue")

ax[1][1].plot(nodes_die1, error_die_tez1, color="black")
ax[1][1].plot(nodes_die2, error_die_tez2, color="blue")

for a in ax.flat:
    a.set_xscale("log")
    a.set_yscale("log")
    a.legend()
    a.set_xlabel("Node Count")
    a.set_ylabel("RMSE")
plt.show()
