import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib import tri


def assemble_k_alpha(
    nodes: np.ndarray, elements: np.ndarray, alpha: np.ndarray | None = None
):
    """
    This function assembles the k_alpha matrix. It does so without using python
    loops which was somewhat difficult to achieve (thank you claude / gemini).
    I also learned that a coo_matrix is the fastest sparse matrix constructor.
    Then a compressed sparse row matrix (csr) is fast for row operations.
    And a compressed sparse column matrix (csc) is fast for column operations.
    The k_alpha matrix is returned as a csr for matrix multiplication speed
    for boundary condition calculations but should be converted to a csc for
    spla.spsolve() (or at least that is what gemini says docs don't say that)

    nodes: np.ndarray of (number_of_nodes, 2), x and y coords of nodes
    elements: np.ndarray of (number_of_elements, 3) 3 node indexes into nodes.
    """
    x = nodes[elements, 0]
    y = nodes[elements, 1]
    bs = np.zeros((len(elements), 3))
    bs[:, 0] = y[:, 1] - y[:, 2]
    bs[:, 1] = y[:, 2] - y[:, 0]
    bs[:, 2] = y[:, 0] - y[:, 1]
    cs = np.zeros((len(elements), 3))
    cs[:, 0] = x[:, 2] - x[:, 1]
    cs[:, 1] = x[:, 0] - x[:, 2]
    cs[:, 2] = x[:, 1] - x[:, 0]
    two_delta = np.abs(bs[:, 0] * cs[:, 1] - bs[:, 1] * cs[:, 0])
    # (number_of_elements, 3, 3)
    # [b1 * b1, b1 * b2, b1 * b3]
    # [b2 * b1, b2 * b2, b2 * b3]
    # [b3 * b1, b3 * b2, b3 * b3]
    k = (bs[:, :, None] * bs[:, None, :] + cs[:, :, None] * cs[:, None, :]) / (
        2.0 * two_delta[:, None, None]
    )
    if alpha is not None:
        # alpha[elements].mean(axis=1) averages the alpha on the element
        k = alpha[elements].mean(axis=1)[:, None, None] * k
    rows = np.repeat(elements, 3, axis=1)  # [1 1 1 2 2 2 3 3 3]
    cols = np.tile(elements, (1, 3))  # [1 2 3 1 2 3 1 2 3]
    return sp.coo_matrix(
        (k.ravel(), (rows.ravel(), cols.ravel())),
        shape=(nodes.shape[0], nodes.shape[0]),
    ).tocsr()


def fem_plot(nodes, elements, phi, savefig=False, savepath="default.png"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

    ax[0].triplot(triang, lw=0.5, color="grey")
    ax[0].scatter(nodes[:, 0], nodes[:, 1], c=phi, cmap="inferno")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x (m)")
    ax[0].set_ylabel("y (m)")
    ax[0].set_title("Node Solutions")

    pc = ax[1].tripcolor(triang, phi, shading="gouraud", cmap="inferno")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("x (m)")
    ax[1].set_ylabel("y (m)")
    ax[1].set_title("Elements with Gouraud Shading")

    fig.colorbar(pc, ax=ax[0], fraction=0.046, pad=0.04)
    fig.colorbar(pc, ax=ax[1], fraction=0.046, pad=0.04)
    if savefig:
        fig.savefig(savepath, dpi=200)
    return fig
