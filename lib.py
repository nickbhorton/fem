import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib import tri


def get_element_area(node_coords: np.ndarray, element_indices: np.ndarray):
    x = node_coords[element_indices, 0]
    y = node_coords[element_indices, 1]
    bs = np.zeros((len(element_indices), 3))
    bs[:, 0] = y[:, 1] - y[:, 2]
    bs[:, 1] = y[:, 2] - y[:, 0]
    bs[:, 2] = y[:, 0] - y[:, 1]
    cs = np.zeros((len(element_indices), 3))
    cs[:, 0] = x[:, 2] - x[:, 1]
    cs[:, 1] = x[:, 0] - x[:, 2]
    cs[:, 2] = x[:, 1] - x[:, 0]
    return 0.5 * np.abs(bs[:, 0] * cs[:, 1] - bs[:, 1] * cs[:, 0])


def assemble_k_alpha(
    node_coords: np.ndarray,
    element_indices: np.ndarray,
    alpha: np.ndarray | None = None,
):
    """
    This function assembles the k_alpha matrix. It does so without using python
    loops which was somewhat difficult to achieve (thank you claude / gemini).
    I also learned that a coo_matrix is the fastest sparse matrix constructor.
    Then a compressed sparse row matrix (csr) is fast for row operations.
    And a compressed sparse column matrix (csc) is fast for column operations.
    The k_alpha matrix is returned as a csr for matrix multiplication speed
    for boundary condition calculations, but should be converted to a csc for
    spla.spsolve()

    node_coords: np.ndarray of (number_of_nodes, 2), x and y coords of nodes
    domain_element_indicies: np.ndarray of (number_of_elements, 3)
        3 node indices that construct a triangle.
    alpha: np.ndarray of (number_of_nodes): represents the alpha value on each
        node.
    """
    x = node_coords[element_indices, 0]
    y = node_coords[element_indices, 1]
    bs = np.zeros((len(element_indices), 3))
    bs[:, 0] = y[:, 1] - y[:, 2]
    bs[:, 1] = y[:, 2] - y[:, 0]
    bs[:, 2] = y[:, 0] - y[:, 1]
    cs = np.zeros((len(element_indices), 3))
    cs[:, 0] = x[:, 2] - x[:, 1]
    cs[:, 1] = x[:, 0] - x[:, 2]
    cs[:, 2] = x[:, 1] - x[:, 0]
    two_area = np.abs(bs[:, 0] * cs[:, 1] - bs[:, 1] * cs[:, 0])
    # (number_of_elements, 3, 3)
    # [b1 * b1, b1 * b2, b1 * b3]
    # [b2 * b1, b2 * b2, b2 * b3]
    # [b3 * b1, b3 * b2, b3 * b3]
    k = (bs[:, :, None] * bs[:, None, :] + cs[:, :, None] * cs[:, None, :]) / (
        2.0 * two_area[:, None, None]
    )
    if alpha is not None:
        # alpha[elements].mean(axis=1) averages the alpha on the element
        k = alpha[element_indices].mean(axis=1)[:, None, None] * k
    rows = np.repeat(element_indices, 3, axis=1)  # [1 1 1 2 2 2 3 3 3]
    cols = np.tile(element_indices, (1, 3))  # [1 2 3 1 2 3 1 2 3]
    return sp.coo_matrix(
        (k.ravel(), (rows.ravel(), cols.ravel())),
        shape=(node_coords.shape[0], node_coords.shape[0]),
        dtype=np.complex128,
    ).tocsr()


def assemble_k_beta(
    node_coords: np.ndarray,
    domain_element_indices: np.ndarray,
    beta: np.ndarray | None = None,
):
    """
    This function assembles the k_beta matrix.

    node_coords: np.ndarray of (number_of_nodes, 2), x and y coords of nodes
    domain_element_indicies: np.ndarray of (number_of_elements, 3)
        3 node indices that construct a triangle.
    beta: np.ndarray of (number_of_nodes): represents the beta value on each
        node.
    """
    element_areas = get_element_area(node_coords, domain_element_indices)
    if beta is not None:
        # alpha[elements].mean(axis=1) averages the alpha on the element
        beta_element_wise = beta[domain_element_indices].mean(axis=1)
        mult_array = (
            np.tile(
                np.array([2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]) / 12.0,
                domain_element_indices.shape[0],
            )
            * np.repeat(element_areas, 9)
            * np.repeat(beta_element_wise, 9)
        )
    else:
        mult_array = np.tile(
            np.array([2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]) / 12.0,
            domain_element_indices.shape[0],
        ) * np.repeat(element_areas, 9)
    rows = np.repeat(domain_element_indices, 3, axis=1)  # [1 1 1 2 2 2 3 3 3]
    cols = np.tile(domain_element_indices, (1, 3))  # [1 2 3 1 2 3 1 2 3]
    return sp.coo_matrix(
        (mult_array.ravel(), (rows.ravel(), cols.ravel())),
        shape=(node_coords.shape[0], node_coords.shape[0]),
        dtype=np.complex128,
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
