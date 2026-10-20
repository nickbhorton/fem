import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib import tri
from scipy.constants import epsilon_0, mu_0
from scipy.special import hankel2, jn


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
    alpha: np.ndarray,
    alpha_is_element_wise=False,
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
    if alpha_is_element_wise:
        alpha_element_wise = alpha
    else:
        alpha_element_wise = alpha[element_indices].mean(axis=1)
    k = alpha_element_wise[:, None, None] * k
    rows = np.repeat(element_indices, 3, axis=1)  # [1 1 1 2 2 2 3 3 3]
    cols = np.tile(element_indices, (1, 3))  # [1 2 3 1 2 3 1 2 3]
    return sp.coo_matrix(
        (k.ravel(), (rows.ravel(), cols.ravel())),
        shape=(node_coords.shape[0], node_coords.shape[0]),
        dtype=np.complex128,
    ).tocsr()


def assemble_k_beta(
    node_coords: np.ndarray,
    element_indices: np.ndarray,
    beta: np.ndarray,
    beta_is_element_wise=False,
):
    """
    This function assembles the k_beta matrix.

    node_coords: np.ndarray of (number_of_nodes, 2), x and y coords of nodes
    domain_element_indicies: np.ndarray of (number_of_elements, 3)
        3 node indices that construct a triangle.
    beta: np.ndarray of (number_of_nodes): represents the beta value on each
        node.
    """
    element_areas = get_element_area(node_coords, element_indices)
    if beta_is_element_wise:
        beta_element_wise = beta
    else:
        beta_element_wise = beta[element_indices].mean(axis=1)
    mult_array = (
        np.tile(
            np.array([2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]) / 12.0,
            element_indices.shape[0],
        )
        * np.repeat(element_areas, 9)
        * np.repeat(beta_element_wise, 9)
    )
    rows = np.repeat(element_indices, 3, axis=1)  # [1 1 1 2 2 2 3 3 3]
    cols = np.tile(element_indices, (1, 3))  # [1 2 3 1 2 3 1 2 3]
    return sp.coo_matrix(
        (mult_array.ravel(), (rows.ravel(), cols.ravel())),
        shape=(node_coords.shape[0], node_coords.shape[0]),
        dtype=np.complex128,
    ).tocsr()


def assemble_k_gamma(
    node_coords: np.ndarray,
    line_indices: np.ndarray,
    gamma: np.ndarray,
):
    edges_gamma = node_coords[line_indices]
    edge_lengths_gamma = np.linalg.norm(
        edges_gamma[:, 0, :] - edges_gamma[:, 1, :], axis=1
    )
    k = np.squeeze(
        gamma[:, np.newaxis]
        * edge_lengths_gamma[:, np.newaxis]
        * np.array([1 / 3, 1 / 6, 1 / 6, 1 / 3])[np.newaxis, :]
    )
    rows_gamma = np.repeat(line_indices, 2, axis=1)  # [1 1 2 2]
    cols_gamma = np.tile(line_indices, (1, 2))  # [1 2 1 2]
    return sp.coo_matrix(
        (k.ravel(), (rows_gamma.ravel(), cols_gamma.ravel())),
        shape=(node_coords.shape[0], node_coords.shape[0]),
        dtype=np.complex128,
    ).tocsr()


def fem_pec_tmz(mesh):
    element_tag = mesh.get_cell_data("gmsh:physical", "triangle")
    line_tag = mesh.get_cell_data("gmsh:physical", "line")
    name_to_tag = {k: int(v[0]) for k, v in mesh.field_data.items()}

    node_coords = mesh.points[:, :2]
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    rho = np.sqrt(x**2 + y**2)

    element_indices = mesh.get_cells_type("triangle")
    line_indices = mesh.get_cells_type("line")

    # boundary condition lines
    inner_line_indices = line_indices[line_tag == name_to_tag["inner"]]
    outer_line_indices = line_indices[line_tag == name_to_tag["outer"]]

    f = 10e9
    k_0 = 2.0 * np.pi * f * np.sqrt(epsilon_0 * mu_0)
    mu_r = np.ones(node_coords.shape[0])
    epsilon_r = np.ones(node_coords.shape[0])
    E0 = 1.0

    beta = -(k_0**2) * epsilon_r
    alpha = 1 / mu_r
    gamma = 1j * k_0 + 1 / (2 * rho[outer_line_indices]).mean(axis=1)

    element_unique_node_indices = np.unique(element_indices.ravel())
    inner_unique_indices = np.unique(inner_line_indices.ravel())
    dirichlet_node_indices = np.concatenate([inner_unique_indices])
    free_node_indices = np.setdiff1d(
        element_unique_node_indices, dirichlet_node_indices
    )

    # k_alpha
    k_alpha = assemble_k_alpha(node_coords, element_indices, alpha)
    k_alpha_free = k_alpha[np.ix_(free_node_indices, free_node_indices)]

    # k_beta
    k_beta = assemble_k_beta(node_coords, element_indices, beta)
    k_beta_free = k_beta[np.ix_(free_node_indices, free_node_indices)]

    # k_gamma
    k_gamma = assemble_k_gamma(node_coords, outer_line_indices, gamma)
    k_gamma_free = k_gamma[np.ix_(free_node_indices, free_node_indices)]

    # b_dirichlet
    phi_dirichlet = np.concatenate(
        [
            np.zeros(len(dirichlet_node_indices)),
        ]
    )
    b_dirichlet = (
        -k_alpha[np.ix_(free_node_indices, dirichlet_node_indices)] @ phi_dirichlet
    )

    # b_neumann
    b_neumann = np.zeros(len(node_coords), dtype=np.complex128)

    kappa = (
        E0
        * np.exp(-1j * k_0 * x[outer_line_indices])
        * (
            -(1j * k_0 * x[outer_line_indices]) / rho[outer_line_indices]
            + 1j * k_0
            + 1 / (2 * rho[outer_line_indices])
        )
    ).mean(axis=1)
    # (number_neumann_edges, (p1, p2), (x, y))
    edges_neumann = node_coords[outer_line_indices]
    edge_lengths_neumann = np.linalg.norm(
        edges_neumann[:, 0, :] - edges_neumann[:, 1, :], axis=1
    )
    np.add.at(
        b_neumann,
        outer_line_indices.ravel(),
        np.repeat(0.5 * kappa * edge_lengths_neumann, 2),
    )
    b_neumann = b_neumann[free_node_indices]

    # final assembly
    k_total_free = k_alpha_free + k_beta_free + k_gamma_free
    b_total = b_neumann + b_dirichlet

    # solve for phi
    phi = np.zeros(len(node_coords), dtype=np.complex128)
    phi[free_node_indices] = np.asarray(spla.spsolve(k_total_free.tocsc(), b_total))
    return phi


def fem_pec_tez(mesh_path):
    mesh = meshio.read(mesh_path)

    element_tag = mesh.get_cell_data("gmsh:physical", "triangle")
    line_tag = mesh.get_cell_data("gmsh:physical", "line")
    name_to_tag = {k: int(v[0]) for k, v in mesh.field_data.items()}

    node_coords = mesh.points[:, :2]
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    rho = np.sqrt(x**2 + y**2)

    element_indices = mesh.get_cells_type("triangle")
    line_indices = mesh.get_cells_type("line")

    # boundary condition lines
    inner_line_indices = line_indices[line_tag == name_to_tag["inner"]]
    outer_line_indices = line_indices[line_tag == name_to_tag["outer"]]

    f = 10e9
    k_0 = 2.0 * np.pi * f * np.sqrt(epsilon_0 * mu_0)
    mu_r = np.ones(node_coords.shape[0])
    epsilon_r = np.ones(node_coords.shape[0])
    E0 = 1.0
    H0 = 1.0 / np.sqrt(mu_0 / epsilon_0)

    beta = -(k_0**2) * mu_r
    alpha = 1 / epsilon_r
    gamma = 1j * k_0 + 1 / (2 * rho[outer_line_indices]).mean(axis=1)

    element_unique_node_indices = np.unique(element_indices.ravel())
    inner_unique_indices = np.unique(inner_line_indices.ravel())
    free_node_indices = element_unique_node_indices

    # k_alpha
    k_alpha = assemble_k_alpha(node_coords, element_indices, alpha)
    k_alpha_free = k_alpha[np.ix_(free_node_indices, free_node_indices)]

    # k_beta
    k_beta = assemble_k_beta(node_coords, element_indices, beta)
    k_beta_free = k_beta[np.ix_(free_node_indices, free_node_indices)]

    # k_gamma
    k_gamma = assemble_k_gamma(node_coords, outer_line_indices, gamma)
    k_gamma_free = k_gamma[np.ix_(free_node_indices, free_node_indices)]

    # b_neumann
    b_neumann = np.zeros(len(node_coords), dtype=np.complex128)

    kappa = (
        H0
        * np.exp(-1j * k_0 * x[outer_line_indices])
        * (
            -(1j * k_0 * x[outer_line_indices]) / rho[outer_line_indices]
            + 1j * k_0
            + 1 / (2 * rho[outer_line_indices])
        )
    ).mean(axis=1)
    # (number_neumann_edges, (p1, p2), (x, y))
    edges_neumann = node_coords[outer_line_indices]
    edge_lengths_neumann = np.linalg.norm(
        edges_neumann[:, 0, :] - edges_neumann[:, 1, :], axis=1
    )
    np.add.at(
        b_neumann,
        outer_line_indices.ravel(),
        np.repeat(0.5 * kappa * edge_lengths_neumann, 2),
    )
    b_neumann = b_neumann[free_node_indices]

    # final assembly
    k_total_free = k_alpha_free + k_beta_free + k_gamma_free
    b_total = b_neumann

    # solve for phi
    phi = np.zeros(len(node_coords), dtype=np.complex128)
    phi[free_node_indices] = np.asarray(spla.spsolve(k_total_free.tocsc(), b_total))
    return phi


def fem_die_tmz(mesh_path):
    mesh = meshio.read(mesh_path)

    element_tag = mesh.get_cell_data("gmsh:physical", "triangle")
    line_tag = mesh.get_cell_data("gmsh:physical", "line")
    name_to_tag = {k: int(v[0]) for k, v in mesh.field_data.items()}

    node_coords = mesh.points[:, :2]
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    rho = np.sqrt(x**2 + y**2)

    element_indices = mesh.get_cells_type("triangle")
    line_indices = mesh.get_cells_type("line")

    # boundary condition lines
    outer_line_indices = line_indices[line_tag == name_to_tag["outer"]]

    f = 10e9
    k_0 = 2.0 * np.pi * f * np.sqrt(epsilon_0 * mu_0)
    mu_r = np.ones(node_coords.shape[0])
    E0 = 1.0

    # element wise permittivity
    epsilon_r = np.zeros(element_indices.shape[0])
    epsilon_r[element_tag == name_to_tag["air"]] = 1.0
    epsilon_r[element_tag == name_to_tag["dielectric"]] = 9.0

    beta = -(k_0**2) * epsilon_r
    alpha = 1 / mu_r
    gamma = 1j * k_0 + 1 / (2 * rho[outer_line_indices]).mean(axis=1)

    element_unique_node_indices = np.unique(element_indices.ravel())
    # no dirichlet boundary condition
    free_node_indices = element_unique_node_indices

    # k_alpha
    k_alpha = assemble_k_alpha(node_coords, element_indices, alpha)
    k_alpha_free = k_alpha[np.ix_(free_node_indices, free_node_indices)]

    # k_beta
    k_beta = assemble_k_beta(node_coords, element_indices, beta, True)
    k_beta_free = k_beta[np.ix_(free_node_indices, free_node_indices)]

    # k_gamma
    k_gamma = assemble_k_gamma(node_coords, outer_line_indices, gamma)
    k_gamma_free = k_gamma[np.ix_(free_node_indices, free_node_indices)]

    # b_neumann
    b_neumann = np.zeros(len(node_coords), dtype=np.complex128)

    kappa = (
        E0
        * np.exp(-1j * k_0 * x[outer_line_indices])
        * (
            -(1j * k_0 * x[outer_line_indices]) / rho[outer_line_indices]
            + 1j * k_0
            + 1 / (2 * rho[outer_line_indices])
        )
    ).mean(axis=1)
    # (number_neumann_edges, (p1, p2), (x, y))
    edges_neumann = node_coords[outer_line_indices]
    edge_lengths_neumann = np.linalg.norm(
        edges_neumann[:, 0, :] - edges_neumann[:, 1, :], axis=1
    )
    np.add.at(
        b_neumann,
        outer_line_indices.ravel(),
        np.repeat(0.5 * kappa * edge_lengths_neumann, 2),
    )
    b_neumann = b_neumann[free_node_indices]

    # final assembly
    k_total_free = k_alpha_free + k_beta_free + k_gamma_free
    b_total = b_neumann

    # solve for phi
    phi = np.zeros(len(node_coords), dtype=np.complex128)
    phi[free_node_indices] = np.asarray(spla.spsolve(k_total_free.tocsc(), b_total))
    return phi


def fem_die_tez(mesh_path):
    mesh = meshio.read(mesh_path)

    element_tag = mesh.get_cell_data("gmsh:physical", "triangle")
    line_tag = mesh.get_cell_data("gmsh:physical", "line")
    name_to_tag = {k: int(v[0]) for k, v in mesh.field_data.items()}

    node_coords = mesh.points[:, :2]
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    rho = np.sqrt(x**2 + y**2)

    element_indices = mesh.get_cells_type("triangle")
    line_indices = mesh.get_cells_type("line")

    # boundary condition lines
    outer_line_indices = line_indices[line_tag == name_to_tag["outer"]]

    f = 10e9
    k_0 = 2.0 * np.pi * f * np.sqrt(epsilon_0 * mu_0)
    mu_r = np.ones(node_coords.shape[0])
    E0 = 1.0
    H0 = E0 / np.sqrt(mu_0 / epsilon_0)

    # element wise permittivity
    epsilon_r = np.zeros(element_indices.shape[0], dtype=np.complex128)
    epsilon_r[element_tag == name_to_tag["air"]] = 1.0
    epsilon_r[element_tag == name_to_tag["dielectric"]] = 9.0

    beta = -(k_0**2) * mu_r
    alpha = 1 / epsilon_r
    gamma = 1j * k_0 + 1 / (2 * rho[outer_line_indices]).mean(axis=1)

    element_unique_node_indices = np.unique(element_indices.ravel())
    # no dirichlet boundary condition
    free_node_indices = element_unique_node_indices

    # k_alpha
    k_alpha = assemble_k_alpha(node_coords, element_indices, alpha, True)
    k_alpha_free = k_alpha[np.ix_(free_node_indices, free_node_indices)]

    # k_beta
    k_beta = assemble_k_beta(node_coords, element_indices, beta)
    k_beta_free = k_beta[np.ix_(free_node_indices, free_node_indices)]

    # k_gamma
    k_gamma = assemble_k_gamma(node_coords, outer_line_indices, gamma)
    k_gamma_free = k_gamma[np.ix_(free_node_indices, free_node_indices)]

    # b_neumann
    b_neumann = np.zeros(len(node_coords), dtype=np.complex128)

    kappa = (
        H0
        * np.exp(-1j * k_0 * x[outer_line_indices])
        * (
            -(1j * k_0 * x[outer_line_indices]) / rho[outer_line_indices]
            + 1j * k_0
            + 1 / (2 * rho[outer_line_indices])
        )
    ).mean(axis=1)
    # (number_neumann_edges, (p1, p2), (x, y))
    edges_neumann = node_coords[outer_line_indices]
    edge_lengths_neumann = np.linalg.norm(
        edges_neumann[:, 0, :] - edges_neumann[:, 1, :], axis=1
    )
    np.add.at(
        b_neumann,
        outer_line_indices.ravel(),
        np.repeat(0.5 * kappa * edge_lengths_neumann, 2),
    )
    b_neumann = b_neumann[free_node_indices]

    # final assembly
    k_total_free = k_alpha_free + k_beta_free + k_gamma_free
    b_total = b_neumann

    # solve for phi
    phi = np.zeros(len(node_coords), dtype=np.complex128)
    phi[free_node_indices] = np.asarray(spla.spsolve(k_total_free.tocsc(), b_total))
    return phi


def bistatic_echo_width_pec_tmz_analytical(phi: np.ndarray):
    f = 10e9
    phase_velocity = 1 / np.sqrt(epsilon_0 * mu_0)
    wavelength = phase_velocity / f
    a = wavelength / 2
    k = 2.0 * np.pi * f / phase_velocity
    ka = k * a

    N = 15
    ns = np.arange(-N, N + 1)
    pre_factor = jn(ns, ka) / hankel2(ns, ka)
    return (4.0 / k) * np.abs(
        (pre_factor[:, None] * np.exp(-1j * ns[:, None] * phi[None, :])).sum(axis=0)
    ) ** 2


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


def fem_plot2(nodes, elements, phi, savefig=False, savepath="default.png"):
    fig, ax = plt.subplots(constrained_layout=True)
    triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

    pc = ax.tripcolor(triang, phi, shading="gouraud", cmap="hot")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Elements with Gouraud Shading")

    fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
    if savefig:
        fig.savefig(savepath, dpi=200)
    return fig


def polar_plot(phi, xs):
    fig, ax = plt.subplots(layout="constrained")
    ax.remove()
    ax = fig.add_subplot(projection="polar")
    for x in xs:
        ax.plot(phi, x)
    ax.grid(True, alpha=0.4)
    return fig
