import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib import tri
from scipy.constants import epsilon_0, mu_0
from scipy.special import h2vp, hankel2, jn, jvp


def gmsh_to_meshio() -> meshio.Mesh:
    """
    This was written by claude.
    """
    # Nodes: tags may be non-contiguous, so build a tag -> index map
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = coords.reshape(-1, 3)  # type: ignore
    order = np.argsort(node_tags)
    node_tags, points = node_tags[order], points[order]
    tag_to_idx = np.full(node_tags.max() + 1, -1, dtype=np.int64)
    tag_to_idx[node_tags] = np.arange(len(node_tags))

    gmsh_to_meshio_type = {1: "line", 2: "triangle"}
    nodes_per_type = {1: 2, 2: 3}

    cells = {}  # meshio type -> list of connectivity arrays
    cell_data = {}  # meshio type -> list of physical tags

    for dim, ptag in gmsh.model.getPhysicalGroups():
        if dim not in (1, 2):
            continue
        for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
            types, _, node_lists = gmsh.model.mesh.getElements(dim, ent)
            for etype, nl in zip(types, node_lists):
                if etype not in gmsh_to_meshio_type:
                    continue
                name = gmsh_to_meshio_type[etype]
                conn = tag_to_idx[nl.reshape(-1, nodes_per_type[etype])]  # type: ignore
                cells.setdefault(name, []).append(conn)
                cell_data.setdefault(name, []).append(
                    np.full(len(conn), ptag, dtype=np.int32)
                )

    cell_blocks = [(k, np.vstack(v)) for k, v in cells.items()]
    phys = [np.concatenate(cell_data[k]) for k, _ in cell_blocks]

    # field_data mirrors what meshio gives you when reading a .msh
    field_data = {
        gmsh.model.getPhysicalName(d, t): np.array([t, d])
        for d, t in gmsh.model.getPhysicalGroups()
    }

    return meshio.Mesh(
        points=points,
        cells=cell_blocks,  # type: ignore
        cell_data={"gmsh:physical": phys},  # type: ignore
        field_data=field_data,
    )


def build_dielectric_mesh(f, lc):
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add("die")

        phase_velocity = 1 / np.sqrt(epsilon_0 * mu_0)
        wavelength = phase_velocity / f
        wavelength_grid = 2.5
        pec_radius = wavelength / 2
        abc_radius = wavelength_grid * wavelength

        a, b = pec_radius, abc_radius

        inner = gmsh.model.occ.addDisk(0, 0, 0, a, a)
        outer = gmsh.model.occ.addDisk(0, 0, 0, b, b)
        annulus, _ = gmsh.model.occ.fragment([(2, outer)], [(2, inner)])
        gmsh.model.occ.synchronize()

        for dim, tag in gmsh.model.getEntities(2):
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
            if np.isclose(xmax, a, rtol=1e-3):
                gmsh.model.addPhysicalGroup(2, [tag], tag=2, name="dielectric")
            else:
                gmsh.model.addPhysicalGroup(2, [tag], tag=1, name="air")

        for dim, tag in gmsh.model.getEntities(1):
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
            if np.isclose(xmax, b, rtol=1e-3):
                gmsh.model.addPhysicalGroup(1, [tag], tag=3, name="outer")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(2)
        return gmsh_to_meshio()
    finally:
        gmsh.finalize()


def build_pec_mesh(f, lc):
    gmsh.initialize()
    try:
        gmsh.model.add("pec")

        phase_velocity = 1 / np.sqrt(epsilon_0 * mu_0)
        wavelength = phase_velocity / f
        wavelength_grid = 2.5
        pec_radius = wavelength / 2
        abc_radius = wavelength_grid * wavelength

        a, b = pec_radius, abc_radius

        # Two concentric circles
        inner = gmsh.model.occ.addDisk(0, 0, 0, a, a)
        outer = gmsh.model.occ.addDisk(0, 0, 0, b, b)
        annulus, _ = gmsh.model.occ.cut([(2, outer)], [(2, inner)])
        gmsh.model.occ.synchronize()

        # Tag boundaries for Dirichlet conditions
        curves = gmsh.model.getBoundary(annulus, oriented=False)
        for dim, tag in curves:
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
            name = "inner" if np.isclose(xmax, a) else "outer"
            gmsh.model.addPhysicalGroup(1, [tag], name=name)
        gmsh.model.addPhysicalGroup(2, [annulus[0][1]], name="domain")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(2)
        return gmsh_to_meshio()
    finally:
        gmsh.finalize()


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


def get_mesh_variables(mesh):
    element_tag = mesh.get_cell_data("gmsh:physical", "triangle")
    line_tag = mesh.get_cell_data("gmsh:physical", "line")
    name_to_tag = {k: int(v[0]) for k, v in mesh.field_data.items()}

    node_coords = mesh.points[:, :2]
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    rho = np.sqrt(x**2 + y**2)

    element_indices = mesh.get_cells_type("triangle")
    line_indices = mesh.get_cells_type("line")
    return (
        element_tag,
        line_tag,
        name_to_tag,
        node_coords,
        x,
        y,
        rho,
        element_indices,
        line_indices,
    )


def assemble_matrices(
    node_coords,
    element_indices,
    free_node_indices,
    gamma_line_indices,
    alpha,
    beta,
    gamma,
    alpha_is_element_wise = False,
    beta_is_element_wise = False
):
    # k_alpha
    k_alpha = assemble_k_alpha(node_coords, element_indices, alpha, alpha_is_element_wise)
    k_alpha_free = k_alpha[np.ix_(free_node_indices, free_node_indices)]

    # k_beta
    k_beta = assemble_k_beta(node_coords, element_indices, beta, beta_is_element_wise)
    k_beta_free = k_beta[np.ix_(free_node_indices, free_node_indices)]

    # k_gamma
    k_gamma = assemble_k_gamma(node_coords, gamma_line_indices, gamma)
    k_gamma_free = k_gamma[np.ix_(free_node_indices, free_node_indices)]
    return k_alpha_free + k_beta_free + k_gamma_free, k_alpha, k_beta, k_gamma


def b_abc_first_order(node_coords, outer_line_indices, x, rho, k_0, E0=1.0):
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
    return b_neumann


def fem_pec_tmz(mesh, f, E0=1.0):
    k_0 = 2.0 * np.pi * f * np.sqrt(epsilon_0 * mu_0)

    (
        _,
        line_tag,
        name_to_tag,
        node_coords,
        x,
        _,
        rho,
        element_indices,
        line_indices,
    ) = get_mesh_variables(mesh)

    # boundary condition lines
    inner_line_indices = line_indices[line_tag == name_to_tag["inner"]]
    outer_line_indices = line_indices[line_tag == name_to_tag["outer"]]

    # materials
    mu_r = np.ones(node_coords.shape[0])
    epsilon_r = np.ones(node_coords.shape[0])

    # free node indices
    element_unique_node_indices = np.unique(element_indices.ravel())
    inner_unique_indices = np.unique(inner_line_indices.ravel())
    dirichlet_node_indices = np.concatenate([inner_unique_indices])
    free_node_indices = np.setdiff1d(
        element_unique_node_indices, dirichlet_node_indices
    )

    k_total_free, k_alpha, _, _ = assemble_matrices(
        node_coords,
        element_indices,
        free_node_indices,
        outer_line_indices,
        1 / mu_r,
        -(k_0**2) * epsilon_r,
        1j * k_0 + 1 / (2 * rho[outer_line_indices]).mean(axis=1),
    )

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
    b_neumann = b_abc_first_order(node_coords, outer_line_indices, x, rho, k_0, E0)[
        free_node_indices
    ]

    # final assembly
    b_total = b_neumann + b_dirichlet

    # solve for phi
    phi = np.zeros(len(node_coords), dtype=np.complex128)
    phi[free_node_indices] = np.asarray(spla.spsolve(k_total_free.tocsc(), b_total))
    return phi


def fem_pec_tez(mesh, f, E0=1.0):
    k_0 = 2.0 * np.pi * f * np.sqrt(epsilon_0 * mu_0)
    H0 = 1.0 / np.sqrt(mu_0 / epsilon_0)

    (
        _,
        line_tag,
        name_to_tag,
        node_coords,
        x,
        _,
        rho,
        element_indices,
        line_indices,
    ) = get_mesh_variables(mesh)

    # boundary condition lines
    inner_line_indices = line_indices[line_tag == name_to_tag["inner"]]
    outer_line_indices = line_indices[line_tag == name_to_tag["outer"]]

    # materials
    mu_r = np.ones(node_coords.shape[0])
    epsilon_r = np.ones(node_coords.shape[0])

    # free node indices
    element_unique_node_indices = np.unique(element_indices.ravel())
    free_node_indices = element_unique_node_indices

    k_total_free, _, _, _ = assemble_matrices(
        node_coords,
        element_indices,
        free_node_indices,
        outer_line_indices,
        1 / epsilon_r,
        -(k_0**2) * mu_r,
        1j * k_0 + 1 / (2 * rho[outer_line_indices]).mean(axis=1),
    )

    # b_neumann
    b_neumann = b_abc_first_order(node_coords, outer_line_indices, x, rho, k_0, H0)[
        free_node_indices
    ]

    # final assembly
    b_total = b_neumann

    # solve for phi
    phi = np.zeros(len(node_coords), dtype=np.complex128)
    phi[free_node_indices] = np.asarray(spla.spsolve(k_total_free.tocsc(), b_total))
    return phi


def fem_die_tmz(mesh, f, E0=1.0):
    k_0 = 2.0 * np.pi * f * np.sqrt(epsilon_0 * mu_0)

    (
        element_tag,
        line_tag,
        name_to_tag,
        node_coords,
        x,
        _,
        rho,
        element_indices,
        line_indices,
    ) = get_mesh_variables(mesh)

    # boundary condition lines
    outer_line_indices = line_indices[line_tag == name_to_tag["outer"]]

    # materials
    mu_r = np.ones(node_coords.shape[0])
    epsilon_r = np.zeros(element_indices.shape[0])
    epsilon_r[element_tag == name_to_tag["air"]] = 1.0
    epsilon_r[element_tag == name_to_tag["dielectric"]] = 9.0

    # free node indices
    element_unique_node_indices = np.unique(element_indices.ravel())
    free_node_indices = element_unique_node_indices

    k_total_free, _, _, _ = assemble_matrices(
        node_coords,
        element_indices,
        free_node_indices,
        outer_line_indices,
        1 / mu_r,
        -(k_0**2) * epsilon_r,
        1j * k_0 + 1 / (2 * rho[outer_line_indices]).mean(axis=1),
        False,
        True
    )

    # b_neumann
    b_neumann = b_abc_first_order(node_coords, outer_line_indices, x, rho, k_0, E0)[
        free_node_indices
    ]

    # final assembly
    b_total = b_neumann

    # solve for phi
    phi = np.zeros(len(node_coords), dtype=np.complex128)
    phi[free_node_indices] = np.asarray(spla.spsolve(k_total_free.tocsc(), b_total))
    return phi


def fem_die_tez(mesh, f, E0=1.0):
    k_0 = 2.0 * np.pi * f * np.sqrt(epsilon_0 * mu_0)
    H0 = E0 / np.sqrt(mu_0 / epsilon_0)

    (
        element_tag,
        line_tag,
        name_to_tag,
        node_coords,
        x,
        _,
        rho,
        element_indices,
        line_indices,
    ) = get_mesh_variables(mesh)

    # boundary condition lines
    outer_line_indices = line_indices[line_tag == name_to_tag["outer"]]

    # materials
    mu_r = np.ones(node_coords.shape[0])
    epsilon_r = np.zeros(element_indices.shape[0], dtype=np.complex128)
    epsilon_r[element_tag == name_to_tag["air"]] = 1.0
    epsilon_r[element_tag == name_to_tag["dielectric"]] = 9.0

    # free node indices

    element_unique_node_indices = np.unique(element_indices.ravel())
    free_node_indices = element_unique_node_indices

    k_total_free, _, _, _ = assemble_matrices(
        node_coords,
        element_indices,
        free_node_indices,
        outer_line_indices,
        1 / epsilon_r,
        -(k_0**2) * mu_r,
        1j * k_0 + 1 / (2 * rho[outer_line_indices]).mean(axis=1),
        True,
        False
    )

    # b_neumann
    b_neumann = b_abc_first_order(node_coords, outer_line_indices, x, rho, k_0, H0)[
        free_node_indices
    ]

    # final assembly
    b_total = b_neumann

    # solve for phi
    phi = np.zeros(len(node_coords), dtype=np.complex128)
    phi[free_node_indices] = np.asarray(spla.spsolve(k_total_free.tocsc(), b_total))
    return phi


def bistatic_echo_width_die_tmz_analytical(phi: np.ndarray):
    f = 10e9
    epsilon_r = 9

    phase_velocity_0 = 1 / np.sqrt(epsilon_0 * mu_0)
    phase_velocity_1 = 1 / np.sqrt(epsilon_r * epsilon_0 * mu_0)
    wavelength = phase_velocity_0 / f
    a = wavelength / 2
    k = 2.0 * np.pi * f / phase_velocity_0
    kd = 2.0 * np.pi * f / phase_velocity_1
    ka = k * a
    kda = kd * a

    N = 15
    ns = np.arange(-N, N + 1)
    coefficients = (
        np.sqrt(epsilon_r) * jn(ns, ka) * jvp(ns, kda) - jvp(ns, ka) * jn(ns, kda)
    ) / (
        h2vp(ns, ka) * jn(ns, kda) - np.sqrt(epsilon_r) * hankel2(ns, ka) * jvp(ns, kda)
    )
    return (4.0 / k) * np.abs(
        (coefficients[:, None] * np.exp(-1j * ns[:, None] * phi[None, :])).sum(axis=0)
    ) ** 2


def bistatic_echo_width_pec_tmz_analytical(phi: np.ndarray):
    f = 10e9
    phase_velocity = 1 / np.sqrt(epsilon_0 * mu_0)
    wavelength = phase_velocity / f
    a = wavelength / 2
    k = 2.0 * np.pi * f / phase_velocity
    ka = k * a

    N = 15
    ns = np.arange(-N, N + 1)
    coefficients = jn(ns, ka) / hankel2(ns, ka)
    return (4.0 / k) * np.abs(
        (coefficients[:, None] * np.exp(-1j * ns[:, None] * phi[None, :])).sum(axis=0)
    ) ** 2


def bistatic_echo_width_die_tez_analytical(phi: np.ndarray):
    f = 10e9
    epsilon_r = 9

    E0 = 1.0
    H0 = E0 / np.sqrt(mu_0 / epsilon_0)

    phase_velocity_0 = 1 / np.sqrt(epsilon_0 * mu_0)
    phase_velocity_1 = 1 / np.sqrt(epsilon_r * epsilon_0 * mu_0)
    wavelength = phase_velocity_0 / f
    a = wavelength / 2
    k = 2.0 * np.pi * f / phase_velocity_0
    kd = 2.0 * np.pi * f / phase_velocity_1
    ka = k * a
    kda = kd * a

    N = 15
    ns = np.arange(-N, N + 1)
    coefficients = (
        H0
        * (
            (1.0 / np.sqrt(epsilon_r)) * jn(ns, ka) * jvp(ns, kda)
            - jvp(ns, ka) * jn(ns, kda)
        )
        / (
            h2vp(ns, ka) * jn(ns, kda)
            - (1.0 / np.sqrt(epsilon_r)) * hankel2(ns, ka) * jvp(ns, kda)
        )
    )
    return (4.0 / k) * np.abs(
        (coefficients[:, None] * np.exp(-1j * ns[:, None] * phi[None, :])).sum(axis=0)
    ) ** 2


def bistatic_echo_width_pec_tez_analytical(phi: np.ndarray):
    f = 10e9
    E0 = 1.0
    H0 = E0 / np.sqrt(mu_0 / epsilon_0)
    phase_velocity = 1 / np.sqrt(epsilon_0 * mu_0)
    wavelength = phase_velocity / f
    a = wavelength / 2
    k = 2.0 * np.pi * f / phase_velocity
    ka = k * a

    N = 15
    ns = np.arange(-N, N + 1)
    coefficients = H0 * jvp(ns, ka) / h2vp(ns, ka)
    return (4.0 / k) * np.abs(
        (coefficients[:, None] * np.exp(-1j * ns[:, None] * phi[None, :])).sum(axis=0)
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


def fem_plot2(phi, mesh, savefig=False, savepath="default.png"):
    nodes = mesh.points[:, :2]
    elements = mesh.get_cells_type("triangle")

    fig, ax = plt.subplots(constrained_layout=True)
    triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=elements)

    pc = ax.tripcolor(triang, phi, shading="gouraud", cmap="hot")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

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
