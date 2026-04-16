import os
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt

# Optional dependency used as fallback converter
try:
    import meshio  # pip install meshio
except Exception:
    meshio = None

import gmsh  # pip install gmsh


def read_gmsh_mesh(mesh_file):
    """
    Read a mesh (Gmsh .msh or foreign formats) into Gmsh, extracting nodes/elements.
    Tries:
      1) gmsh.open for .msh
      2) gmsh.merge for foreign formats (.mesh, .stl, ...)
      3) meshio -> temp .msh -> gmsh.open (fallback)
    Returns:
        nodes: dict {node_tag: [x,y,z]}
        elements: dict {elem_tag: [node_tags]}
        element_types: dict {elem_tag: elem_type}
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("imported")

    def _extract():
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = {}
        if node_tags is None or len(node_tags) == 0:
            return {}, {}, {}
        for i, tag in enumerate(node_tags):
            j = 3 * i
            nodes[int(tag)] = [node_coords[j], node_coords[j + 1], node_coords[j + 2]]

        elements = {}
        element_types = {}
        # Collect all element types present
        types = gmsh.model.mesh.getElementTypes()
        for et in types:
            etags, enodes = gmsh.model.mesh.getElementsByType(et)
            if len(etags) == 0:
                continue
            npe = len(enodes) // len(etags)
            for i, etag in enumerate(etags):
                start = i * npe
                end = start + npe
                elements[int(etag)] = [int(n) for n in enodes[start:end]]
                element_types[int(etag)] = int(et)
        return nodes, elements, element_types

    original_error = None
    try:
        try:
            # Native Gmsh files
            if mesh_file.lower().endswith(".msh"):
                gmsh.open(mesh_file)
            else:
                # Try importing foreign mesh directly
                gmsh.merge(mesh_file)
        except Exception as e1:
            # Save original gmsh error (avoid calling gmsh.finalize() here)
            original_error = e1
            # Fallback: convert with meshio to Gmsh v2.2 and reopen
            if meshio is None:
                raise RuntimeError(f"Failed to read mesh with Gmsh and meshio not installed. Original error: {e1}") from e1
            with tempfile.TemporaryDirectory() as td:
                tmp_msh = os.path.join(td, "converted.msh")
                try:
                    m = meshio.read(mesh_file)
                except Exception as e_read:
                    # include both errors for diagnostics
                    raise RuntimeError(f"meshio failed to read '{mesh_file}': {e_read}. Original gmsh error: {e1}") from e_read
                try:
                    meshio.write(tmp_msh, m, file_format="gmsh22")
                    gmsh.open(tmp_msh)
                except Exception as e_open:
                    # include both errors for diagnostics
                    raise RuntimeError(f"Failed to open converted mesh with Gmsh: {e_open}. Original gmsh error: {e1}") from e_open

        nodes, elements, element_types = _extract()
        return nodes, elements, element_types
    finally:
        try:
            gmsh.finalize()
        except Exception:
            # If finalize fails, don't mask the original exception; just ignore
            pass
def plot_mesh_2d(nodes, elements, highlighted_nodes=None, title="Gmsh Mesh Visualization"):
    fig, ax = plt.subplots(figsize=(10, 8))
    # Draw element edges (tri, quad, line)
    for etag, conn in elements.items():
        if len(conn) == 2:  # line
            if conn[0] in nodes and conn[1] in nodes:
                pts = [nodes[conn[0]], nodes[conn[1]]]
                ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], 'k-', lw=0.5, alpha=0.3)
        elif len(conn) == 3:  # tri
            poly = np.array([nodes[n] for n in conn + [conn[0]]])
            ax.plot(poly[:, 0], poly[:, 1], 'k-', lw=0.4, alpha=0.2)
        elif len(conn) == 4:  # quad
            poly = np.array([nodes[n] for n in conn + [conn[0]]])
            ax.plot(poly[:, 0], poly[:, 1], 'k-', lw=0.4, alpha=0.2)

    coords = np.array(list(nodes.values()))
    ax.plot(coords[:, 0], coords[:, 1], 'o', ms=2, color='blue', alpha=0.5, label='All nodes')

    if highlighted_nodes:
        highlighted_coords = np.array([nodes[tag] for tag in highlighted_nodes if tag in nodes])
        if len(highlighted_coords) > 0:
            ax.plot(highlighted_coords[:, 0], highlighted_coords[:, 1], 
                   'o', markersize=8, color='red', alpha=0.8, 
                   label=f'Highlighted nodes ({len(highlighted_coords)})')
            # Add labels to highlighted nodes
            for tag in highlighted_nodes:
                if tag in nodes:
                    ax.text(nodes[tag][0], nodes[tag][1], str(tag), 
                           fontsize=8, ha='right', va='bottom', color='red')

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_mesh_3d(nodes, elements, highlighted_nodes=None, title="Gmsh Mesh Visualization 3D"):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates
    node_tags = list(nodes.keys())
    coords = np.array([nodes[tag] for tag in node_tags])
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    z_coords = coords[:, 2]

    # Plot all edges from elements
    for elem_tag, node_list in elements.items():
        if len(node_list) >= 2:  # At least a line element
            # Plot edges between consecutive nodes
            for i in range(len(node_list)):
                n1 = node_list[i]
                n2 = node_list[(i + 1) % len(node_list)]
                if n1 in nodes and n2 in nodes:
                    x_vals = [nodes[n1][0], nodes[n2][0]]
                    y_vals = [nodes[n1][1], nodes[n2][1]]
                    z_vals = [nodes[n1][2], nodes[n2][2]]
                    ax.plot(x_vals, y_vals, z_vals, 'k-', linewidth=0.3, alpha=0.2)

    # Plot all nodes
    ax.scatter(x_coords, y_coords, z_coords, c='blue', marker='o', s=10, alpha=0.3, label='All nodes')

    # Highlight specific nodes
    if highlighted_nodes:
        highlighted_coords = np.array([nodes[tag] for tag in highlighted_nodes if tag in nodes])
        if len(highlighted_coords) > 0:
            ax.scatter(highlighted_coords[:, 0], highlighted_coords[:, 1], highlighted_coords[:, 2],
                      c='red', marker='o', s=100, alpha=0.9, edgecolors='black', linewidths=2,
                      label=f'Highlighted nodes ({len(highlighted_coords)})')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def print_mesh_info(nodes, elements, element_types):
    print("\n" + "=" * 60)
    print("MESH INFORMATION")
    print("=" * 60)
    print(f"Total nodes: {len(nodes)}")
    print(f"Total elements: {len(elements)}")
    # Count element types
    counts = {}
    for et in element_types.values():
        counts[et] = counts.get(et, 0) + 1
    print("Element type counts:", counts)
    coords = np.array(list(nodes.values())) if nodes else np.zeros((0, 3))
    if len(coords):
        print(f"Bounding box X:[{coords[:,0].min():.6f}, {coords[:,0].max():.6f}] "
              f"Y:[{coords[:,1].min():.6f}, {coords[:,1].max():.6f}] "
              f"Z:[{coords[:,2].min():.6f}, {coords[:,2].max():.6f}]")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage:
    # Pass a mesh path as first arg, else change this default
    mesh_file = "/home/mciarlatani/Hilbert/beta-aero-optim/Optimization/FailedSims/FailedADP/ogv1c_g1_c14.mesh"
    highlighted_nodes = [1, 23729]

    print(f"Reading mesh from: {mesh_file}")
    try:
        nodes, elements, element_types = read_gmsh_mesh(mesh_file)
    except Exception as e:
        print(f"Failed to read mesh: {e}")
        sys.exit(1)

    print_mesh_info(nodes, elements, element_types)

    # Decide 2D vs 3D
    coords = np.array(list(nodes.values())) if nodes else np.zeros((0, 3))
    is_2d = len(coords) > 0 and np.allclose(coords[:, 2], 0.0)

    if is_2d:
        print("Detected 2D mesh -> 2D visualization")
        plot_mesh_2d(nodes, elements, highlighted_nodes, title=os.path.basename(mesh_file))
    else:
        print("Detected 3D mesh -> 3D visualization")
        plot_mesh_3d(nodes, elements, highlighted_nodes, title=os.path.basename(mesh_file))

    plt.show()