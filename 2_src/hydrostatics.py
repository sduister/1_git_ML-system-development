import numpy as np
import trimesh
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt
import matplotlib.pyplot as plt


def read_step_file(path):
    reader = STEPControl_Reader()
    reader.ReadFile(path)
    reader.TransferRoots()
    return reader.OneShape()


def get_bounding_box(shape):
    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)
    return bbox.Get()


def clip_underwater_brep(shape, draught):
    xmin, ymin, zmin, xmax, ymax, zmax = get_bounding_box(shape)
    print(f"📦 Bounding box Z = ({zmin:.2f}, {zmax:.2f})")
    p1 = gp_Pnt(xmin - 1, ymin - 1, zmin - 1)
    p2 = gp_Pnt(xmax + 1, ymax + 1, draught + 1.0)
    return BRepAlgoAPI_Common(shape, BRepPrimAPI_MakeBox(p1, p2).Shape()).Shape()


def brep_to_trimesh(shape):
    BRepMesh_IncrementalMesh(shape, 0.5).Perform()
    vertices, faces, vert_map, index = [], [], {}, 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        tri = BRep_Tool.Triangulation(explorer.Current(), explorer.Current().Location())
        if tri:
            nodes, tris = tri.Nodes(), tri.Triangles()
            for i in range(1, tri.NbNodes() + 1):
                p = nodes.Value(i)
                key = (p.X(), p.Y(), p.Z())
                if key not in vert_map:
                    vert_map[key] = index
                    vertices.append([p.X(), p.Y(), p.Z()])
                    index += 1
            for i in range(1, tri.NbTriangles() + 1):
                t = tris.Value(i)
                faces.append([
                    vert_map[(nodes.Value(t.Value(1)).X(), nodes.Value(t.Value(1)).Y(), nodes.Value(t.Value(1)).Z())],
                    vert_map[(nodes.Value(t.Value(2)).X(), nodes.Value(t.Value(2)).Y(), nodes.Value(t.Value(2)).Z())],
                    vert_map[(nodes.Value(t.Value(3)).X(), nodes.Value(t.Value(3)).Y(), nodes.Value(t.Value(3)).Z())]
                ])
        explorer.Next()
    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=True)
    mesh.apply_scale(0.001)
    mesh.fix_normals()
    return mesh


def compute_waterplane_aft_point(mesh, draught):
    section = mesh.section(plane_origin=[0, 0, draught], plane_normal=[0, 0, 1])
    if section is None:
        return None
    try:
        points = np.vstack(section.discrete)
        idx = np.argmin(points[:, 0])
        return points[idx]  # returns [x, y, z]
    except Exception:
        return None


def compute_cross_section_area(mesh, x_pos):
    section = mesh.section(plane_origin=[x_pos, 0, 0], plane_normal=[1, 0, 0])
    if section is None:
        return 0.0
    try:
        lines = section.discrete
        total_area = 0.0
        for poly in lines:
            if poly.shape[0] >= 3:
                x, y = poly[:, 1], poly[:, 2]
                total_area += 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return total_area
    except Exception:
        return 0.0


def compute_waterplane_area_and_extent(mesh, draught):
    section = mesh.section(plane_origin=[0, 0, draught], plane_normal=[0, 0, 1])
    if section is None:
        return 0.0, 0.0, 0.0
    try:
        lines = section.discrete
        total_area = 0.0
        all_points = []
        for poly in lines:
            if poly.shape[0] >= 3:
                x, y = poly[:, 0], poly[:, 1]
                total_area += 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
                all_points.append(poly)
        combined = np.vstack(all_points)
        Lwl = combined[:, 0].ptp()
        Bwl = combined[:, 1].ptp()
        return total_area, Lwl, Bwl
    except Exception:
        return 0.0, 0.0, 0.0


def compute_amax_at_max_beam(mesh, draught):
    section = mesh.section(plane_origin=[0, 0, draught], plane_normal=[0, 0, 1])
    if section is None:
        return 0.0, None
    try:
        lines = section.discrete
        combined = np.vstack(lines)
        # Bin into X slices and find max Y extent
        x_vals = combined[:, 0]
        y_vals = combined[:, 1]
        bins = np.linspace(np.min(x_vals), np.max(x_vals), 100)
        max_span = 0
        best_x = None
        for i in range(len(bins) - 1):
            xmask = (x_vals >= bins[i]) & (x_vals < bins[i + 1])
            y_slice = y_vals[xmask]
            if y_slice.size > 0:
                span = np.ptp(y_slice)
                if span > max_span:
                    max_span = span
                    best_x = 0.5 * (bins[i] + bins[i + 1])
        if best_x is not None:
            Amax = compute_cross_section_area(mesh, best_x)
            return Amax, best_x
        return 0.0, None
    except Exception:
        return 0.0, None
    

def compute_hydrostatics(mesh, draught, local_point):
    volume = mesh.volume
    bbox = mesh.bounds
    LoS = bbox[1, 0] - bbox[0, 0]
    B = bbox[1, 1] - bbox[0, 1]
    T = draught * 0.001

    midship_x = bbox[0, 0] + 0.5 * LoS
    A_mid = compute_cross_section_area(mesh, midship_x)

    Aw, Lwl, Bwl = compute_waterplane_area_and_extent(mesh, draught * 0.001)
    wetted_surface = mesh.area - Aw

    Amax, x_at_max_beam = compute_amax_at_max_beam(mesh, draught * 0.001)
    Cp = volume / (Amax * Lwl) if Amax * Lwl > 0 else 0

    Cb = volume / (Lwl * Bwl * T) if Lwl * Bwl * T > 0 else 0
    Cm = A_mid / (B * T) if B * T > 0 else 0
    Cwp = Aw / (Lwl * Bwl) if Lwl * Bwl > 0 else 0

    lcb_global = mesh.center_mass[0]
    lcb_local = (lcb_global - local_point[0]) / Lwl * 100 if Lwl > 0 else 0

    print("\n📊 Hydrostatics of underwaterbody:")
    print(f"Length submerged (LoS):         {LoS:.3f} m")
    print(f"Length waterline (Lwl):         {Lwl:.3f} m")
    print(f"Beam at waterline (Bwl):        {Bwl:.3f} m")
    print(f"Draught (T):                    {T:.3f} m")
    print(f"Displacement (∇):               {volume:.3f} m³")
    print(f"LCB from origin:                {lcb_global:.3f} m")
    print(f"LCB from local point:           {lcb_local:.2f} %Lwl")
    print(f"Waterplane Area (Aw):           {Aw:.3f} m²")
    print(f"Wetted Surface Area (S):        {wetted_surface:.3f} m²")
    print(f"Midship Area (Amid):            {A_mid:.3f} m²")
    print(f"Max. Width Area (Amax):         {Amax:.3f} m² (X = {x_at_max_beam:.2f} m)")
    print(f"Block Coefficient (Cb):         {Cb:.3f}")
    print(f"Midship Coefficient (Cm):       {Cm:.3f}")
    print(f"Waterplane Coefficient (Cwp):   {Cwp:.3f}")
    print(f"Prismatic Coefficient (Cp):     {Cp:.3f}")



def plot_views(mesh, local_point):
    points = mesh.vertices
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].scatter(points[:, 0], points[:, 1], s=0.5)
    axs[0].scatter(0, 0, color='black', marker='x', label='Global origin')
    if local_point is not None:
        axs[0].scatter(local_point[0], local_point[1], color='red', marker='x', label='Local point')
    axs[0].set_title("Top View (X-Y)")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].axis("equal")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].scatter(points[:, 0], points[:, 2], s=0.5)
    axs[1].scatter(0, 0, color='black', marker='x', label='Global origin')
    if local_point is not None:
        axs[1].scatter(local_point[0], local_point[2], color='red', marker='x', label='Local point')
    axs[1].set_title("Side View (X-Z)")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Z")
    axs[1].axis("equal")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    from trimesh.scene import Scene
    from trimesh.visual import ColorVisuals

    step_file = r"C:\\Users\\sietse.duister\\OneDrive - De Voogt Naval Architects\\00_specialists group\\1_projects\\2_ML system development\\2_parasolids\\716-VRM-H-90-CASCO-CFD-28112022_x_t.stp"
    draught_m = 3.4
    unit_scale = 1000.0
    draught = draught_m * unit_scale

    print("📥 Reading STEP...")
    full_shape = read_step_file(step_file)
    clipped_shape = clip_underwater_brep(full_shape, draught)
    underwater_mesh = brep_to_trimesh(clipped_shape)
    underwater_mesh.visual = ColorVisuals(underwater_mesh, face_colors=[50, 100, 255, 255])

    local_point = compute_waterplane_aft_point(underwater_mesh, draught * 0.001)
    if local_point is not None:
        print(f"📏 Local point (m): X={local_point[0]:.3f}, Y={local_point[1]:.3f}, Z={local_point[2]:.3f}")
    else:
        print("⚠️ Could not compute local point")

    plot_views(underwater_mesh, local_point)
    compute_hydrostatics(underwater_mesh, draught, local_point)

    print("🧭 Launching 3D viewer...")
    Scene([underwater_mesh]).show()


if __name__ == "__main__":
    main()