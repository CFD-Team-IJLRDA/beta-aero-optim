import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from shapely.geometry import LinearRing

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

EPS = 1e-6


# General purpose geometric functions
# these are theoretically compatible with any 2D geometry defined as an ordered set of points
def get_area(pts: np.ndarray) -> float:
    """
    **Returns** the geometry signed area computed with the shoelace formula.</br>
    see https://rosettacode.org/wiki/Shoelace_formula_for_polygonal_area#Python
    """
    x, y = zip(*pts)
    return abs( sum(i * j for i, j in zip(x,             y[1:] + y[:1]))
               -sum(i * j for i, j in zip(x[1:] + x[:1], y            ))) / 2


def get_camber_th(
        upper: np.ndarray,
        lower: np.ndarray,
        interpolate: bool = False
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """
    **Returns** an approximation of the camber line, of the geometry absolute (th)
    and axial (th_x) thicknesses, and the coordinates of the points maximizing the thickness.

    If interpolate is set to True, the upper and lower profiles are interpolated with scipy
    which gives a more precise estimation of the thicknesses.
    """
    min_dx = []
    min_dvec = []
    if interpolate:
        c, _ = get_chords(np.vstack((upper, lower)))
        int_upper_f = scipy.interpolate.interp1d(upper[:, 0], upper[:, 1])
        int_lower_f = scipy.interpolate.interp1d(lower[:, 0], lower[:, 1])
        xnew_lower = np.arange(min(lower[:, 0]) + EPS, max(lower[:, 0]), 0.0025 * c)
        xnew_upper = np.arange(min(upper[:, 0]) + EPS, max(upper[:, 0]), 0.0025 * c)
        int_upper = int_upper_f(xnew_upper)
        int_lower = int_lower_f(xnew_lower)
        upper = np.column_stack((xnew_upper, int_upper))
        lower = np.column_stack((xnew_lower, int_lower))
    for x in upper:
        d_vec = np.sqrt(np.einsum("ij,ij->i", lower - x, lower - x))
        idx_min = np.argmin(d_vec)
        min_dx.append(idx_min)
        min_dvec.append(d_vec[idx_min])
    # lower_n is the vector of points minimizing the distance wrt the upper points
    # such that (upper[i], lower_n[i]) forms the pair of closest points between both sides
    lower_n = lower[min_dx]
    camber_line = (upper + lower_n) / 2.
    th_idx = np.argmax(min_dvec)
    le_x = np.min(np.vstack((upper, lower)), axis=0)[0]
    # the pair of points with the maximal thickness
    th_vec = np.array([upper[th_idx], lower_n[th_idx]])
    return camber_line, min_dvec[th_idx], camber_line[th_idx][0] - le_x, th_vec


def get_chords(pts: np.ndarray) -> tuple[float, float]:
    """
    **Returns** chord (c) and axial chord (c_ax).
    """
    idx_le, idx_te = get_edges_idx(pts)
    return float(np.linalg.norm(pts[idx_le] - pts[idx_te])), float((pts[idx_te] - pts[idx_le])[0])


def get_circle(origin: np.ndarray, r: float) -> np.ndarray:
    """
    **Returns** the coordinates of the points on the circle centered on origin with radius r.
    """
    theta = np.linspace(0., 2 * np.pi, 100)
    x_c = r * np.cos(theta) + origin[0]
    y_c = r * np.sin(theta) + origin[1]
    return np.column_stack((x_c, y_c))


def get_circle_centers(upper: np.ndarray, lower: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    **Returns** the origins of the edge circles as the mean point of the curvature extrema
    on each tip of the profile.

    Note:
        this function is suited for naca/blade profiles with smooth tips
        i.e. where the point density is significantly higher than on the rest of the geometry.
    """
    # upper side
    s = get_curv_abs(upper)
    d_s = np.gradient(s)
    u_dd_s = np.gradient(d_s)
    u_min_dd_s = np.argmin(u_dd_s)
    u_max_dd_s = np.argmax(u_dd_s)

    # lower side
    s = get_curv_abs(lower)
    d_s = np.gradient(s)
    l_dd_s = np.gradient(d_s)
    l_min_dd_s = np.argmin(l_dd_s)
    l_max_dd_s = np.argmax(l_dd_s)

    mean_le_pt = 0.5 * (upper[u_max_dd_s] + lower[l_min_dd_s])
    mean_te_pt = 0.5 * (upper[u_min_dd_s] + lower[l_max_dd_s])
    return mean_le_pt, mean_te_pt


def get_cog(pts: np.ndarray) -> np.ndarray:
    """
    **Returns** the coordinates of the geometry's center of gravity.
    """
    area = get_area(pts)
    x = pts[:, 0]; y = pts[:, 1]
    x_rot = np.roll(x, -1); y_rot = np.roll(y, -1)
    cross = x * y_rot - x_rot * y
    area_signed = np.sum(cross) / 2         
    x_cg = np.sum((x + x_rot) * cross) / (6 * area_signed)
    y_cg = np.sum((y + y_rot) * cross) / (6 * area_signed)
    return np.array([x_cg, y_cg])


def get_curv_abs(pts: np.ndarray) -> np.ndarray:
    """
    **Returns** the curvilinear abscissa vector of the given points.
    """
    s: list[float] = [0.] * len(pts)
    for pt_id in range(1, len(pts)):
        d = np.linalg.norm(pts[pt_id] - pts[pt_id - 1])
        s[pt_id] = s[pt_id - 1] + float(d)
    return np.array(s)


def get_edges_idx(pts: np.ndarray) -> tuple[int, int]:
    """
    **Returns** the indices of the leading/trailing edges
    i.e. computed from the left/right-most points.
    """
    return np.argmin(pts, axis=0)[0], np.argmax(pts, axis=0)[0]


def split_profile(pts: np.ndarray,) -> tuple[np.ndarray, np.ndarray]:
    """
    **Returns** the upper and lower parts wrt the leading/trailing edges.
    """
    idx_le, idx_te = get_edges_idx(pts)
    start: int = min(idx_le, idx_te)
    end: int = max(idx_le, idx_te)
    if (
        max([p[1] for p in pts[start:end + 1]])
        > max([p[1] for p in np.vstack((pts[:start + 1], pts[end:]))])
    ):
        upper = pts[start:end + 1]
        if pts[-1][0] > pts[start][0]:
            lower = np.vstack((pts[end:], pts[:start + 1]))
        else:
            lower = np.vstack((pts[:start + 1], pts[end:]))
    else:
        lower = pts[start:end + 1]
        if pts[start][0] > pts[end][0]:
            upper = np.vstack((pts[:start + 1], pts[end:]))
        else:
            upper = np.vstack((pts[end:], pts[:start + 1]))
    return upper, lower


# Constraint verification functions
# return float value based on whether the constraint is violated (> 0) or not (< 0)
def get_radius_violation(pts: np.ndarray, origin: np.ndarray, d: float) -> float:
    """
    **Returns** the value of the difference between the given value d
    and the minimal origin to profile distance.

    If this value is positive, the circle of radius d centered on origin
    does not fit inside the profile. If the value is negative, it does.

    Note:
        this mechanism complies with the way pymoo handles constraints.
    """
    pts_dist = np.sqrt(np.einsum("ij,ij->i", pts[:, :2] - origin, pts[:, :2] - origin))
    return np.min(d - pts_dist)

def adim_and_rotate_chord(x, y, i_LE, i_TE):
    """
    Adimensionalitze profile coordinates wrt of the computed chord, shift them to have LE in (0,0),
    rotate the blade for -stagger angle (computed inside the function).
    """
    xm = np.min(x)
    xM = np.max(x)
    Cax = xM - xm

    LE = np.array([x[i_LE], y[i_LE]])
    TE = np.array([x[i_TE], y[i_TE]])

    chord = np.linalg.norm(TE - LE)
    stagger_angle = np.degrees(np.arctan2(TE[1] - LE[1], TE[0] - LE[0]))

    # shift so LE at origin, scale by true chord
    Xb = (x - LE[0]) / chord
    Yb = (y - LE[1]) / chord

    # rotate 
    rad = np.radians(-stagger_angle)
    cos_r, sin_r = np.cos(rad), np.sin(rad)
    xr = Xb * cos_r - Yb * sin_r
    yr = Xb * sin_r + Yb * cos_r

    return Xb, Yb, xr, yr, chord, Cax, stagger_angle

def split_airfoil(prof, i_LE, i_TE):
    """
    Splits a closed-loop blade profile into upper/lower surface arcs,
    given known LE/TE indices.
    **Returns** idx_upper, idx_lower: 1D integer arrays of indices into prof,
    ordered the same way the point arrays used to be (each INCLUDING
    the LE and TE indices as their first/last entries, no gap at the
    apexes).
    """
    n = len(prof)
    LE = prof[i_LE]
    TE = prof[i_TE]

    # --- Build the two arcs between LE and TE
    # Arc A: 
    if i_LE <= i_TE:
        idx_A = np.arange(i_LE, i_TE + 1)
    else:
        idx_A = np.concatenate([np.arange(i_LE, n), np.arange(0, i_TE + 1)])

    # Arc B: 
    if i_TE <= i_LE:
        idx_B = np.arange(i_TE, i_LE + 1)
    else:
        idx_B = np.concatenate([np.arange(i_TE, n), np.arange(0, i_LE + 1)])

    # --- Label upper vs lower using signed distance to the LE->TE chord ---
    v = TE - LE

    def mean_signed_offset(idx):
        arc = prof[idx]
        cross = v[0] * (arc[:, 1] - LE[1]) - v[1] * (arc[:, 0] - LE[0])
        return np.mean(cross)

    if mean_signed_offset(idx_A) > mean_signed_offset(idx_B):
        idx_upper, idx_lower = idx_A, idx_B
    else:
        idx_upper, idx_lower = idx_B, idx_A

    return idx_upper, idx_lower

def find_LE_TE(pts):
    """
    Finds the index of the point closest to TE and LE (via farthest-pair
    + curvature refinement).
    **Returns** index of LE and TE in the profile array.
    """
    x = pts[:,0]
    y = pts[:,1]
    n = len(pts)

    # farthest pair, orientation-independent
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)

    # refine with local curvature (max turning angle in a window)
    def refine(idx, window=8):
        lo, hi = idx - window, idx + window
        best_idx, best_ang = idx, -1
        for k in range(lo, hi):
            p0, p1, p2 = pts[(k-1) % n], pts[k % n], pts[(k+1) % n]
            v1, v2 = p1 - p0, p2 - p1
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-12 or n2 < 1e-12:
                continue
            cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            ang = np.arccos(cosang)
            if ang > best_ang:
                best_ang, best_idx = ang, k % n
        return best_idx

    i_ref, j_ref = refine(i), refine(j)

    # associate to LE or TE
    if x[i_ref] > x[j_ref]:
        idx_TE = i_ref
        idx_LE = j_ref
    else:
        idx_TE = j_ref
        idx_LE = i_ref

    return idx_LE, idx_TE

def min_max_thickness(xr, yr, upper_idx, lower_idx, n_check=150):
    """
    Checks minimum and maximum thickness of the blade profile.
    """
    #x_up, y_up = profile[upper_idx, 0], profile[upper_idx, 1]
    #x_lo, y_lo = profile[lower_idx, 0], profile[lower_idx, 1]
    x_up, y_up = xr[upper_idx], yr[upper_idx]
    x_lo, y_lo = xr[lower_idx], yr[lower_idx]

    order_up = np.argsort(x_up)
    order_lo = np.argsort(x_lo)
    x_up, y_up = x_up[order_up], y_up[order_up]
    x_lo, y_lo = x_lo[order_lo], y_lo[order_lo]

    x_common = np.linspace(max(x_up.min(), x_lo.min()),
                            min(x_up.max(), x_lo.max()), n_check)
    y_up_i = np.interp(x_common, x_up, y_up)
    y_lo_i = np.interp(x_common, x_lo, y_lo)

    thickness = y_up_i - y_lo_i

    idx_min = np.argmin(thickness)
    idx_max = np.argmax(thickness)

    th_min = thickness[idx_min]
    th_max = thickness[idx_max]
    Xth_min = x_common[idx_min]
    Xth_max = x_common[idx_max]

    return th_min, th_max, Xth_min, Xth_max

def corner_radius(profile, window):
    """
    Local radius of curvature near a corner (e.g. LE or TE),
    `window` = slice of point indices around that corner.
    """
    pts = profile[window]
    dx, dy = np.gradient(pts[:, 0]), np.gradient(pts[:, 1])
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-12)**1.5

    return 1.0 / (curvature.max() + 1e-12)

def is_self_intersecting(profile):
    # Remove duplicated points
    diffs = np.diff(profile, axis=0)
    dup_mask = np.all(np.abs(diffs) < 1e-12, axis=1)
    keep = np.concatenate([[True], dup_mask])

    ring = LinearRing(profile[keep])          # closed loop of (x,y) points
    
    return not ring.is_simple

# Plotting functions
# for visual assessment and debugging purposes
def plot_profile(pts: np.ndarray, cog: np.ndarray = np.array([]), figname: str = ""):
    """
    **Plots** the complete profile and other optional attributes
    such as the center of gravity and the leading/trailing edges.

    If figname exists, the graph is saved to figname (i.e. figname includes the path).
    """
    idx_le, idx_te = get_edges_idx(pts)
    # Figure
    fsize = (12, 4)
    _, ax = plt.subplots(figsize=fsize)
    ax.plot(pts[:, 0], pts[:, 1], label="profile")
    dy = 1.5 / 100. * (np.max(pts, axis=1)[1] - np.min(pts, axis=1)[1])
    # leading edges
    ax.scatter(pts[idx_le][0], pts[idx_le][1], c="red", s=40, marker="+", zorder=40)
    ax.annotate("phys. le", xy=(pts[idx_le][0], pts[idx_le][1]),
                xytext=(pts[idx_le][0], pts[idx_le][1] + dy), color="red")
    # trailing edges
    ax.scatter(pts[idx_te][0], pts[idx_te][1], c="black", s=40, marker="+", zorder=40)
    ax.annotate("phys. te", xy=(pts[idx_te][0], pts[idx_te][1]),
                xytext=(pts[idx_te][0], pts[idx_te][1] + dy), color="black")
    # CoG
    if cog.size > 0:
        ax.scatter(cog[0], cog[1], c="green", s=12)
        ax.annotate("CoG", (cog[0], cog[1]), color="green")
    # legend and display
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if figname:
        plt.savefig(figname, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_sides(
        baseline: np.ndarray,
        profile: np.ndarray,
        Delta: float = 0.005,
        figname: str = ""
):
    """
    **Plots** the upper and lower sides of the profile and other optional attributes
    such as the camber line, the leading/trailing edge circles and the maximal thickness.

    If figname exists, the graph is saved to figname (i.e. figname includes the path).
    """
    fsize = (12, 4)
    _, ax = plt.subplots(figsize=fsize)   
    ax.plot(baseline[:, 0], baseline[:, 1], color="k", linestyle="dashed", label="baseline")
    ax.plot(profile[:, 0], profile[:, 1], linewidth=1, label=f"Rejected candidate")
    ax.set(xlabel='$x$ [m]', ylabel='$y$ [m]')
    ax.legend()

    # === Leading Edge Inset ===
    # axins1 = inset_axes(ax, width="30%", height="30%", loc="lower center", borderpad=2)
    axins1 = ax.inset_axes([0.27, 0.05, 0.35, 0.35])
    axins1.plot(baseline[:, 0], baseline[:, 1], color="k", linestyle="dashed")
    axins1.plot(profile[:, 0], profile[:, 1], linewidth=1)
    axins1.set_xlim(min(baseline[:, 0]) - 0.15 * Delta, min(baseline[:, 1]) + 2 * Delta)
    axins1.set_ylim(min(baseline[:, 1]) - 0.25 * Delta, min(baseline[:, 1]) + Delta)
    axins1.set_xticks([])
    axins1.set_yticks([])
    mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.5")

    # === Trailing Edge Inset ===
    # axins2 = inset_axes(ax, width="30%", height="30%", loc="center right", borderpad=2)
    axins2 = ax.inset_axes([0.63, 0.3, 0.35, 0.35])
    axins2.plot(baseline[:, 0], baseline[:, 1], color="k", linestyle="dashed")
    axins2.plot(profile[:, 0], profile[:, 1], linewidth=1)
    axins2.set_xlim(max(baseline[:, 0]) - 3 * Delta, max(baseline[:, 0]) + 0.4 * Delta)
    axins2.set_ylim(max(baseline[:, 1]) - 1.25 * Delta, max(baseline[:, 1]) + 0.3 * Delta)
    axins2.set_xticks([])
    axins2.set_yticks([])
    mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.tight_layout()
    if figname:
        plt.savefig(figname, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    
# def plot_sides(
#         upper: np.ndarray,
#         lower: np.ndarray,
#         camber: np.ndarray = np.array([]),
#         le_circle: np.ndarray = np.array([]),
#         te_circle: np.ndarray = np.array([]),
#         th_vec: np.ndarray = np.array([]),
#         figname: str = ""
# ):
#     """
#     **Plots** the upper and lower sides of the profile and other optional attributes
#     such as the camber line, the leading/trailing edge circles and the maximal thickness.

#     If figname exists, the graph is saved to figname (i.e. figname includes the path).
#     """
#     # Figure
#     fsize = (12, 4)
#     _, ax = plt.subplots(figsize=fsize)
#     ax.plot(upper[0, :], upper[1, :], label="upper side")
#     ax.plot(lower[0, :], lower[1, :], label="lower side")
#     if camber.size > 0:
#         ax.plot(camber[:, 0], camber[:, 1], label="camber line")
#     if le_circle.size > 0:
#         ax.plot(le_circle[:, 0], le_circle[:, 1], label="le circle")
#         axins = zoomed_inset_axes(ax, 3.5, loc="upper left")
#         axins.plot(upper[:, 0], upper[:, 1])
#         axins.plot(lower[:, 0], lower[:, 1])
#         axins.plot(camber[:, 0], camber[:, 1])
#         axins.plot(le_circle[:, 0], le_circle[:, 1], linestyle="dotted")
#         axins.scatter(
#             np.sum(le_circle[:, 0]) / len(le_circle),
#             np.sum(le_circle[:, 1]) / len(le_circle),
#             s=20, marker="+"
#         )
#         axins.set_xlim(-0.00075, 0.01)
#         axins.set_ylim(-0.0012, 0.005)
#         plt.xticks(visible=False)
#         plt.yticks(visible=False)
#         mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
#     if te_circle.size > 0:
#         ax.plot(te_circle[:, 0], te_circle[:, 1], label="te circle")
#         axins = zoomed_inset_axes(ax, 3.5, loc="lower right")
#         axins.plot(upper[:, 0], upper[:, 1])
#         axins.plot(lower[:, 0], lower[:, 1])
#         axins.plot(camber[:, 0], camber[:, 1])
#         axins.plot(te_circle[:, 0], te_circle[:, 1], linestyle="dotted")
#         axins.scatter(
#             np.sum(te_circle[:, 0]) / len(te_circle),
#             np.sum(te_circle[:, 1]) / len(te_circle),
#             s=20, marker="+"
#         )
#         axins.set_xlim(0.065, 0.068)
#         axins.set_ylim(0.018, 0.021)
#         plt.xticks(visible=False)
#         plt.yticks(visible=False)
#         mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
#     if th_vec.size > 0:
#         ax.plot(th_vec[:, 0], th_vec[:, 1], label="th_max")
#     # legend and display
#     ax.set_xlabel("x [m]")
#     ax.set_ylabel("y [m]")
#     ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
#     if figname:
#         plt.savefig(figname, bbox_inches='tight')
#     else:
#         plt.show()
#     plt.close()
