#!/usr/bin/env python3

import numpy as np
import scipy.sparse.csgraph
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, Point


def inside_any_polygon(xy, polys):
    """
    Test whether a point is within any of a collection of polygons.

    Parameters
    ----------
    xy: array_like
        The point under test.
    polys: list[shapely.geometry.Polygon]
        The list of polygons which should be tested for collision with `xy`
    """
    pt = Point(xy)
    return any([pt.within(poly) for poly in polys])


def sample_free(obstacles, N):
    """
    Sample points which do not collide with any obstacles.

    Parameters
    ----------
    obstacles: list[shapely.geometry.Polygon]
        The polygons which should be avoided.
    N: int
        The number of points to return.

    Returns
    -------
    Exactly `N` points which do not collide with the given obstacles.
    """
    samples = []

    while len(samples) != N:
        xy = np.random.rand(2)
        if not inside_any_polygon(xy, obstacles):
            samples.append(xy)

    return np.array(samples)


def near(pt, pts, r):
    """
    Keep points which fall within a given radius of a central point.

    Parameters
    ----------
    pt: array_like
        The central point about which other points should be near.
    pts: array_like
        The list of points which will be filtered.
    r: scalar
        The maximum distance of any returned point.

    Returns
    -------
    The indices in `pts` which are less than or equal `r` distance from `pt`.
    """
    return np.where(cdist(pts, pt[None]) <= r)[0]


def collision_free(obstacles, v, u):
    """
    Test whether a line drawn between two points is free of collision with a
    list of obstacles.

    Parameters
    ----------
    obstacles: list[shapely.geometry.Polygon]
        The obstacles which should be tested for collision.
    v, u: array_like
        The endpoints of the line (order not important).

    Returns
    -------
    If the line drawn between `v` and `u` avoids collision with the obstacle
    set.
    """
    line = LineString([v, u])
    return not any([line.intersects(o) for o in obstacles])


def find_shortest_path(E, i, j):
    """
    Find the shortest path between two vertices of a weighted adjacency matrix.

    Parameters
    ----------
    G: scipy.sparse.csr_matrix
        A sparse matrix representing the connected edges of the graph.
    i, j: int
        The indices of the start and end points of the path.

    Returns
    -------
    The list of vertex indices that form a path from the `i` to `j`.
    """
    D, Pr = scipy.sparse.csgraph.shortest_path(
        E, indices=[i, j], directed=False, return_predecessors=True
    )

    # Helper to find a path given the matrix of predecessors.
    # Source: https://stackoverflow.com/a/53078901
    def get_path(Pr, i, j):
        path = [j]
        k = j
        while Pr[i, k] != -9999:
            path.append(Pr[i, k])
            k = Pr[i, k]
        return path[::-1]

    return get_path(Pr, i, j)


def nearest(V, u):
    """
    Return the point from a set of points which is nearest to another point (by
    2-norm).

    Parameters
    ----------
    V: array_like
        The list of points from which the nearest candidate should be drawn.
    u: array_like
        The central point.

    Returns
    -------
    The point on `V` nearest to `u`.
    """

    V = np.asarray(V)
    u = np.asarray(u)

    return np.argmin(np.linalg.norm(V - u, axis=1))


def steer(u, v, alpha=0.5):
    """
    Perform linear interpolation between two points.

    Parameters
    ----------
    u, v: array_like
        1-D arrays of real values.
    alpha: scalar
        The blend ratio (0 being all `u`, 1 being all `v`).

    Returns
    -------
    The point on `V` nearest to `u`.
    """

    u = np.asarray(u)
    v = np.asarray(v)

    w = v - u
    return u + alpha * w


def prm(obstacles, V=None, sample_count=50, connectivity_radius=0.33, **kwargs):
    x_init = [0.0, 0.0]
    x_goal = [1.0, 1.0]

    if V is None:
        V = np.concatenate(([x_init], sample_free(obstacles, sample_count), [x_goal]))

    # build a sparse graph of connected vertices
    r, c, d = [], [], []

    for i, v in enumerate(V):
        U = near(v, V, r=connectivity_radius)
        for j in U:
            u = V[j]
            if not np.equal(v, u).all() and collision_free(obstacles, v, u):
                r.append(i)
                c.append(j)
                d.append(np.linalg.norm(v - u))

    E = csr_matrix((d, (r, c)), shape=(len(V), len(V)))
    path = find_shortest_path(E, 0, len(V) - 1)

    return {"x_init": x_init, "x_goal": x_goal, "V": V, "E": E, "path": path}


def prm_star(obstacles, sample_count=50, k_prm=1.0, **kwargs):
    mu_xfree = 0.5  # estimate volume to be 50% obstacles (TODO)
    zeta_d = np.pi * 0.5**2
    gamma_star_prm = 2 * (1 + 1 / 2) ** (1 / 2) * (mu_xfree / zeta_d) ** (1 / 2)
    gamma_prm = k_prm * gamma_star_prm

    connectivity_radius = gamma_prm * (np.log(sample_count) / sample_count) ** (1.0 / 2)

    return prm(
        obstacles,
        sample_count=sample_count,
        **kwargs | dict(connectivity_radius=connectivity_radius)
    )


def rrt(obstacles, sample_count=50, steer_alpha=0.5, **kwargs):
    x_init = [0.0, 0.0]
    x_goal = [1.0, 1.0]

    V = [x_init]

    # build a sparse graph of connected vertices
    r, c, d = [], [], []

    S = sample_free(obstacles, N=sample_count)

    j = 1  # skip start
    for x_rand in S:
        i = nearest(V, x_rand)
        x_nearest = V[i]
        x_new = steer(x_nearest, x_rand, alpha=steer_alpha)

        if collision_free(obstacles, x_nearest, x_new):
            V.append(x_new)

            c.append(i)
            r.append(j)
            d.append(np.linalg.norm(x_new - x_nearest))

            j += 1

    E = csr_matrix((d, (r, c)), shape=(len(V), len(V)))
    path = find_shortest_path(E, 0, nearest(V, np.array(x_goal)))

    return {"x_init": x_init, "x_goal": x_goal, "V": np.array(V), "E": E, "path": path}
