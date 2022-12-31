#!/usr/bin/env python3

from collections import OrderedDict

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import Button, Div, RadioButtonGroup, Slider, Spacer
from bokeh.plotting import figure, curdoc
import shapely.affinity
from shapely.geometry import Polygon

from planners import prm, prm_star, rrt


def random_polygons(xmax, ymax, rmax, N):
    cfgs = np.random.rand(N, 4)

    polys = []
    for (x, y, r, rot) in cfgs:
        x = xmax * x
        y = ymax * y
        r = rmax * r
        rot = 90.0 * rot

        pts = [
            (x + r * np.cos(z), y + r * np.sin(z))
            for z in np.linspace(0.0, 2.0 * np.pi, num=4, endpoint=False)
        ]

        p = Polygon(pts)
        p = shapely.affinity.rotate(p, rot)
        polys.append(p)
    return polys


def plot_polygon(p, polys, **kwargs):
    default_opts = dict(alpha=0.5, line_width=2)
    opts = default_opts | kwargs

    xs = [poly.exterior.xy[0] for poly in polys]
    ys = [poly.exterior.xy[1] for poly in polys]

    p.patches(xs, ys, **opts)


# create the scene
obstacles = random_polygons(1.0, 1.0, 0.1, N=15)

# create the figure
p = figure(toolbar_location=None)

# draw obstacles
plot_polygon(p, obstacles, alpha=0.15, color="red")

# placeholders for generated plans
q = p.multi_line(
    [], [], line_alpha=0.2, line_color="gray", line_width=1.5, line_dash="dashed"
)
m = p.line([], [], line_color="tomato", line_width=2)
r = p.circle(size=7.0, color="cornflowerblue")
z = p.triangle_dot([], [], size=20.0, fill_color="aliceblue", color="cornflowerblue")


def update(resample=False, **kwargs):
    global V, params
    params = params | kwargs

    # Some parameter updates can reuse the sampled vertices. Do so if possible,
    # because it makes comparisons of parameter behavior easier in the UI.
    if resample:
        V = None

    # Run the planning algorithm and extract some common results.
    res = params["algo"](obstacles, V=V, **params)
    V = res["V"]
    E = res["E"]
    path = res["path"]

    # Halt document updates until we are finished updating all the data, then
    # blast them all at once.
    curdoc().hold("combine")

    # update vertex list
    dat = {}
    dat["x"] = V[:, 0]
    dat["y"] = V[:, 1]
    r.data_source.data = dat

    # update edges
    xs = []
    ys = []
    for i in range(0, len(V)):
        for j in range(i, len(V)):
            if E[i, j] > 0 or E[j, i] > 0:
                xs.append((V[i, 0], V[j, 0]))
                ys.append((V[i, 1], V[j, 1]))

    dat = {}
    dat["xs"] = xs
    dat["ys"] = ys
    q.data_source.data = dat

    # update shortest path
    dat = {}
    dat["x"] = [V[i, 0] for i in path]
    dat["y"] = [V[i, 1] for i in path]
    m.data_source.data = dat

    # update init and goal poses
    dat = {}
    dat["x"] = [res[n][0] for n in ["x_init", "x_goal"]]
    dat["y"] = [res[n][1] for n in ["x_init", "x_goal"]]
    z.data_source.data = dat

    curdoc().unhold()


params = {
    "algo": prm,
    "sample_count": 50,
    "connectivity_radius": 0.5,
    "k_prm": 1.0,
    "steer_alpha": 0.5,
}
update(resample=True, **params)


algo_steps = {
    "PRM": [
        r"$$V \gets \{x_{init}\} \cup \{\text{SampleFree}_i\}_{i=1,\ldots,n}; E \gets \emptyset$$",
        r"$$\textbf{foreach } v \in V \textbf{ do}$$",
        r"$$\quad U \gets \text{Near}(G = (V, E), v, r) \setminus \{v\}$$",
        r"$$\quad \textbf{foreach } u \in U \textbf{ do}$$",
        r"$$\qquad \textbf{if } \text{CollisionFree}(v, u) \textbf{ then } E \gets E \cup \{(v, u), (u, v)\}$$",
        r"$$P \gets \text{ShortestPath}(G = (V, E), x_{init}, x_{goal})$$",
    ],
    "PRM*": [
        r"$$V \gets \{x_{init}\} \cup \{\text{SampleFree}_i\}_{i=1,\ldots,n}; E \gets \emptyset$$",
        r"$$\textbf{foreach } v \in V \textbf{ do}$$",
        r"$$\quad U \gets \text{Near}(G = (V, E), v, \gamma_{PRM}(\log(n) / n)^{1 / d}) \setminus \{v\}$$",
        r"$$\quad \textbf{foreach } u \in U \textbf{ do}$$",
        r"$$\qquad \textbf{if } \text{CollisionFree}(v, u) \textbf{ then } E \gets E \cup \{(v, u), (u, v)\}$$",
        r"$$P \gets \text{ShortestPath}(G = (V, E), x_{init}, x_{goal})$$",
    ],
    "RRT*": [
        r"$$V \gets \{x_{init}\}; E \gets \emptyset$$",
        r"$$\textbf{for } i,\ldots,n \textbf{ do}$$",
        r"$$\quad x_{rand} \gets \text{ SampleFree}$$",
        r"$$\quad x_{nearest} \gets \text{ Nearest}(G = (V, E), x_{rand})$$",
        r"$$\quad x_{new} \gets \text{ Steer}(x_{nearest}, x_{rand}, \alpha)$$",
        r"$$\quad \textbf{if } \text{CollisionFree}(x_{nearest}, x_{new}) \textbf{ then }$$",
        r"$$\qquad V \gets V \cup \{ x_{new} \}; E \gets E \cup \{ (x_{nearest}, x_{new}) \}$$",
        r"$$P \gets \text{ShortestPath}(G = (V, E), x_{init}, x_{goal})$$",
    ],
}

replan_button = Button(label="Replan!")
replan_button.on_event("button_click", lambda: update(resample=True))


sample_count_slider = Slider(
    start=1,
    end=200,
    value=50,
    step=1,
    title="Sample Count",
)
sample_count_slider.on_change(
    "value_throttled",
    lambda attr, old, new: update(resample=True, sample_count=new),
)

connectivity_radius_slider = Slider(
    start=0,
    end=1,
    value=params["connectivity_radius"],
    step=0.01,
    title="Connectivity Radius",
)
connectivity_radius_slider.on_change(
    "value_throttled", lambda attr, old, new: update(connectivity_radius=new)
)

gamma_scale_slider = Slider(
    start=1.0,
    end=2.0,
    value=params["k_prm"],
    step=0.01,
    title=r"$$k$$ (as in $$k \cdot \gamma^*_{PRM}$$)",
)
gamma_scale_slider.on_change(
    "value_throttled", lambda attr, old, new: update(k_prm=new)
)


steer_alpha_slider = Slider(
    start=0.0,
    end=1.0,
    value=params["steer_alpha"],
    step=0.01,
    title=r"Steer Linear Blend Coefficient ($$\alpha$$)",
)
steer_alpha_slider.on_change(
    "value_throttled", lambda attr, old, new: update(steer_alpha=new, resample=True)
)

algos = OrderedDict([("sPRM", prm), ("PRM*", prm_star), ("RRT", rrt)])

algo_step_divs = [
    column(
        *[
            Div(
                text=f"{i + 1}. $$\\,$$ {step}",
            )
            for (i, step) in enumerate(steps)
        ],
        Spacer(height=10),
        Div(text=r"Adapted from Karaman & Frazzoli, 2011", align="end"),
        width=400,
        visible=i == 0,
    )
    for i, steps in enumerate(algo_steps.values())
]


def on_change_algorithm(attr, old, new):
    new_algo = list(algos.keys())[new]

    for (algo, col) in zip(algos.keys(), algo_step_divs):
        col.visible = algo == new_algo

    connectivity_radius_slider.visible = new_algo == "sPRM"
    gamma_scale_slider.visible = new_algo == "PRM*"
    steer_alpha_slider.visible = new_algo == "RRT"

    update(resample=True, algo=algos[new_algo])


on_change_algorithm(None, None, 0)

algo_radio = RadioButtonGroup(
    labels=list(algos.keys()), active=0, button_type="primary"
)
algo_radio.on_change("active", on_change_algorithm)

curdoc().title = "Planner Playground"
curdoc().add_root(
    column(
        Div(
            text='<h3 style="color: #333;">Planner Playground</h3>',
            margin=(0, 0, 0, 30),
        ),
        row(
            p,
            column(
                Spacer(height=10),
                *algo_step_divs,
                column(
                    row(algo_radio, replan_button),
                    sample_count_slider,
                    connectivity_radius_slider,
                    gamma_scale_slider,
                    steer_alpha_slider,
                    Spacer(height=50),
                    margin=(50, 10, 0, 10),
                ),
            ),
        ),
    )
)
