from typing import Collection, List, Optional, Tuple, Union, Dict

import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import visdom

def plot_transform(
    fig: go.Figure,
    t: np.array, # 3x4 or 4x4 matrix
    label: str = "",
    linelength: float = 0.1,
    linewidth: int = 5,
) -> None:
    orig = t[:3, 3].squeeze()
    xaxis = t[:3, 0].squeeze()
    yaxis = t[:3, 1].squeeze()
    zaxis = t[:3, 2].squeeze()

    plot_vector(
        fig, orig, xaxis, "red", label=label, width=linewidth, linelength=linelength
    )
    plot_vector(
        fig, orig, yaxis, "green", label=label, width=linewidth, linelength=linelength
    )
    plot_vector(
        fig, orig, zaxis, "blue", label=label, width=linewidth, linelength=linelength
    )


def plot_vector(
    fig: go.Figure,
    p: np.array,
    v: np.array,
    color: str,
    width: int = 5,
    label: str = "",
    linelength: float = 0.1,
) -> None:
    p = p.squeeze()
    v = v.squeeze()

    c = linelength
    v = c * v
    e = p + v  # compute the endpoint

    sx, sy, sz = tuple(p[:3])
    ex, ey, ez = tuple(e[:3])

    fig.add_trace(
        go.Scatter3d(
            x=[sx, ex],
            y=[sy, ey],
            z=[sz, ez],
            mode="lines",
            line={"width": width, "color": color},
            showlegend=False,
            name=label,
        )
    )


def plot_points(
    fig: go.Figure,
    points: np.array, # 3 x N
    color: str = "blue",
    alpha: float = 1.0,
    size: int = 5,
    hovertext: Optional[List[str]] = None,
    name: Optional[str] = None,
    line: Optional[Dict] = None,
) -> None:
    xs, ys, zs = points[0], points[1], points[2]

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker={
                "size": size,
                "color": color,
                "opacity": alpha,
            },
            name=name,
            hovertext=hovertext,
        )
    )

def plot_points_sequence(
    fig: go.Figure,
    points: np.array, # 3 x N
    colorscale: str = 'Viridis',
    size: int = 6,
    hovertext: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> None:
    xs, ys, zs = points[0], points[1], points[2]

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            marker={
                "size": size,
                "color": np.linspace(0, 1, num=points.shape[1]),
                "colorbar": dict(thickness=10),
                "colorscale": colorscale,
                "opacity": 0.8,
            },
            line=dict(
                color='darkblue',
                width=2
            ),
            name=name,
            hovertext=hovertext,
        )
    )


def generate_plotly_loss_figure(loss_dict):
    splits = ["train", "val"]
    fig = go.Figure()
    for split in splits:
        split_dict = loss_dict[split]

        x = split_dict["iterations"]
        ys = []
        labels = []

        for k, v in split_dict.items():
            if k == "iterations":
                continue
            ys.append(v)
            labels.append(k)

        for label, y in zip(labels, ys):
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"{split}_{label}"))
    return fig

class PlotlyScene():
    def __init__(
        self,
        size: Tuple[int, int] = (800, 800),
        x_range: Tuple[int, int] = (-1, 1),
        y_range: Tuple[int, int] = (-1, 1),
        z_range: Tuple[int, int] = (-1, 1),
        show_legend: bool = False,
        aspectmode: str = "cube",
        aspectratio: Dict = None,
    ):
        # Construct a figure for the scene.
        self.figure = go.Figure()

        # Determine some parts of layout.
        self.figure.update_layout(
            autosize=False,
            width=size[0],
            height=size[1],
            showlegend=show_legend,
            margin=dict(l=0, r=0, t=0, b=0),
            scene_aspectmode=aspectmode,
            scene_aspectratio=aspectratio,
            scene=dict(
                xaxis=dict(nticks=4, range=x_range),
                yaxis=dict(nticks=4, range=y_range),
                zaxis=dict(nticks=4, range=z_range),
            ),
            legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1),
        )

    def plot_scene_to_html(self, window_name: str):
        self.figure.write_html(f"{window_name}.html", auto_open=True)

class VisdomVisualizer():
    def __init__(self, vis: visdom.Visdom, env: str):
        self.vis = vis
        self.env = env

    def plot_rgb(self, rgb_hwc: np.array, window_name: str):
        self.vis.image(rgb_hwc.transpose((2, 0, 1)), win=window_name, env=self.env)

    def plot_depth(self, depth: np.array, window_name: str):
        depth_normalized = depth.squeeze() / np.max(depth) * 255
        self.vis.image(
            depth_normalized.astype(np.uint8), win=window_name, env=self.env
        )

    def plot_mask(self, mask: np.array, window_name: str):
        mask_0_255 = mask.squeeze() * 255
        self.vis.image(
            mask_0_255.astype(np.uint8), win=window_name, env=self.env
        )

    def plot_scene(self, scene: PlotlyScene, window_name: str):
        self.vis.plotlyplot(scene.figure, win=window_name, env=self.env)

    def plot_plotlyfigure(self, figure: go.Figure, window_name: str):
        self.vis.plotlyplot(figure, win=window_name, env=self.env)

if __name__ == "__main__":
    vis = visdom.Visdom()

    visualizer = VisdomVisualizer(vis, "main")
    scene = PlotlyScene(
        size=(600, 600), x_range=(-1, 1), y_range=(-1, 1), z_range=(-1, 1)
    )
    plot_transform(scene.figure, np.eye(4), label="world origin")
    scene.plot_scene_to_html("test")
    visualizer.plot_scene(scene, window_name="test")
