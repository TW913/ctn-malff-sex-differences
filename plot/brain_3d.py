"""Create anatomical slice views of activation NIfTI files.

This is a Python counterpart of ``brain_3d.m``.  The default paths mirror the
MATLAB script, but all input/output paths and plotting parameters can be
overridden from the command line.

Dependencies: numpy, scipy, nibabel, matplotlib and pillow.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import affine_transform


DEEP_BLUE = np.array([68, 87, 181], dtype=float) / 255.0
LIGHT_BLUE = np.array([24, 154, 225], dtype=float) / 255.0
YELLOW_START = np.array([255, 225, 0], dtype=float) / 255.0
ORANGE_END = np.array([255, 96, 31], dtype=float) / 255.0
NEAR_WHITE = np.array([0.98, 0.98, 0.98], dtype=float)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Render activation NIfTI files on an MNI152 template."
    )
    parser.add_argument("--label-atlas", type=Path, default=script_dir / "AAL3v1_1mm.nii")
    parser.add_argument("--template", type=Path, default=script_dir / "mni152.nii")
    parser.add_argument("--activation-folder", type=Path, default=script_dir / "nii")
    parser.add_argument("--output-folder", type=Path, default=script_dir / "python results")
    parser.add_argument("--act-thresh", type=float, default=0.0)
    parser.add_argument("--color-threshold", type=float, default=2.0)
    parser.add_argument("--slice-tolerance", type=float, default=1.0)
    parser.add_argument(
        "--num-slices",
        "--num-y-slices",
        dest="num_slices",
        type=int,
        default=10,
        help="Number of evenly spaced slices to export for each X, Y and Z direction.",
    )
    parser.add_argument("--export-resolution", type=int, default=1200)
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove existing PNG/TIFF files from the output folder.",
    )
    return parser.parse_args()


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float64))
    return data, np.asarray(image.affine, dtype=float), image


def resample_label_atlas_to_template(
    atlas_data: np.ndarray,
    atlas_affine: np.ndarray,
    template_shape: tuple[int, ...],
    template_affine: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour atlas resampling in world coordinates."""
    # scipy maps output indices to input indices, which is exactly
    # inv(atlas_affine) @ template_affine for NIfTI voxel coordinates.
    transform = np.linalg.solve(atlas_affine, template_affine)
    atlas_on_template = affine_transform(
        np.nan_to_num(atlas_data, nan=0.0),
        transform[:3, :3],
        offset=transform[:3, 3],
        output_shape=template_shape,
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    rounded = np.rint(atlas_on_template)
    if not np.allclose(atlas_on_template, rounded, atol=1e-4):
        raise ValueError("AAL3v1 atlas contains non-integer values; a discrete label atlas is required.")
    return rounded.astype(np.int32, copy=False)


def same_nifti_space(
    data_a: np.ndarray, affine_a: np.ndarray, data_b: np.ndarray, affine_b: np.ndarray
) -> bool:
    return (
        data_a.shape == data_b.shape
        and np.all(np.isfinite(affine_a))
        and np.all(np.isfinite(affine_b))
        and np.max(np.abs(affine_a - affine_b)) <= 1e-4
    )


def voxel_to_world(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=float)
    return indices @ affine[:3, :3].T + affine[:3, 3]


def interpolate_rgb(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return np.asarray(end, dtype=float)[None, :]
    return np.column_stack([np.linspace(start[i], end[i], n) for i in range(3)])


def interpolate_value_range(
    values: np.ndarray,
    range_start: float,
    range_end: float,
    start_rgb: np.ndarray,
    end_rgb: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not values.size:
        return np.empty((0, 3), dtype=float)
    if not np.isfinite(range_start) or not np.isfinite(range_end) or abs(range_end - range_start) < np.finfo(float).eps:
        return np.repeat(np.asarray(end_rgb)[None, :], values.size, axis=0)
    t = np.clip((values - range_start) / (range_end - range_start), 0.0, 1.0)
    return np.asarray(start_rgb)[None, :] + t[:, None] * (np.asarray(end_rgb) - np.asarray(start_rgb))[None, :]


def sample_activation_colors(values: np.ndarray, color_limit: float, color_threshold: float) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    colors = np.repeat(NEAR_WHITE[None, :], values.size, axis=0)
    color_limit = max(float(color_limit), np.finfo(float).eps)
    color_threshold = max(float(color_threshold), 0.0)
    if color_limit <= color_threshold:
        negative = values < 0
        positive = values > 0
        colors[negative] = interpolate_value_range(values[negative], -color_limit, 0, DEEP_BLUE, NEAR_WHITE)
        colors[positive] = interpolate_value_range(values[positive], 0, color_limit, NEAR_WHITE, ORANGE_END)
        return colors

    masks_and_ranges = (
        (values <= -color_threshold, -color_limit, -color_threshold, DEEP_BLUE, LIGHT_BLUE),
        ( (values > -color_threshold) & (values < 0), -color_threshold, 0, LIGHT_BLUE, NEAR_WHITE),
        ( (values > 0) & (values < color_threshold), 0, color_threshold, NEAR_WHITE, YELLOW_START),
        (values >= color_threshold, color_threshold, color_limit, YELLOW_START, ORANGE_END),
    )
    for mask, start, end, start_rgb, end_rgb in masks_and_ranges:
        colors[mask] = interpolate_value_range(values[mask], start, end, start_rgb, end_rgb)
    return colors


def activation_colormap(color_limit: float, color_threshold: float, n: int = 256) -> ListedColormap:
    values = np.linspace(-max(color_limit, np.finfo(float).eps), max(color_limit, np.finfo(float).eps), n)
    return ListedColormap(sample_activation_colors(values, color_limit, color_threshold), name="activation")


def save_colorbar_images(
    color_limit: float, color_threshold: float, output_folder: Path, base_name: str
) -> None:
    if color_limit > color_threshold:
        negative_values = np.linspace(-color_limit, -color_threshold, 256)
        positive_values = np.linspace(color_threshold, color_limit, 256)
    else:
        negative_values = np.linspace(-color_limit, 0, 256)
        positive_values = np.linspace(0, color_limit, 256)

    for label, values in (("negative", negative_values), ("positive", positive_values)):
        gradient = sample_activation_colors(values, color_limit, color_threshold)
        height, width, border = 90, 2200, 2
        image = np.full((height, width, 3), [0.20, 0.20, 0.20], dtype=float)
        sample_indices = np.rint(np.linspace(0, len(gradient) - 1, width - 2 * border)).astype(int)
        gradient_strip = gradient[sample_indices]
        image[border:-border, border:-border] = gradient_strip[np.newaxis, :, :]
        path = output_folder / f"{base_name}_colorbar_{label}.png"
        Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8), mode="RGB").save(path)
        print(f"Saved {label} colorbar [{values[0]:.4f}, {values[-1]:.4f}]: {path}")


def create_white_background_template_rgb(template_slice: np.ndarray) -> np.ndarray:
    values = np.asarray(template_slice, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        scaled = np.zeros(values.shape, dtype=float)
    else:
        lo, hi = np.nanmin(values), np.nanmax(values)
        scaled = np.zeros(values.shape, dtype=float) if hi <= lo else (values - lo) / (hi - lo)
        scaled[~finite] = 0
    mask = scaled > 0
    rgb = np.ones((*scaled.shape, 3), dtype=float)
    rgb[mask] = 0.75 * scaled[mask, None]
    return np.clip(rgb, 0, 1)


def axis_world_coordinates(affine: np.ndarray, shape: tuple[int, int, int], axis: int) -> np.ndarray:
    indices = np.zeros((shape[axis], 3), dtype=float)
    indices[:, axis] = np.arange(shape[axis], dtype=float)
    return voxel_to_world(indices, affine)[:, axis]


def save_slice(
    world_coord: float,
    axis: str,
    tolerance: float,
    template: np.ndarray,
    affine: np.ndarray,
    world_points: np.ndarray,
    activation_values: np.ndarray,
    color_limit: float,
    color_threshold: float,
    save_path: Path,
    export_resolution: int,
) -> None:
    axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    shape = template.shape
    axis_coords = axis_world_coordinates(affine, shape, axis_index)
    index = int(np.argmin(np.abs(axis_coords - world_coord)))

    if axis_index == 0:
        template_slice = template[index, :, :].T
        coords1 = axis_world_coordinates(affine, shape, 1)
        coords2 = axis_world_coordinates(affine, shape, 2)
        point_mask = np.abs(world_points[:, 0] - world_coord) <= tolerance
        point_columns = (1, 2)
        labels = ("Y (mm)", "Z (mm)")
    elif axis_index == 1:
        template_slice = template[:, index, :].T
        coords1 = axis_world_coordinates(affine, shape, 0)
        coords2 = axis_world_coordinates(affine, shape, 2)
        point_mask = np.abs(world_points[:, 1] - world_coord) <= tolerance
        point_columns = (0, 2)
        labels = ("X (mm)", "Z (mm)")
    else:
        template_slice = template[:, :, index].T
        coords1 = axis_world_coordinates(affine, shape, 0)
        coords2 = axis_world_coordinates(affine, shape, 1)
        point_mask = np.abs(world_points[:, 2] - world_coord) <= tolerance
        point_columns = (0, 1)
        labels = ("X (mm)", "Y (mm)")

    points = world_points[point_mask][:, point_columns]
    values = activation_values[point_mask]
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(values)
        & (points[:, 0] >= coords1.min())
        & (points[:, 0] <= coords1.max())
        & (points[:, 1] >= coords2.min())
        & (points[:, 1] <= coords2.max())
    ) if points.size else np.zeros(0, dtype=bool)
    points, values = points[valid], values[valid]

    # Axial (Z) slices are narrower than sagittal/coronal slices in the
    # template coordinate system.  Match the canvas aspect to the data extent
    # so a fixed 6 x 5 inch canvas does not leave large side margins.
    figure_height = 5.0
    figure_width = 6.0
    if axis_index == 2:
        data_width = abs(float(coords1.max() - coords1.min()))
        data_height = abs(float(coords2.max() - coords2.min()))
        if data_width > 0 and data_height > 0:
            figure_width = figure_height * data_width / data_height
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=export_resolution)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    rgb = create_white_background_template_rgb(template_slice)
    ax.imshow(
        rgb,
        extent=(float(coords1.min()), float(coords1.max()), float(coords2.min()), float(coords2.max())),
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )
    if points.size:
        ax.scatter(
            points[:, 0], points[:, 1], s=40, c=values,
            cmap=activation_colormap(color_limit, color_threshold),
            norm=Normalize(-color_limit, color_limit), marker="s", linewidths=0, alpha=0.9,
        )
    ax.set_xlim(coords1.min(), coords1.max())
    ax.set_ylim(coords2.min(), coords2.max())
    if axis.lower() == "x":
        # Keep every sagittal output consistent with a left-side view.
        ax.invert_xaxis()
    ax.axis("off")
    fig.savefig(save_path, dpi=export_resolution, facecolor="white", edgecolor="white")
    plt.close(fig)
    if not points.size:
        print(f"No slice points found near {axis.upper()} = {world_coord:.2f} +/- {tolerance:.2f} mm")
    print(f"Saved slice view: {save_path}")


def slice_components(
    axis: str,
    world_coord: float,
    template: np.ndarray,
    affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, str], float]:
    """Return a template slice, its world-coordinate axes and voxel index."""
    axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    shape = template.shape
    axis_coords = axis_world_coordinates(affine, shape, axis_index)
    index = int(np.argmin(np.abs(axis_coords - world_coord)))
    if axis_index == 0:
        template_slice = template[index, :, :].T
        coords1 = axis_world_coordinates(affine, shape, 1)
        coords2 = axis_world_coordinates(affine, shape, 2)
        labels = ("Y (mm)", "Z (mm)")
    elif axis_index == 1:
        template_slice = template[:, index, :].T
        coords1 = axis_world_coordinates(affine, shape, 0)
        coords2 = axis_world_coordinates(affine, shape, 2)
        labels = ("X (mm)", "Z (mm)")
    else:
        template_slice = template[:, :, index].T
        coords1 = axis_world_coordinates(affine, shape, 0)
        coords2 = axis_world_coordinates(affine, shape, 1)
        labels = ("X (mm)", "Y (mm)")
    return template_slice, coords1, coords2, labels, float(axis_coords[index])


def slice_activation_rgba(
    axis: str,
    index_world_coord: float,
    activation: np.ndarray,
    activation_affine: np.ndarray,
    template: np.ndarray,
    template_affine: np.ndarray,
    color_limit: float,
    color_threshold: float,
    act_thresh: float,
    allowed_mask: np.ndarray | None,
) -> np.ndarray | None:
    """Make a crisp, voxel-aligned RGBA activation overlay for a slice."""
    if not same_nifti_space(activation, activation_affine, template, template_affine):
        return None
    axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    coords = axis_world_coordinates(activation_affine, activation.shape, axis_index)
    index = int(np.argmin(np.abs(coords - index_world_coord)))
    if axis_index == 0:
        values = activation[index, :, :].T
        mask = np.isfinite(values) & (np.abs(values) > act_thresh)
        if allowed_mask is not None:
            mask &= allowed_mask[index, :, :].T > 0
    elif axis_index == 1:
        values = activation[:, index, :].T
        mask = np.isfinite(values) & (np.abs(values) > act_thresh)
        if allowed_mask is not None:
            mask &= allowed_mask[:, index, :].T > 0
    else:
        values = activation[:, :, index].T
        mask = np.isfinite(values) & (np.abs(values) > act_thresh)
        if allowed_mask is not None:
            mask &= allowed_mask[:, :, index].T > 0
    cmap = activation_colormap(color_limit, color_threshold)
    norm = Normalize(-color_limit, color_limit)
    rgba = cmap(norm(np.clip(np.nan_to_num(values, nan=0.0), -color_limit, color_limit)))
    # Keep the activation edge voxel-aligned, but fade very small values so
    # the anatomical template remains visible underneath.
    strength = np.clip(np.abs(np.nan_to_num(values, nan=0.0)) / max(color_threshold, 1e-6), 0.25, 0.95)
    rgba[..., 3] = np.where(mask, strength, 0.0)
    return rgba


def draw_publication_slice(
    ax,
    axis: str,
    world_coord: float,
    template: np.ndarray,
    template_affine: np.ndarray,
    activation: np.ndarray,
    activation_affine: np.ndarray,
    world_points: np.ndarray,
    activation_values: np.ndarray,
    color_limit: float,
    color_threshold: float,
    act_thresh: float,
    tolerance: float,
    allowed_mask: np.ndarray | None,
    show_title: bool = True,
) -> None:
    template_slice, coords1, coords2, labels, actual_coord = slice_components(
        axis, world_coord, template, template_affine
    )
    extent = (float(coords1.min()), float(coords1.max()), float(coords2.min()), float(coords2.max()))
    ax.imshow(
        create_white_background_template_rgb(template_slice),
        extent=extent,
        origin="lower",
        interpolation="bicubic",
        aspect="equal",
    )
    overlay = slice_activation_rgba(
        axis, actual_coord, activation, activation_affine, template, template_affine,
        color_limit, color_threshold, act_thresh, allowed_mask,
    )
    if overlay is not None:
        ax.imshow(overlay, extent=extent, origin="lower", interpolation="nearest", aspect="equal")
    else:
        axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        point_mask = np.abs(world_points[:, axis_index] - actual_coord) <= tolerance
        point_columns = tuple(i for i in range(3) if i != axis_index)
        points = world_points[point_mask][:, point_columns]
        values = activation_values[point_mask]
        if points.size:
            ax.scatter(
                points[:, 0], points[:, 1], s=22, c=values,
                cmap=activation_colormap(color_limit, color_threshold),
                norm=Normalize(-color_limit, color_limit), marker="s", linewidths=0, alpha=0.9,
            )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    if axis.lower() == "x":
        # A sagittal image is shown as seen from the patient's left side.
        ax.invert_xaxis()
    ax.axis("off")
    if show_title:
        titles = {"x": "Sagittal", "y": "Coronal", "z": "Axial"}
        ax.set_title(
            f"{titles[axis.lower()]}  {axis.upper()} = {actual_coord:.1f} mm",
            fontsize=13,
            fontweight="bold",
            pad=8,
        )


def save_orthogonal_slices(
    template: np.ndarray,
    template_affine: np.ndarray,
    activation: np.ndarray,
    activation_affine: np.ndarray,
    allowed_mask: np.ndarray | None,
    world_points: np.ndarray,
    activation_values: np.ndarray,
    color_limit: float,
    color_threshold: float,
    act_thresh: float,
    tolerance: float,
    output_path: Path,
) -> None:
    coords = [
        best_slice_coord(i, activation_affine, activation.shape, world_points, tolerance)[0]
        for i in range(3)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), dpi=220)
    fig.patch.set_facecolor("white")
    for ax, axis, coord in zip(axes, ("x", "y", "z"), coords):
        draw_publication_slice(
            ax, axis, coord, template, template_affine, activation, activation_affine,
            world_points, activation_values, color_limit, color_threshold, act_thresh,
            tolerance, allowed_mask,
        )
    fig.suptitle("Activation pattern on anatomical template", fontsize=17, fontweight="bold", y=0.98)
    sm = plt.cm.ScalarMappable(norm=Normalize(-color_limit, color_limit), cmap=activation_colormap(color_limit, color_threshold))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.055, pad=0.06, aspect=45)
    cbar.set_label("Activation value (Z)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    fig.savefig(output_path, dpi=220, facecolor="white", edgecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved orthogonal slice montage: {output_path}")


def save_projection_montage(
    axis: str,
    axis_coords: np.ndarray,
    template: np.ndarray,
    template_affine: np.ndarray,
    activation: np.ndarray,
    activation_affine: np.ndarray,
    allowed_mask: np.ndarray | None,
    world_points: np.ndarray,
    activation_values: np.ndarray,
    color_limit: float,
    color_threshold: float,
    act_thresh: float,
    tolerance: float,
    output_path: Path,
) -> None:
    """Save a multi-slice projection with labels outside the brain panels."""
    axis = axis.lower()
    coords = np.asarray(axis_coords, dtype=float).ravel()
    if not coords.size:
        return
    ncols = min(5, max(1, coords.size))
    nrows = int(math.ceil(coords.size / ncols))
    panel_height = 3.05
    panel_width = 3.25
    if axis == "z":
        _, layout_coords1, layout_coords2, _, _ = slice_components(
            axis, float(coords[0]), template, template_affine
        )
        data_width = abs(float(layout_coords1.max() - layout_coords1.min()))
        data_height = abs(float(layout_coords2.max() - layout_coords2.min()))
        if data_width > 0 and data_height > 0:
            panel_width = panel_height * data_width / data_height

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * panel_width, nrows * panel_height),
        dpi=220,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    axes_flat = axes.ravel()
    titles = {"x": "Sagittal", "y": "Coronal", "z": "Axial"}
    for ax, coord in zip(axes_flat, coords):
        draw_publication_slice(
            ax,
            axis,
            float(coord),
            template,
            template_affine,
            activation,
            activation_affine,
            world_points,
            activation_values,
            color_limit,
            color_threshold,
            act_thresh,
            tolerance,
            allowed_mask,
            show_title=False,
        )
        # Put the coordinate in the reserved margin below the image, not on
        # top of activation clusters or anatomical detail.
        _, _, _, _, actual_coord = slice_components(axis, float(coord), template, template_affine)
        ax.text(
            0.5,
            -0.075,
            f"{axis.upper()} = {actual_coord:.1f} mm",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )
        if axis in {"y", "z"}:
            # Keep hemisphere labels above the panel and clear of the tissue.
            ax.text(
                0.03,
                1.02,
                "L",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                clip_on=False,
            )
            ax.text(
                0.97,
                1.02,
                "R",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                clip_on=False,
            )
    for ax in axes_flat[coords.size :]:
        ax.axis("off")
    fig.suptitle(
        f"{titles[axis]} activation projections",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    if axis == "z":
        # Axial panels need more vertical room than the other directions.
        # A compact title margin and row gap enlarge the brain slices without
        # changing their anatomical aspect ratio.
        fig.subplots_adjust(
            left=0.005,
            right=0.995,
            top=0.91,
            bottom=0.07,
            wspace=-0.08,
            hspace=0.32,
        )
    else:
        fig.subplots_adjust(
            left=0.015,
            right=0.985,
            top=0.78,
            bottom=0.15,
            # A slight overlap in the otherwise borderless axes removes excess
            # whitespace around the small inferior cerebellar slices.
            wspace=-0.08,
            hspace=0.32,
        )
    fig.savefig(output_path, dpi=220, facecolor="white", edgecolor="white")
    plt.close(fig)
    print(f"Saved {axis.upper()} projection montage: {output_path}")


def best_slice_coord(axis: int, affine: np.ndarray, shape: tuple[int, ...], points: np.ndarray, tolerance: float) -> tuple[float, int]:
    coords = axis_world_coordinates(affine, shape, axis)
    counts = np.sum(np.abs(points[:, axis, None] - coords[None, :]) <= tolerance, axis=0)
    index = int(np.argmax(counts))
    return float(coords[index]), int(counts[index])


def clean_output_folder(output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    for path in output_folder.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".tif", ".tiff"}:
            path.unlink()


def main() -> None:
    args = parse_args()
    for path in (args.label_atlas, args.template, args.activation_folder):
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    if not args.activation_folder.is_dir():
        raise NotADirectoryError(args.activation_folder)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        clean_output_folder(args.output_folder)

    activation_paths = sorted(
        p for p in args.activation_folder.glob("*.nii")
        if p.name not in {args.label_atlas.name, args.template.name}
    )
    if not activation_paths:
        raise FileNotFoundError(f"No activation NIfTI files were found in {args.activation_folder}")
    print(f"Found {len(activation_paths)} activation files.")

    template, template_affine, _ = load_nifti(args.template)
    atlas, atlas_affine, _ = load_nifti(args.label_atlas)
    atlas_on_template = resample_label_atlas_to_template(
        atlas, atlas_affine, template.shape, template_affine
    )
    print(f"Loaded AAL3v1 atlas ({np.unique(atlas_on_template[atlas_on_template > 0]).size} labels) and MNI152 display template.")

    for activation_path in activation_paths:
        base_name = activation_path.stem
        print(f"\nProcessing: {activation_path}")
        activation, activation_affine, _ = load_nifti(activation_path)
        valid_values = activation[np.isfinite(activation) & (np.abs(activation) > args.act_thresh)]
        color_limit = float(np.max(np.abs(valid_values))) if valid_values.size else 1.0
        print(f"Color limit for this activation file: {color_limit:.4f}")
        mask = np.isfinite(activation) & (np.abs(activation) > args.act_thresh)
        if same_nifti_space(activation, activation_affine, template, template_affine):
            mask &= atlas_on_template > 0
        indices = np.argwhere(mask)
        if not len(indices):
            print(f"Warning: no suprathreshold voxels found in {activation_path}; skipped.")
            continue
        world_points = voxel_to_world(indices, activation_affine)
        values = activation[tuple(indices.T)]
        save_colorbar_images(color_limit, args.color_threshold, args.output_folder, base_name)

        in_template_space = same_nifti_space(
            activation, activation_affine, template, template_affine
        )

        orthogonal_path = args.output_folder / f"{base_name}_orthogonal_slices.png"
        save_orthogonal_slices(
            template,
            template_affine,
            activation,
            activation_affine,
            atlas_on_template > 0 if in_template_space else None,
            world_points,
            values,
            color_limit,
            args.color_threshold,
            args.act_thresh,
            args.slice_tolerance,
            orthogonal_path,
        )

        # Export the same number of slices in all three anatomical directions.
        # Coordinates are based on the activation extent, matching the MATLAB
        # script's Y-slice behavior while extending it to sagittal and axial views.
        num_slices = max(int(args.num_slices), 1)
        for axis, axis_index in (("x", 0), ("y", 1), ("z", 2)):
            axis_min = float(np.min(world_points[:, axis_index]))
            axis_max = float(np.max(world_points[:, axis_index]))
            axis_slices = np.linspace(axis_min, axis_max, num_slices)
            projection_path = args.output_folder / f"{base_name}_projection_{axis}_slices.png"
            save_projection_montage(
                axis,
                axis_slices,
                template,
                template_affine,
                activation,
                activation_affine,
                atlas_on_template > 0 if in_template_space else None,
                world_points,
                values,
                color_limit,
                args.color_threshold,
                args.act_thresh,
                args.slice_tolerance,
                projection_path,
            )
            for slice_idx, axis_coord in enumerate(axis_slices, start=1):
                path = args.output_folder / f"{base_name}_slice_{axis}_{slice_idx:02d}.png"
                save_slice(
                    float(axis_coord),
                    axis,
                    args.slice_tolerance,
                    template,
                    template_affine,
                    world_points,
                    values,
                    color_limit,
                    args.color_threshold,
                    path,
                    args.export_resolution,
                )


if __name__ == "__main__":
    main()
