"""Render highlighted anatomical slices for selected AAL3 regions.

For every requested ROI, the script creates three whole-brain slices centered
on the ROI MNI centroid (sagittal, coronal, and axial), with both clean and
annotated versions. The default run produces twelve PNG files for AAL3 ROI 17
(left olfactory cortex) and ROI 43 (left parahippocampal gyrus).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform


DEFAULT_ROI_IDS = (17, 43)
ROI_COLORS = {
    17: "#FFD400",  # Bright yellow for left olfactory cortex.
    43: "#E64242",  # Warm red remains distinct beside the yellow ROI 17 highlight.
}
ROI_OUTLINE_COLORS = {
    17: "#6B5200",
    43: "#7A1E1E",
}
ORIENTATIONS = {
    "x": (0, "sagittal", "X"),
    "y": (1, "coronal", "Y"),
    "z": (2, "axial", "Z"),
}


@dataclass(frozen=True)
class RoiInfo:
    roi_id: int
    abbreviation: str
    name: str


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Render highlighted sagittal, coronal, and axial AAL3 ROI slices."
    )
    parser.add_argument("--label-atlas", type=Path, default=script_dir / "AAL3v1_1mm.nii")
    parser.add_argument("--atlas-labels", type=Path, default=script_dir / "aal3.csv")
    parser.add_argument("--template", type=Path, default=script_dir / "mni152.nii")
    parser.add_argument("--output-folder", type=Path, default=script_dir / "python ROI")
    parser.add_argument(
        "--roi-ids",
        type=int,
        nargs="+",
        default=DEFAULT_ROI_IDS,
        help="AAL3 ROI IDs to render. Default: 17 43.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=1200,
        help="PNG export resolution. Default: 1200, matching brain_3d.py slices.",
    )
    return parser.parse_args()


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float64))
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI image: {path}")
    return data, np.asarray(image.affine, dtype=float)


def load_roi_info(path: Path) -> dict[int, RoiInfo]:
    roi_info: dict[int, RoiInfo] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        expected_columns = {"ROIid", "ROIabbr", "ROIname"}
        if reader.fieldnames is None or not expected_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain semicolon-delimited ROIid, ROIabbr, and ROIname columns."
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                roi_id = int((row.get("ROIid") or "").strip())
            except ValueError as error:
                raise ValueError(f"Invalid ROIid at {path}:{line_number}") from error
            abbreviation = (row.get("ROIabbr") or "").strip()
            name = (row.get("ROIname") or "").strip()
            if not abbreviation or not name:
                raise ValueError(f"Incomplete ROI metadata at {path}:{line_number}")
            roi_info[roi_id] = RoiInfo(roi_id, abbreviation, name)
    return roi_info


def resample_label_atlas_to_template(
    atlas_data: np.ndarray,
    atlas_affine: np.ndarray,
    template_shape: tuple[int, int, int],
    template_affine: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor resampling preserves the atlas's discrete ROI values."""
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
        raise ValueError("The label atlas contains non-integer values after resampling.")
    return rounded.astype(np.int32, copy=False)


def voxel_to_world(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=float)
    return indices @ affine[:3, :3].T + affine[:3, 3]


def axis_world_coordinates(
    affine: np.ndarray, shape: tuple[int, int, int], axis: int
) -> np.ndarray:
    indices = np.zeros((shape[axis], 3), dtype=float)
    indices[:, axis] = np.arange(shape[axis], dtype=float)
    return voxel_to_world(indices, affine)[:, axis]


def nearest_slice_index(
    affine: np.ndarray, shape: tuple[int, int, int], axis: int, world_coord: float
) -> tuple[int, float]:
    coordinates = axis_world_coordinates(affine, shape, axis)
    index = int(np.argmin(np.abs(coordinates - world_coord)))
    return index, float(coordinates[index])


def create_template_rgb(template_slice: np.ndarray) -> np.ndarray:
    """Match the darker white-background anatomy rendering used by brain_3d.py."""
    values = np.asarray(template_slice, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        scaled = np.zeros(values.shape, dtype=float)
    else:
        low, high = np.nanmin(values), np.nanmax(values)
        scaled = np.zeros(values.shape, dtype=float) if high <= low else (values - low) / (high - low)
        scaled[~finite] = 0.0
    rgb = np.ones((*scaled.shape, 3), dtype=float)
    brain = scaled > 0.0
    rgb[brain] = 0.75 * scaled[brain, None]
    return np.clip(rgb, 0.0, 1.0)


def slice_components(
    axis: str,
    world_coord: float,
    template: np.ndarray,
    template_affine: np.ndarray,
    roi_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return matching template and ROI-mask arrays for one anatomical plane."""
    axis_index, _, _ = ORIENTATIONS[axis]
    index, actual_coord = nearest_slice_index(
        template_affine, template.shape, axis_index, world_coord
    )
    if axis_index == 0:
        template_slice = template[index, :, :].T
        mask_slice = roi_mask[index, :, :].T
        coordinates_1 = axis_world_coordinates(template_affine, template.shape, 1)
        coordinates_2 = axis_world_coordinates(template_affine, template.shape, 2)
    elif axis_index == 1:
        template_slice = template[:, index, :].T
        mask_slice = roi_mask[:, index, :].T
        coordinates_1 = axis_world_coordinates(template_affine, template.shape, 0)
        coordinates_2 = axis_world_coordinates(template_affine, template.shape, 2)
    else:
        template_slice = template[:, :, index].T
        mask_slice = roi_mask[:, :, index].T
        coordinates_1 = axis_world_coordinates(template_affine, template.shape, 0)
        coordinates_2 = axis_world_coordinates(template_affine, template.shape, 1)
    return template_slice, mask_slice, coordinates_1, coordinates_2, actual_coord


def figure_size(axis: str, coordinates_1: np.ndarray, coordinates_2: np.ndarray) -> tuple[float, float]:
    height = 5.4
    width = 6.0
    if axis == "z":
        data_width = abs(float(coordinates_1.max() - coordinates_1.min()))
        data_height = abs(float(coordinates_2.max() - coordinates_2.min()))
        if data_height > 0:
            width = height * data_width / data_height
    return width, height


def add_roi_overlay(
    ax: plt.Axes,
    mask_slice: np.ndarray,
    coordinates_1: np.ndarray,
    coordinates_2: np.ndarray,
    color: str,
    outline_color: str,
) -> None:
    extent = (
        float(coordinates_1.min()),
        float(coordinates_1.max()),
        float(coordinates_2.min()),
        float(coordinates_2.max()),
    )
    rgba = np.zeros((*mask_slice.shape, 4), dtype=float)
    rgba[..., :3] = to_rgba(color)[:3]
    rgba[..., 3] = np.where(mask_slice, 0.86, 0.0)
    ax.imshow(rgba, extent=extent, origin="lower", interpolation="nearest", zorder=2)
    if np.any(mask_slice):
        ax.contour(
            coordinates_1,
            coordinates_2,
            mask_slice.astype(float),
            levels=[0.5],
            colors=[outline_color],
            linewidths=1.25,
            zorder=3,
        )


def plot_slice(
    axis: str,
    template: np.ndarray,
    template_affine: np.ndarray,
    roi_mask: np.ndarray,
    centroid: np.ndarray,
    roi: RoiInfo,
    output_path: Path,
    annotated: bool,
    dpi: int,
) -> float:
    """Save one clean or annotated whole-brain slice for a selected ROI."""
    axis_index, orientation_name, coordinate_label = ORIENTATIONS[axis]
    template_slice, mask_slice, coordinates_1, coordinates_2, actual_coord = slice_components(
        axis, centroid[axis_index], template, template_affine, roi_mask
    )
    if not np.any(mask_slice):
        raise RuntimeError(
            f"ROI {roi.roi_id} is absent from its {orientation_name} slice at {actual_coord:.1f} mm."
        )
    width, height = figure_size(axis, coordinates_1, coordinates_2)
    top = 0.80 if annotated else 1.0
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=top)
    extent = (
        float(coordinates_1.min()),
        float(coordinates_1.max()),
        float(coordinates_2.min()),
        float(coordinates_2.max()),
    )
    ax.imshow(
        create_template_rgb(template_slice),
        extent=extent,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        zorder=1,
    )
    add_roi_overlay(
        ax,
        mask_slice,
        coordinates_1,
        coordinates_2,
        ROI_COLORS.get(roi.roi_id, "#00E5FF"),
        ROI_OUTLINE_COLORS.get(roi.roi_id, "#003E52"),
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    if axis == "x":
        ax.invert_xaxis()
    ax.axis("off")

    if annotated:
        projected_centroid = np.delete(centroid, axis_index)
        # A two-ring marker remains visible on both the yellow and magenta regions.
        ax.scatter(
            projected_centroid[0],
            projected_centroid[1],
            s=84,
            facecolors="none",
            edgecolors="white",
            linewidths=2.2,
            zorder=4,
        )
        ax.scatter(
            projected_centroid[0],
            projected_centroid[1],
            s=102,
            facecolors="none",
            edgecolors="#202020",
            linewidths=0.85,
            zorder=5,
        )
        centroid_text = ", ".join(f"{value:.1f}" for value in centroid)
        fig.text(
            0.5,
            0.975,
            f"ROI {roi.roi_id} ({roi.abbreviation})",
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#1A1A1A",
        )
        fig.text(
            0.5,
            0.938,
            f"MNI center: ({centroid_text}) mm",
            ha="center",
            va="top",
            fontsize=9.5,
            color="#303030",
        )
        fig.text(
            0.5,
            0.901,
            f"{orientation_name.title()} slice | {coordinate_label} = {actual_coord:.1f} mm",
            ha="center",
            va="top",
            fontsize=9.5,
            color="#444444",
        )

    fig.savefig(output_path, dpi=dpi, facecolor="white", edgecolor="white")
    plt.close(fig)
    return actual_coord


def output_stem(roi: RoiInfo) -> str:
    abbreviation = "".join(character for character in roi.abbreviation if character.isalnum() or character == "_")
    return f"roi_{roi.roi_id:03d}_{abbreviation}"


def main() -> None:
    args = parse_args()
    for path in (args.label_atlas, args.atlas_labels, args.template):
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    template, template_affine = load_nifti(args.template)
    atlas, atlas_affine = load_nifti(args.label_atlas)
    atlas_on_template = resample_label_atlas_to_template(
        atlas, atlas_affine, template.shape, template_affine
    )
    roi_metadata = load_roi_info(args.atlas_labels)
    requested_roi_ids = list(dict.fromkeys(args.roi_ids))
    missing_metadata = [roi_id for roi_id in requested_roi_ids if roi_id not in roi_metadata]
    if missing_metadata:
        raise ValueError(f"ROI IDs absent from {args.atlas_labels}: {missing_metadata}")

    args.output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Writing ROI slice images to: {args.output_folder}")
    for roi_id in requested_roi_ids:
        roi = roi_metadata[roi_id]
        roi_mask = atlas_on_template == roi_id
        template_voxel_indices = np.argwhere(roi_mask)
        native_voxel_indices = np.argwhere(np.rint(atlas).astype(np.int32) == roi_id)
        if template_voxel_indices.size == 0 or native_voxel_indices.size == 0:
            raise ValueError(f"ROI {roi_id} is absent from the resampled atlas.")
        # The target coordinate comes from the native AAL3 MNI grid. The plotted
        # plane is then snapped only to the nearest display-template voxel.
        centroid = voxel_to_world(native_voxel_indices, atlas_affine).mean(axis=0)
        print(
            f"ROI {roi.roi_id} ({roi.abbreviation}; {roi.name}): "
            f"MNI center = ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}) mm"
        )
        for axis in ORIENTATIONS:
            _, orientation_name, _ = ORIENTATIONS[axis]
            clean_path = args.output_folder / f"{output_stem(roi)}_{orientation_name}_unlabeled.png"
            annotated_path = args.output_folder / f"{output_stem(roi)}_{orientation_name}_annotated.png"
            actual_coord = plot_slice(
                axis,
                template,
                template_affine,
                roi_mask,
                centroid,
                roi,
                clean_path,
                annotated=False,
                dpi=args.dpi,
            )
            plot_slice(
                axis,
                template,
                template_affine,
                roi_mask,
                centroid,
                roi,
                annotated_path,
                annotated=True,
                dpi=args.dpi,
            )
            print(
                f"  {orientation_name.title()} {ORIENTATIONS[axis][2]} = {actual_coord:.1f} mm: "
                f"{clean_path.name}, {annotated_path.name}"
            )


if __name__ == "__main__":
    main()
