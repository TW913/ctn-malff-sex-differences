import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "malff_output"
INPUT_PATTERN = "*_mALFF_nuisanceRegressed.nii*"
TEMPLATE_PATH = BASE_DIR / "mni152.nii"
ATLAS_PATH = BASE_DIR / "AAL3v1_1mm.nii"
OUT_DIR = BASE_DIR / "nii_stats"
COMBINED_CSV = OUT_DIR / "all_mALFF_nuisanceRegressed_AAL3_mean_long.csv"
STATUS_CSV = OUT_DIR / "batch_AAL3_mean_status.csv"

OVERWRITE_OUTPUT = False
BAD_ROI = -2147483648


def load_nifti(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    if path.is_dir():
        candidates = sorted(
            p for p in path.iterdir()
            if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
        )
        if not candidates:
            raise FileNotFoundError(f"No NIfTI file found inside directory: {path}")
        same_name = [p for p in candidates if p.name == path.name]
        path = same_name[0] if same_name else candidates[0]
        print("Resolved NIfTI directory to file:", path)
    if path.stat().st_size == 0:
        raise ValueError(f"NIfTI file is empty: {path}")
    return nib.load(str(path))


def find_input_images(input_dir: Path, pattern: str):
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    files = sorted(
        p for p in input_dir.glob(pattern)
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
    )
    if not files:
        raise FileNotFoundError(
            f"No mALFF NIfTI files matching {pattern} were found in: {input_dir}"
        )
    return files


def nifti_stem(path: Path):
    name = path.name
    lower_name = name.lower()
    if lower_name.endswith(".nii.gz"):
        return name[:-7]
    if lower_name.endswith(".nii"):
        return name[:-4]
    return path.stem


def safe_filename(text: str):
    keep = []
    for char in text:
        if char.isalnum() or char in "-_":
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


def extract_subject_id(stem: str):
    parts = stem.split("_", 1)
    return parts[0] if parts and parts[0] else ""


def make_output_csv(input_path: Path):
    return OUT_DIR / f"{safe_filename(nifti_stem(input_path))}_AAL3_mean.csv"


def same_grid(img, ref_img, atol=1e-3):
    return img.shape[:3] == ref_img.shape[:3] and np.allclose(
        img.affine, ref_img.affine, atol=atol
    )


def mean_3d_image(img):
    if len(img.shape) == 3:
        data = np.asarray(img.dataobj, dtype=np.float32)
    elif len(img.shape) == 4:
        acc = np.zeros(img.shape[:3], dtype=np.float64)
        counts = np.zeros(img.shape[:3], dtype=np.int32)

        for i in range(img.shape[3]):
            volume = np.asarray(img.dataobj[..., i], dtype=np.float32)
            finite = np.isfinite(volume)
            acc[finite] += volume[finite]
            counts[finite] += 1

        data = np.full(img.shape[:3], np.nan, dtype=np.float32)
        valid = counts > 0
        data[valid] = (acc[valid] / counts[valid]).astype(np.float32)
    else:
        raise ValueError(f"Expected 3D or 4D NIfTI, got shape: {img.shape}")

    return nib.Nifti1Image(data, img.affine)


def extract_roi_means(input_path: Path, template_img, atlas_data, roi_ids):
    input_img = load_nifti(input_path)
    mean_img = mean_3d_image(input_img)
    if not same_grid(mean_img, template_img):
        print("Resampling input mean image to MNI template grid with continuous interpolation.")
        mean_img = resample_to_img(
            mean_img,
            template_img,
            interpolation="continuous",
            force_resample=True,
        )

    data = mean_img.get_fdata(dtype=np.float32)
    if data.shape != atlas_data.shape:
        raise ValueError(
            f"Image and atlas grids do not match after resampling: "
            f"image={data.shape}, atlas={atlas_data.shape}"
        )

    stem = nifti_stem(input_path)
    subject_id = extract_subject_id(stem)
    rows = []
    for roi in roi_ids:
        mask = atlas_data == roi
        values = data[mask]
        values = values[np.isfinite(values)]
        mean_value = float(values.mean()) if values.size else np.nan
        rows.append(
            {
                "Subject": subject_id,
                "Image": input_path.name,
                "ROI": int(roi),
                "MeanValue": mean_value,
            }
        )

    return pd.DataFrame(rows).sort_values(["Subject", "ROI"])


def load_existing_output(csv_path: Path):
    df = pd.read_csv(csv_path)
    required_columns = {"Subject", "Image", "ROI", "MeanValue"}
    if not required_columns.issubset(df.columns):
        missing = ", ".join(sorted(required_columns.difference(df.columns)))
        raise ValueError(f"Existing CSV is missing required columns: {missing}")
    return df


def append_status(rows, image_file: Path, output_file: Path, status: str, message: str):
    rows.append(
        {
            "Image": image_file.name if image_file else "",
            "OutputFile": str(output_file) if output_file else "",
            "Status": status,
            "Message": str(message).replace("\r", " ").replace("\n", " "),
        }
    )


def write_status_csv(rows):
    pd.DataFrame(rows).to_csv(STATUS_CSV, index=False, encoding="utf-8-sig")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Input mALFF directory:", INPUT_DIR)
    print("MNI template:", TEMPLATE_PATH)
    print("AAL3 atlas:", ATLAS_PATH)
    print("Output directory:", OUT_DIR)
    print(
        "Note: this script only resamples images. Make sure the input image is already "
        "registered/normalized to MNI space before interpreting ROI values."
    )

    input_files = find_input_images(INPUT_DIR, INPUT_PATTERN)
    print(f"mALFF files found: {len(input_files)}")

    template_img = load_nifti(TEMPLATE_PATH)
    atlas_img = load_nifti(ATLAS_PATH)

    if not same_grid(atlas_img, template_img):
        print("Resampling AAL3 atlas to MNI template grid with nearest-neighbor interpolation.")
        atlas_img = resample_to_img(
            atlas_img,
            template_img,
            interpolation="nearest",
            force_resample=True,
        )

    atlas_float = atlas_img.get_fdata()
    atlas_float = np.nan_to_num(atlas_float, nan=0.0, posinf=0.0, neginf=0.0)
    atlas_data = np.rint(atlas_float).astype(np.int32)
    atlas_data[atlas_data == BAD_ROI] = 0

    roi_ids = np.unique(atlas_data)
    roi_ids = roi_ids[(roi_ids != 0) & (roi_ids != BAD_ROI)]
    print(f"ROI count: {len(roi_ids)}")

    all_results = []
    status_rows = []

    for index, input_path in enumerate(input_files, start=1):
        out_csv = make_output_csv(input_path)
        print("\n" + "=" * 60)
        print(f"File {index}/{len(input_files)}: {input_path.name}")
        print("Output CSV:", out_csv)

        try:
            if out_csv.exists() and not OVERWRITE_OUTPUT:
                try:
                    df = load_existing_output(out_csv)
                    all_results.append(df)
                    append_status(
                        status_rows,
                        input_path,
                        out_csv,
                        "SKIPPED",
                        "Valid output already exists; set OVERWRITE_OUTPUT=True to rerun",
                    )
                    write_status_csv(status_rows)
                    print("SKIP: valid output already exists.")
                    continue
                except Exception as exc:
                    print(f"Existing output is not reusable and will be regenerated: {exc}")
                    out_csv.unlink()

            df = extract_roi_means(input_path, template_img, atlas_data, roi_ids)
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            all_results.append(df)
            append_status(status_rows, input_path, out_csv, "DONE", "Done")
            write_status_csv(status_rows)
            print("Saved:", out_csv)
        except Exception as exc:
            append_status(status_rows, input_path, out_csv, "FAILED", exc)
            write_status_csv(status_rows)
            print(f"FAILED: {exc}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(COMBINED_CSV, index=False, encoding="utf-8-sig")
        print("\nSaved combined CSV:", COMBINED_CSV)

    print("Status CSV:", STATUS_CSV)


if __name__ == "__main__":
    main()
