import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

CT_NAMES = ("ct.nii.gz", "ct.nii")
LABEL_NAMES = ("label.nii.gz", "label.nii")

DATE_FORMATS = ("%d-%m-%Y", "%d_%m_%Y", "%Y-%m-%d")


def parse_date(name: str) -> datetime | None:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(name, fmt)
        except ValueError:
            continue
    # formats without a year are ambiguous to strptime directly; pin a dummy year instead
    for fmt in ("%d-%m", "%d_%m"):
        try:
            return datetime.strptime(f"{name}-2000", f"{fmt}-%Y")
        except ValueError:
            continue
    return None


CSDH_MMA_VOXEL_THRESHOLD = 20_000


@dataclass
class Scan:
    case_id: str
    ct_path: Path
    label_path: Path
    # Explicit per-instance-value -> class mapping from a labels.csv, when one
    # ships with the scan. None means "fall back to the voxel-count heuristic".
    csdh_values: set[int] | None = None
    mma_values: set[int] | None = None


def parse_labels_csv(csv_path: Path) -> tuple[set[int], set[int]]:
    """Parse a 3D Slicer-style '*.labels.csv' into explicit csdh/mma value sets.

    These files name each segmented instance (e.g. 'cSDH right', 'MMA left'),
    which is authoritative and should be preferred over the voxel-count
    heuristic used for scans that don't ship one.
    """
    csdh_values: set[int] = set()
    mma_values: set[int] = set()
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            value = int(row["LabelValue"])
            name = row["Name"].strip().lower()
            if "sdh" in name:  # covers both "cSDH ..." and the older "SDH ..." naming
                csdh_values.add(value)
            elif "mma" in name:
                mma_values.add(value)
            else:
                msg = f"Unrecognized label name {row['Name']!r} in {csv_path}"
                raise ValueError(msg)
    return csdh_values, mma_values


def remap_label(
    label_path: Path,
    out_path: Path,
    csdh_values: set[int] | None = None,
    mma_values: set[int] | None = None,
) -> None:
    """Merge left/right instances: cSDH -> 1, MMA -> 2.

    Segmentators labelled left/right cSDH and MMA as separate instances.
    We don't distinguish sides for training, so each instance needs to
    collapse to its class. When a labels.csv names the instances explicitly,
    pass its csdh_values/mma_values and we use those directly. Otherwise we
    fall back to a voxel-count heuristic: the instance *value order* is not
    reliable (most scans block cSDH before MMA, e.g. {1,2}=cSDH, {3,4}=MMA,
    but a few interleave them instead), whereas size is: a cSDH instance is
    always >= 226k voxels and an MMA instance is always <= 8.8k voxels.
    """
    image = sitk.ReadImage(str(label_path))
    array = sitk.GetArrayFromImage(image)

    remapped = np.zeros_like(array)
    if csdh_values is not None or mma_values is not None:
        for value in csdh_values or ():
            remapped[array == value] = 1
        for value in mma_values or ():
            remapped[array == value] = 2
    else:
        for value in np.unique(array):
            if value == 0:
                continue
            instance_mask = array == value
            new_value = 1 if instance_mask.sum() >= CSDH_MMA_VOXEL_THRESHOLD else 2
            remapped[instance_mask] = new_value

    out_image = sitk.GetImageFromArray(remapped)
    out_image.CopyInformation(image)
    sitk.WriteImage(out_image, str(out_path))


def _find_literal(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def _find_one(directory: Path, names: tuple[str, ...], exclude: tuple[str, ...] = ()) -> tuple[Path | None, bool]:
    """Returns (path, used_fallback). used_fallback is True when the literal
    name wasn't found and we fell back to a lone .nii/.nii.gz file instead."""
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate, False
    # Fall back to a lone .nii/.nii.gz file directly in the directory (raw
    # exports keep the segmentation's original, non-standard filename).
    loose = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.name not in exclude and (p.name.endswith(".nii.gz") or p.name.endswith(".nii"))
    ]
    return (loose[0], True) if len(loose) == 1 else (None, False)


def _find_dicom_dirs(root: Path) -> list[Path]:
    """Directories directly containing .dcm files, searched at any depth
    under root (raw exports nest each series under its own UID-named folder,
    e.g. LumiScans/<series-uid>/*.dcm, or drop files straight into a single
    folder, e.g. scan/*.dcm)."""
    found = []
    for d in (root, *(p for p in root.rglob("*") if p.is_dir())):
        try:
            has_dcm = any(f.suffix.lower() == ".dcm" for f in d.iterdir() if f.is_file())
        except OSError:
            continue
        if has_dcm:
            found.append(d)
    return found


def _convert_dicom_series(scan_dir: Path, reference_path: Path) -> Path | None:
    """Reads whichever DICOM series in scan_dir matches reference_path's
    geometry (there may be more than one series exported, e.g. a scout plus
    the diagnostic series) and writes it out as ct.nii.gz."""
    dicom_dirs = _find_dicom_dirs(scan_dir)
    if not dicom_dirs:
        return None

    reference_size = sitk.ReadImage(str(reference_path)).GetSize()

    candidates = []
    for d in dicom_dirs:
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(str(d))
        if not files:
            continue
        reader.SetFileNames(files)
        try:
            candidates.append(reader.Execute())
        except RuntimeError:
            continue
    if not candidates:
        return None

    exact = [img for img in candidates if img.GetSize() == reference_size]
    # No exact match: assume the largest series (most slices) is the real
    # diagnostic scan rather than a scout/localizer.
    chosen = exact[0] if exact else max(candidates, key=lambda img: img.GetSize()[2])

    out_path = scan_dir / "ct.nii.gz"
    sitk.WriteImage(chosen, str(out_path))
    print(f"  [info] {scan_dir}: converted DICOM series to ct.nii.gz {chosen.GetSize()}")
    return out_path


def resolve_scan(scan_dir: Path, case_id: str) -> Scan | None:
    ct = _find_literal(scan_dir, CT_NAMES)

    label, used_fallback = _find_one(scan_dir, LABEL_NAMES, exclude=(ct.name,) if ct else ())
    if label is None:
        return None

    if ct is None:
        ct = _convert_dicom_series(scan_dir, reference_path=label)
        if ct is None:
            return None

    csdh_values = mma_values = None
    if used_fallback:
        # Only a raw, un-renamed export needs (and ships) a labels.csv;
        # patients that already have a literal label.nii(.gz) may have an
        # unrelated stray labels.csv left over elsewhere, which we should
        # ignore rather than misinterpret.
        csv_matches = list(scan_dir.glob("*.labels.csv"))
        final_dir = scan_dir / "Final"
        if final_dir.is_dir():
            csv_matches += list(final_dir.glob("*.labels.csv"))
        if len(csv_matches) == 1:
            csdh_values, mma_values = parse_labels_csv(csv_matches[0])
        elif len(csv_matches) > 1:
            print(f"  [warn] {scan_dir}: multiple labels.csv found, ignoring all of them")

    return Scan(case_id, ct, label, csdh_values, mma_values)


def find_dated_scans(patient_dir: Path) -> list[Path]:
    scan_dirs = [d for d in patient_dir.iterdir() if d.is_dir() and resolve_scan(d, "_probe") is not None]

    parsed = [(parse_date(d.name), d) for d in scan_dirs]
    if all(date is not None for date, _ in parsed):
        parsed.sort(key=lambda item: item[0])
    else:
        print(f"  [warn] could not parse dates for {patient_dir.name}, falling back to name sort")
        parsed.sort(key=lambda item: item[1].name)
    return [d for _, d in parsed]


def collect_cases_grouped(source: Path) -> dict[str, list[Scan]]:
    """Returns {patient_name: [Scan, ...]}.

    Keeping cases grouped by patient lets callers split train/test at the
    patient level, so a patient with repeat scans never leaks across splits.
    """
    groups: dict[str, list[Scan]] = {}
    for patient_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        scan = resolve_scan(patient_dir, patient_dir.name)
        if scan is not None:
            groups[patient_dir.name] = [scan]
            continue

        dated_scans = find_dated_scans(patient_dir)
        if dated_scans:
            entries = []
            for idx, scan_dir in enumerate(dated_scans):
                entry = resolve_scan(scan_dir, f"{patient_dir.name}_{idx}")
                if entry is not None:
                    entries.append(entry)
            if entries:
                groups[patient_dir.name] = entries
                continue

        print(f"  [skip] {patient_dir.name}: no ct/label found")
    return groups


def collect_cases(source: Path) -> list[Scan]:
    """Returns a flat list of Scan objects."""
    cases = []
    for case_list in collect_cases_grouped(source).values():
        cases.extend(case_list)
    return cases


if __name__ == "__main__":
    source = Path(r"C:\Users\MathijsdeBoer\Data\CSDH\Segmentaties")
    target = Path(r"C:\Users\MathijsdeBoer\Data\nnUNet\Dataset002_cSDH")

    imagesTr = target / "imagesTr"
    labelsTr = target / "labelsTr"
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    cases = collect_cases(source)
    print(f"Found {len(cases)} cases")

    for scan in cases:
        shutil.copy2(scan.ct_path, imagesTr / f"{scan.case_id}_0000.nii.gz")
        remap_label(scan.label_path, labelsTr / f"{scan.case_id}.nii.gz", scan.csdh_values, scan.mma_values)

    labels = {
        "background": 0,
        "csdh": 1,
        "mma": 2,
    }

    generate_dataset_json(
        str(target),
        {0: "CT"},
        labels,
        len(cases),
        ".nii.gz",
        dataset_name="Dataset002_cSDH",
        description=(
            "Chronic Subdural Hematoma (CSDH) dataset collected at the UMCU. "
            "Left/right instances of cSDH and MMA were merged into single classes."
        ),
        reference="University Medical Center Utrecht (UMCU)",
        release="1.0",
        license="None, all rights reserved",
        converted_by="Mathijs de Boer",
    )
