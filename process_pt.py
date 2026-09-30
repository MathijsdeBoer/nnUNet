from pathlib import Path

import SimpleITK as sitk


def process_patient(source: Path, target: Path) -> None:
    """Process a single patient."""
    if not source.exists():
        msg = f"Source dir {source} does not exist"
        raise FileNotFoundError(msg)
    if not target.exists():
        target.mkdir(parents=True)

    colortable = source / "Segmentation-label_ColorTable.ctbl"
    if not colortable.exists():
        msg = f"Colortable {colortable} does not exist"
        raise FileNotFoundError(msg)

    label_mapping = {}
    with colortable.open() as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            label_id = int(parts[0])
            label_name = parts[1]
            label_mapping[label_id] = label_name

    segmentation = source / "Segmentation-label.nii"
    if not segmentation.exists():
        msg = f"Segmentation file {segmentation} does not exist"
        raise FileNotFoundError(msg)
    segmentation_image = sitk.ReadImage(str(segmentation))
    for label_id, label_name in label_mapping.items():
        binary_mask = sitk.BinaryThreshold(segmentation_image, lowerThreshold=label_id, upperThreshold=label_id)
        output_path = target / f"label_{label_name}.nii.gz"
        sitk.WriteImage(binary_mask, str(output_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process patient segmentation data.")
    parser.add_argument("source", type=Path, help="Path to the source patient directory")
    args = parser.parse_args()

    process_patient(args.source, args.source)
