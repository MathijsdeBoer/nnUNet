import json
import random
import shutil
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from nnunetv2.dataset_conversion.Dataset002_cSDH import Scan, collect_cases_grouped, remap_label
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

SEED = 42


def split_patients(
    groups: dict[str, list[Scan]], train_target: int, seed: int = SEED
) -> tuple[set[str], set[str]]:
    """Patient-level train/test split: a patient's scans never span both splits.

    Finds an exact-count split (train_target scans in train, the rest in
    test) by treating patients as "sizes" (1 scan, 2 scans, ...) and solving
    a small subset-sum, randomized (but seeded) over which patients land
    where.
    """
    names = sorted(groups)
    sizes = {n: len(groups[n]) for n in names}
    total = sum(sizes.values())
    test_target = total - train_target

    multi = [n for n in names if sizes[n] > 1]
    single = [n for n in names if sizes[n] == 1]

    rng = random.Random(seed)
    rng.shuffle(multi)
    rng.shuffle(single)

    k_order = list(range(len(multi) + 1))
    rng.shuffle(k_order)
    for k in k_order:
        train_multi = multi[:k]
        remaining = train_target - sum(sizes[n] for n in train_multi)
        if 0 <= remaining <= len(single):
            train_names = set(train_multi) | set(single[:remaining])
            test_names = set(names) - train_names
            assert sum(sizes[n] for n in train_names) == train_target
            assert sum(sizes[n] for n in test_names) == test_target
            return train_names, test_names

    msg = f"No valid {train_target}/{test_target} split for patient group sizes {sorted(sizes.values())}"
    raise ValueError(msg)


if __name__ == "__main__":
    source = Path(r"C:\Users\MathijsdeBoer\Data\CSDH\Segmentaties")
    target = Path(r"C:\Users\MathijsdeBoer\Data\nnUNet\Dataset004_cSDH_Abstract")

    TRAIN_SCANS = 20
    TEST_SCANS = 15

    imagesTr = target / "imagesTr"
    labelsTr = target / "labelsTr"
    imagesTs = target / "imagesTs"
    labelsTs = target / "labelsTs"
    for d in (imagesTr, labelsTr, imagesTs, labelsTs):
        maybe_mkdir_p(d)

    groups = collect_cases_grouped(source)
    total_scans = sum(len(v) for v in groups.values())
    print(f"Found {len(groups)} patients, {total_scans} scans")
    if total_scans != TRAIN_SCANS + TEST_SCANS:
        print(f"  [warn] expected {TRAIN_SCANS + TEST_SCANS} scans, got {total_scans} -- splitting anyway")

    train_patients, test_patients = split_patients(groups, TRAIN_SCANS, seed=SEED)

    train_cases: list[Scan] = []
    test_cases: list[Scan] = []
    for patient, scans in groups.items():
        (train_cases if patient in train_patients else test_cases).extend(scans)

    print(f"Train: {len(train_cases)} scans from {len(train_patients)} patients")
    print(f"Test:  {len(test_cases)} scans from {len(test_patients)} patients")

    for scan in train_cases:
        shutil.copy2(scan.ct_path, imagesTr / f"{scan.case_id}_0000.nii.gz")
        remap_label(scan.label_path, labelsTr / f"{scan.case_id}.nii.gz", scan.csdh_values, scan.mma_values)

    for scan in test_cases:
        shutil.copy2(scan.ct_path, imagesTs / f"{scan.case_id}_0000.nii.gz")
        remap_label(scan.label_path, labelsTs / f"{scan.case_id}.nii.gz", scan.csdh_values, scan.mma_values)

    manifest = {
        "seed": SEED,
        "train_patients": sorted(train_patients),
        "test_patients": sorted(test_patients),
        "train_cases": sorted(s.case_id for s in train_cases),
        "test_cases": sorted(s.case_id for s in test_cases),
    }
    with (target / "split_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=4)

    labels = {
        "background": 0,
        "csdh": 1,
        "mma": 2,
    }

    generate_dataset_json(
        str(target),
        {0: "CT"},
        labels,
        len(train_cases),
        ".nii.gz",
        dataset_name="Dataset004_cSDH_Abstract",
        description=(
            "Chronic Subdural Hematoma (CSDH) dataset collected at the UMCU. "
            "Left/right instances of cSDH and MMA were merged into single classes. "
            f"Fixed {TRAIN_SCANS}/{TEST_SCANS} patient-level train/test split (seed={SEED}) "
            "for an abstract submission -- see split_manifest.json."
        ),
        reference="University Medical Center Utrecht (UMCU)",
        release="1.0",
        license="None, all rights reserved",
        converted_by="Mathijs de Boer",
    )
