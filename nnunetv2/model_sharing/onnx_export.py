import json
from os.path import isdir, join
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import onnx
import onnxruntime
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.dataset_name_id_conversion import \
    maybe_convert_to_dataset_name
from nnunetv2.utilities.file_path_utilities import get_output_folder


def export_onnx_model(
    dataset_name_or_id: Union[int, str],
    output_dir: Path,
    configurations: Tuple[str] = (
        "2d",
        "3d_lowres",
        "3d_fullres",
        "3d_cascade_fullres",
    ),
    batch_size: int = 0,
    trainer: str = "nnUNetTrainer",
    plans_identifier: str = "nnUNetPlans",
    folds: Tuple[Union[int, str], ...] = (0, 1, 2, 3, 4),
    strict: bool = True,
    save_checkpoints: Tuple[str, ...] = ("checkpoint_final.pth",),
    output_names: tuple[str, ...] = None,
    verbose: bool = False,
) -> None:
    if not output_names:
        output_names = (f"{checkpoint[:-4]}.onnx" for checkpoint in save_checkpoints)

    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")

    use_dynamic_axes = batch_size == 0

    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    for c in configurations:
        print(f"Configuration {c}")
        trainer_output_dir = get_output_folder(
            dataset_name, trainer, plans_identifier, c
        )
        dataset_json = load_json(join(trainer_output_dir, "dataset.json"))

        # While we load in this file indirectly, we need the plans file to
        # determine the foreground intensity properties.
        plans = load_json(join(trainer_output_dir, 'plans.json'))
        foreground_intensity_properties = plans['foreground_intensity_properties_per_channel']

        if not isdir(trainer_output_dir):
            if strict:
                raise RuntimeError(
                    f"{dataset_name} is missing the trained model of configuration {c}"
                )
            else:
                print(f"Skipping configuration {c}, does not exist")
                continue

        predictor = nnUNetPredictor(
            perform_everything_on_gpu=False,
            device=torch.device("cpu"),
        )

        for checkpoint_name, output_name in zip(save_checkpoints, output_names):
            predictor.initialize_from_trained_model_folder(
                model_training_output_dir=trainer_output_dir,
                use_folds=folds,
                checkpoint_name=checkpoint_name,
                disable_compilation=True,
            )

            list_of_parameters = predictor.list_of_parameters
            network = predictor.network
            config = predictor.configuration_manager

            for fold, params in zip(folds, list_of_parameters):
                network.load_state_dict(params)

                network.eval()

                curr_output_dir = output_dir / c / f"fold_{fold}"
                if not curr_output_dir.exists():
                    curr_output_dir.mkdir(parents=True)
                else:
                    if len(list(curr_output_dir.iterdir())) > 0:
                        raise RuntimeError(
                            f"Output directory {curr_output_dir} is not empty"
                        )

                if use_dynamic_axes:
                    rand_input = torch.rand((1, 1, *config.patch_size))
                    torch_output = network(rand_input)

                    torch.onnx.export(
                        network,
                        rand_input,
                        curr_output_dir / output_name,
                        export_params=True,
                        verbose=verbose,
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes={
                            "input": {0: "batch_size"},
                            "output": {0: "batch_size"},
                        },
                    )
                else:
                    rand_input = torch.rand((batch_size, 1, *config.patch_size))
                    torch_output = network(rand_input)

                    torch.onnx.export(
                        network,
                        rand_input,
                        curr_output_dir / output_name,
                        export_params=True,
                        verbose=verbose,
                        input_names=["input"],
                        output_names=["output"],
                    )

                onnx_model = onnx.load(curr_output_dir / output_name)
                onnx.checker.check_model(onnx_model)

                ort_session = onnxruntime.InferenceSession(
                    curr_output_dir / output_name, providers=["CPUExecutionProvider"]
                )
                ort_inputs = {ort_session.get_inputs()[0].name: rand_input.numpy()}
                ort_outs = ort_session.run(None, ort_inputs)

                try:
                    np.testing.assert_allclose(
                        torch_output.detach().cpu().numpy(),
                        ort_outs[0],
                        rtol=1e-03,
                        atol=1e-05,
                        verbose=True,
                    )
                except AssertionError as e:
                    print("WARN: Differences found between torch and onnx:\n")
                    print(e)
                    print(
                        "\nExport will continue, but please verify that your pipeline matches the original."
                    )

                print(f"Exported {curr_output_dir / output_name}")

                with open(curr_output_dir / "config.json", "w") as f:
                    config_dict = {
                        "configuration": c,
                        "fold": fold,
                        "model_parameters": {
                            "batch_size": batch_size
                            if not use_dynamic_axes
                            else "dynamic",
                            "patch_size": config.patch_size,
                            "spacing": config.spacing,
                            "normalization_schemes": config.normalization_schemes,
                            # These are mostly interesting for certification
                            # uses, but they are also useful for debugging.
                            "UNet_class_name": config.UNet_class_name,
                            "UNet_base_num_features": config.UNet_base_num_features,
                            "unet_max_num_features": config.unet_max_num_features,
                            "conv_kernel_sizes": config.conv_kernel_sizes,
                            "pool_op_kernel_sizes": config.pool_op_kernel_sizes,
                            "num_pool_per_axis": config.num_pool_per_axis,
                        },
                        "dataset_parameters": {
                            "dataset_name": dataset_name,
                            "num_channels": len(dataset_json["channel_names"].keys()),
                            "channels": {
                                k: {
                                    "name": v,
                                    # For when normalization is not Z-Score
                                    "foreground_properties": foreground_intensity_properties[k],
                                }
                                for k, v in dataset_json["channel_names"].items()
                            },
                            "num_classes": len(dataset_json["labels"].keys()),
                            "class_names": {
                                v: k for k, v in dataset_json["labels"].items()
                            },
                        },
                    }

                    json.dump(
                        config_dict,
                        f,
                        indent=4,
                    )
