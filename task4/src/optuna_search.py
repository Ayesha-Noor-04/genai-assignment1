import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import optuna


def objective(trial, args):
    """
    Run one short training trial.

    We use the final generator loss printed by train.py
    as the Optuna objective. Lower is better.
    """

    lr_g = trial.suggest_float(
        "lr_g",
        5e-5,
        5e-4,
        log=True,
    )

    lr_d = trial.suggest_float(
        "lr_d",
        5e-5,
        5e-4,
        log=True,
    )

    lambda_l1 = trial.suggest_float(
        "lambda_l1",
        50.0,
        150.0,
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [4, 8],
    )

    trial_dir = (
        Path(args.output_dir)
        / f"trial_{trial.number:03d}"
    )

    trial_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        "-m",
        "task4.src.train",

        "--data-root",
        args.data_root,

        "--output-dir",
        str(trial_dir),

        "--epochs",
        str(args.trial_epochs),

        "--batch-size",
        str(batch_size),

        "--image-size",
        str(args.image_size),

        "--lr-g",
        str(lr_g),

        "--lr-d",
        str(lr_d),

        "--lambda-l1",
        str(lambda_l1),

        "--num-workers",
        str(args.num_workers),

        "--seed",
        str(args.seed),
    ]

    print()
    print("=" * 70)
    print(f"OPTUNA TRIAL {trial.number}")
    print("=" * 70)
    print(f"lr_g       = {lr_g}")
    print(f"lr_d       = {lr_d}")
    print(f"lambda_l1  = {lambda_l1}")
    print(f"batch_size = {batch_size}")
    print(f"epochs     = {args.trial_epochs}")
    print()

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)

        raise RuntimeError(
            f"Training failed for trial {trial.number}"
        )

    # Find the last checkpoint produced by this trial.
    checkpoints = sorted(
        (
            trial_dir / "checkpoints"
        ).glob("epoch_*.pt")
    )

    if not checkpoints:
        raise RuntimeError(
            f"No checkpoint produced for trial {trial.number}"
        )

    # Parse the final training line:
    #
    # Epoch X/X | G: ... | D: ...
    #
    # We use G as the optimization objective.
    final_g = None
    final_d = None

    for line in result.stdout.splitlines():

        if "Epoch" in line and "| G:" in line:

            try:
                parts = line.split("|")

                g_part = parts[1].strip()
                d_part = parts[2].strip()

                final_g = float(
                    g_part.replace("G:", "").strip()
                )

                final_d = float(
                    d_part.replace("D:", "").strip()
                )

            except (
                ValueError,
                IndexError,
            ):
                continue

    if final_g is None:
        raise RuntimeError(
            "Could not parse final generator loss."
        )

    trial.set_user_attr(
        "final_d_loss",
        final_d,
    )

    trial.set_user_attr(
        "checkpoint",
        str(checkpoints[-1]),
    )

    print(
        f"Trial {trial.number} "
        f"final G loss: {final_g:.6f}"
    )

    print(
        f"Trial {trial.number} "
        f"final D loss: {final_d:.6f}"
    )

    return final_g


def main(args):

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    study_path = (
        output_dir
        / "optuna_study.db"
    )

    study = optuna.create_study(
        study_name="fs2k_photo_to_sketch",
        storage=f"sqlite:///{study_path}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed,
        ),
    )

    print()
    print("=" * 70)
    print("FS2K OPTUNA HYPERPARAMETER SEARCH")
    print("=" * 70)
    print(
        f"Trials: {args.trials}"
    )
    print(
        f"Epochs per trial: "
        f"{args.trial_epochs}"
    )
    print()

    study.optimize(
        lambda trial: objective(
            trial,
            args,
        ),
        n_trials=args.trials,
    )

    print()
    print("=" * 70)
    print("OPTUNA RESULTS")
    print("=" * 70)

    print(
        f"Completed trials: "
        f"{len(study.trials)}"
    )

    print(
        f"Best trial: "
        f"{study.best_trial.number}"
    )

    print(
        f"Best objective: "
        f"{study.best_value:.6f}"
    )

    print()
    print("Best parameters:")

    for key, value in (
        study.best_params.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print()

    best_checkpoint = study.best_trial.user_attrs.get(
        "checkpoint"
    )

    if best_checkpoint:
        print(
            "Best trial checkpoint:"
        )
        print(best_checkpoint)

    # Save a portable JSON summary.
    results = {
        "study_name": study.study_name,
        "direction": "minimize",
        "number_of_trials": len(
            study.trials
        ),
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_attributes": (
            study.best_trial.user_attrs
        ),
    }

    results_path = (
        output_dir
        / "best_params.json"
    )

    with open(
        results_path,
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print(
        "Results saved to:",
        results_path,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default="task4/data/FS2K",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="task4/outputs/optuna",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--trial-epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    main(args)
