<div align="center">

# POCO

## Posterior Optimization with Clipped Objective for Bridging Efficiency and Stability in Generative Policy Learning

Yuhui Chen, Haoran Li, Zhennan Jiang, Yuxing Qin, Yuxuan Wan, Weiheng Liu, and Dongbin Zhao

[[Project Page](https://cccedric.github.io/poco/)] [[Paper](https://arxiv.org/abs/2604.01860)] [[PDF](https://arxiv.org/pdf/2604.01860)]

</div>

This repository contains the official implementation of **Posterior Optimization with Clipped Objective (POCO)**. POCO is an offline-to-online reinforcement learning framework for fine-tuning generative policies over temporal action chunks. It treats policy improvement as likelihood-free posterior inference, uses reward-weighted action candidates in an implicit Expectation-Maximization procedure, and clips the policy objective to keep online updates anchored to the offline behavioral prior.

For the method, results, and real-world videos, see the [POCO project page](https://cccedric.github.io/poco/) and [paper](https://arxiv.org/abs/2604.01860).

## Installation

The experiments require a Linux machine with an NVIDIA GPU, CUDA 12, and MuJoCo EGL rendering.

```bash
git clone https://github.com/DRL-CASIA/poco_test.git
cd poco_test

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For the Robomimic experiments, also install the bundled Robomimic package and its simulator dependencies:

```bash
pip install -e ./robomimic
```

## Datasets

The OGBench Puzzle and Scene scripts use the dataset names configured in the scripts. OGBench downloads missing standard datasets through its dataset loader.

For the Robomimic experiments, download the Multi-Human, low-dimensional datasets from the [Robomimic v0.1 dataset page](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html). Arrange them under `ROBOMIMIC_DATA_DIR` as follows:

```text
$ROBOMIMIC_DATA_DIR/
├── can/mh/low_dim_v15.hdf5
└── square/mh/low_dim_v15.hdf5
```

If `ROBOMIMIC_DATA_DIR` is unset, the loader uses `~/.robomimic`. The Can experiment additionally uses a 30-trajectory subset at `data/can_mh_low_dim_v15_subset30.hdf5` because its script passes `--is_subset=True`.

## Running the Experiments

The released experiments are defined by the following four scripts. Run them from the repository root.

| Benchmark | Environment | Script |
| --- | --- | --- |
| OGBench Puzzle | `puzzle-3x3-play-singletask-task1-v0` | [`scripts/run_ogbench_puzzle1_mpo_v10.sh`](scripts/run_ogbench_puzzle1_mpo_v10.sh) |
| OGBench Scene | `scene-play-singletask-task1-v0` | [`scripts/run_ogbench_scene1_mpo_v10.sh`](scripts/run_ogbench_scene1_mpo_v10.sh) |
| Robomimic Can | `can-mh-low_dim` | [`scripts/run_robomimic_can_mpo_v10_clip0.3.sh`](scripts/run_robomimic_can_mpo_v10_clip0.3.sh) |
| Robomimic Square | `square-mh-low_dim` | [`scripts/run_robomimic_square_mpo_v10.sh`](scripts/run_robomimic_square_mpo_v10.sh) |

```bash
# OGBench Puzzle
bash scripts/run_ogbench_puzzle1_mpo_v10.sh

# OGBench Scene
bash scripts/run_ogbench_scene1_mpo_v10.sh

# Robomimic Can
bash scripts/run_robomimic_can_mpo_v10_clip0.3.sh

# Robomimic Square
bash scripts/run_robomimic_square_mpo_v10.sh
```

Each script runs seeds `42`, `43`, and `44`. You can override the GPU, temporary directory, and output root without editing the scripts:

```bash
CUDA_VISIBLE_DEVICES=0 \
TMPDIR=/tmp/poco \
SAVE_DIR="$PWD/exp" \
WANDB_MODE=offline \
bash scripts/run_ogbench_puzzle1_mpo_v10.sh
```

To log online with Weights & Biases, authenticate with `wandb login` or export `WANDB_API_KEY` in your shell. Set `WANDB_ENTITY` when logging to a team account. Never store API keys in the repository. The scripts accept `True` as their first argument to enable online behavior cloning for the Robomimic experiments:

```bash
bash scripts/run_robomimic_can_mpo_v10_clip0.3.sh True
bash scripts/run_robomimic_square_mpo_v10.sh True
```

Experiment outputs are written below `$SAVE_DIR/<project>/<run-group>/<environment>/<experiment>/`. The default scripts train three seeds and may take substantial time and GPU resources.

## Citation

If you find this work useful, please cite:

```bibtex
@article{chen2026pocp,
  title   = {Posterior Optimization with Clipped Objective for Bridging Efficiency and Stability in Generative Policy Learning},
  author  = {Chen, Yuhui and Li, Haoran and Jiang, Zhennan and Qin, Yuxing and Wan, Yuxuan and Liu, Weiheng and Zhao, Dongbin},
  journal = {arXiv preprint arXiv:2604.01860},
  year    = {2026}
}
```

## Acknowledgments

This codebase builds on [FQL](https://github.com/seohongpark/fql). The `rlpd_distributions` and `rlpd_networks` modules are derived from [RLPD](https://github.com/ikostrikov/rlpd), and the bundled Robomimic code is based on [robomimic](https://github.com/ARISE-Initiative/robomimic).

## License

This repository is released under the [MIT License](LICENSE).
