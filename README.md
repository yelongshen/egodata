# motion-dataset-downloaders

Utility repo for organizing and downloading the public datasets discussed in the EgoScale and SONIC ecosystem.

This repo does three things:

1. Automates downloads for datasets that expose stable direct file URLs.
2. Tracks datasets that require manual registration, license approval, Git LFS, Google Drive, or external portals.
3. Manages locally downloaded manual archives once they are placed under `downloads/<slug>/`.

## Covered datasets

| Slug | Dataset | Modality | Relation | Access |
| --- | --- | --- | --- | --- |
| `egodex` | EgoDex | First-person egocentric video + hand/body motion | Directly relevant to EgoScale | Public direct download |
| `ego4d` | Ego4D | First-person activity video | Related to EgoScale-style pretraining | Manual license approval |
| `ego_exo4d` | Ego-Exo4D | First-person + third-person synchronized video | Related to EgoScale and SONIC | Manual license approval |
| `ego_exo4d_egopose` | Ego-Exo4D EgoPose | Ego body pose + ego hand pose benchmark assets | Related to ego/exo motion learning | Public benchmark repo + manual dataset access |
| `amass` | AMASS | Third-person mocap / SMPL motion | Directly relevant to SONIC | Manual registration |
| `babel` | BABEL | Language labels aligned to AMASS motion | Related to SONIC text/motion conditioning | Manual registration |
| `lafan1` | LaFAN1 | Third-person mocap / BVH motion | Directly used as SONIC-scale baseline | Public repo with Git LFS |
| `interhand26m` | InterHand2.6M | Large-scale hand pose data | Hand motion pretraining / evaluation | Public mixed hosting |
| `ho3d` | HO-3D | Hand-object interaction | Hand motion / hand-object learning | Public external links |
| `h2o3d` | H2O-3D | Two-hand object interaction | Hand motion / hand-object learning | Public external links |
| `dexycb` | DexYCB | Hand grasping with object pose | Hand-object interaction / robotics | Public external links |

## Not included as downloadable datasets

The following are not publicly downloadable as datasets at the time of writing:

- EgoScale 20,854-hour pretraining mixture
- EgoScale aligned human-robot mid-training data
- SONIC 700-hour in-house mocap corpus
- SONIC 300-trajectory VR teleoperation manipulation set

The CLI reports them as unavailable rather than pretending they can be downloaded.

## Why ego data can accelerate teleoperation

Short answer: yes, the current public evidence says that large-scale egocentric human data can materially improve downstream dexterous robot learning, including teleoperation-adjacent manipulation settings.

### Public paper evidence

| Paper | Core setup | Public result relevant to teleoperation | What this repo can support |
| --- | --- | --- | --- |
| [EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data](https://arxiv.org/abs/2602.16710) | Pretrain a VLA policy on 20,854 hours of egocentric human video, then add a small aligned human-robot mid-training stage before downstream robot finetuning | Reports a 54% average success-rate improvement over a no-pretraining baseline on five dexterous real-robot tasks with a 22-DoF hand. Also reports a near log-linear scaling law between human-data scale and validation loss, with downstream robot performance improving consistently as pretraining data grows. | Use `egodex`, `ego4d`, and `ego_exo4d` as public ego-data proxies for pretraining-style experiments. Exact EgoScale numbers are not reproducible here because the 20k-hour mixture and aligned human-robot mid-training set are not public. |

### Practical interpretation

- Ego data is most useful as a reusable motor prior, not as a drop-in replacement for robot data.
- The strongest public result is not "ego only", but "large-scale ego pretraining + a small amount of aligned human-robot data".
- For teleoperation workflows, this means the likely gain is reduced robot-data demand, faster task adaptation, and better initialization for dexterous control policies.
- This repo can help you assemble the public side of that pipeline, but it cannot reproduce the closed aligned human-robot mid-training data used in EgoScale.

### Experiment results you can cite

- EgoScale reports a 54% average success-rate gain over no pretraining on five dexterous manipulation tasks.
- EgoScale reports that larger ego pretraining sets produce monotonic validation improvements and a near-perfect log-linear scaling fit with $R^2 = 0.9983$.
- EgoScale reports one-shot transfer behavior when aligned human-robot mid-training is added, using only one robot demonstration per task during post-training together with aligned human demonstrations.

### What is and is not reproducible in this repo

Reproducible with public or manually accessible data in this repo:

- Egocentric pretraining-style data collection with `egodex`, `ego4d`, and `ego_exo4d`
- Motion prior experiments with `amass`, `babel`, and `lafan1`
- Hand and hand-object evaluation with `interhand26m`, `ho3d`, `h2o3d`, and `dexycb`

Not reproducible exactly from this repo:

- EgoScale 20,854-hour internal pretraining mixture
- EgoScale aligned human-robot mid-training data
- SONIC 700-hour in-house mocap corpus
- SONIC 300-trajectory VR teleoperation manipulation set

So the defensible claim is: public ego data can support the same training direction and likely improve teleoperation-oriented manipulation learning, but the headline closed-data results from EgoScale and SONIC cannot be exactly matched with this repository alone.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m motion_dataset_downloaders.cli list
python -m motion_dataset_downloaders.cli info egodex
python -m motion_dataset_downloaders.cli local-status amass --root downloads
python -m motion_dataset_downloaders.cli extract-local amass --root downloads
python -m motion_dataset_downloaders.cli download-public --root downloads
```

Or use the helper shell script:

```bash
./scripts/download_public.sh
```

## CLI overview

```bash
python -m motion_dataset_downloaders.cli list
python -m motion_dataset_downloaders.cli plan
python -m motion_dataset_downloaders.cli info <slug>
python -m motion_dataset_downloaders.cli local-status <slug> --root downloads
python -m motion_dataset_downloaders.cli download <slug> --root downloads
python -m motion_dataset_downloaders.cli extract-local <slug> --root downloads
python -m motion_dataset_downloaders.cli download-public --root downloads
```

## Managing local manual datasets

For datasets that require manual download, place the archives under `downloads/<slug>/`.

AMASS is tracked in that form:

- Expected location: `downloads/amass/`
- Tracked archive set: 22 `SMPL+H G` archives
- Batch extraction target: `downloads/amass/extracted/`

Example workflow:

```bash
python -m motion_dataset_downloaders.cli local-status amass --root downloads
python -m motion_dataset_downloaders.cli extract-local amass --root downloads
```

To extract only a subset of AMASS archives:

```bash
python -m motion_dataset_downloaders.cli extract-local amass --root downloads --archive CMU --archive KIT
```

## Access model

### `public_direct`

The repo can download these assets directly with Python.

Currently supported for automation:

- EgoDex

### `manual_license`

These datasets require account creation, license acceptance, or approval before download.

- Ego4D
- Ego-Exo4D
- AMASS
- BABEL

### `external_public`

These datasets are public but hosted through portals not worth hardcoding as a reliable downloader without site-specific clients.

- InterHand2.6M
- HO-3D
- H2O-3D
- DexYCB
- LaFAN1

The CLI gives you the canonical URLs and notes for each of them.

## Notes

- This repo does not bypass license gates or authentication flows.
- Very large datasets are intentionally not downloaded automatically unless their hosting is simple and stable.
- Output files are written under `downloads/` and ignored by git.