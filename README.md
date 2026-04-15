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