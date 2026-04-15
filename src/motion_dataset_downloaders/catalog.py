from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    filename: str


@dataclass(frozen=True)
class Dataset:
    slug: str
    name: str
    modality: str
    relation: str
    access: str
    homepage: str
    assets: tuple[Asset, ...] = ()
    notes: tuple[str, ...] = ()
    manual_steps: tuple[str, ...] = ()
    tracked_files: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        slug="egodex",
        name="EgoDex",
        modality="First-person egocentric video with 3D head, upper-body, and hand pose",
        relation="Directly relevant to EgoScale",
        access="public_direct",
        homepage="https://github.com/apple/ml-egodex",
        assets=(
            Asset("train_part1", "https://ml-site.cdn-apple.com/datasets/egodex/part1.zip", "egodex_part1.zip"),
            Asset("train_part2", "https://ml-site.cdn-apple.com/datasets/egodex/part2.zip", "egodex_part2.zip"),
            Asset("train_part3", "https://ml-site.cdn-apple.com/datasets/egodex/part3.zip", "egodex_part3.zip"),
            Asset("train_part4", "https://ml-site.cdn-apple.com/datasets/egodex/part4.zip", "egodex_part4.zip"),
            Asset("train_part5", "https://ml-site.cdn-apple.com/datasets/egodex/part5.zip", "egodex_part5.zip"),
            Asset("test", "https://ml-site.cdn-apple.com/datasets/egodex/test.zip", "egodex_test.zip"),
            Asset("extra", "https://ml-site.cdn-apple.com/datasets/egodex/extra.zip", "egodex_extra.zip"),
        ),
        notes=(
            "Large download: about 1.7TB total if you fetch every archive.",
            "Good starting point: test split first, or training set part 2 for basic_pick_place.",
        ),
        tags=("egocentric", "hand", "dexterous"),
    ),
    Dataset(
        slug="ego4d",
        name="Ego4D",
        modality="First-person daily-life activity video",
        relation="Related to EgoScale-style large-scale egocentric pretraining",
        access="manual_license",
        homepage="https://ego4d-data.org/",
        manual_steps=(
            "Review and sign the Ego4D license agreement.",
            "Wait for approval and AWS credentials.",
            "Use the official CLI/downloader from Ego4D docs.",
        ),
        notes=("Publicly available after license approval.",),
        tags=("egocentric", "activity"),
    ),
    Dataset(
        slug="ego_exo4d",
        name="Ego-Exo4D",
        modality="Synchronized first-person and third-person skilled activity video",
        relation="Bridges EgoScale-style ego learning and SONIC-style motion learning",
        access="manual_license",
        homepage="https://ego-exo4d-data.org/",
        manual_steps=(
            "Sign the Ego-Exo4D license.",
            "Wait for approval.",
            "Use the official CLI downloader from the docs.",
        ),
        notes=("Publicly available after license approval.",),
        tags=("egocentric", "exo", "pose"),
    ),
    Dataset(
        slug="ego_exo4d_egopose",
        name="Ego-Exo4D EgoPose",
        modality="Ego body pose and ego hand pose benchmark code",
        relation="Relevant benchmark repo for ego body and hand pose learning",
        access="external_public",
        homepage="https://github.com/EGO4D/ego-exo4d-egopose",
        manual_steps=(
            "Clone the benchmark repo for baseline code.",
            "Obtain the underlying Ego-Exo4D data via the licensed downloader.",
        ),
        tags=("egocentric", "pose", "benchmark"),
    ),
    Dataset(
        slug="amass",
        name="AMASS",
        modality="Third-person mocap unified as SMPL motion",
        relation="Directly relevant to SONIC evaluation and motion learning",
        access="manual_license",
        homepage="https://amass.is.tue.mpg.de/",
        manual_steps=(
            "Register on the AMASS site.",
            "Accept the research license.",
            "Download the desired AMASS archives, typically the SMPL+H G releases.",
            "Place the downloaded archives under downloads/amass/ for local CLI management.",
        ),
        tracked_files=(
            "ACCAD.tar.bz2",
            "BMLhandball.tar.bz2",
            "BMLmovi.tar.bz2",
            "BMLrub.tar.bz2",
            "CMU.tar.bz2",
            "DanceDB.tar.bz2",
            "DFaust.tar.bz2",
            "EKUT.tar.bz2",
            "EyesJapanDataset.tar.bz2",
            "GRAB.tar.bz2",
            "HDM05.tar.bz2",
            "HUMAN4D.tar.bz2",
            "HumanEva.tar.bz2",
            "KIT.tar.bz2",
            "MoSh.tar.bz2",
            "PosePrior.tar.bz2",
            "SFU.tar.bz2",
            "SOMA.tar.bz2",
            "SSM.tar.bz2",
            "TCDHands.tar.bz2",
            "TotalCapture.tar.bz2",
            "Transitions.tar.bz2",
        ),
        notes=(
            "This repo tracks the manually downloaded SMPL+H G archive set under downloads/amass/.",
            "Use the local-status and extract-local commands after the archives land on disk.",
        ),
        tags=("mocap", "whole-body", "smpl"),
    ),
    Dataset(
        slug="babel",
        name="BABEL",
        modality="Language labels aligned to AMASS motion",
        relation="Relevant to SONIC-style multimodal motion understanding",
        access="manual_license",
        homepage="https://babel.is.tue.mpg.de/",
        manual_steps=(
            "Register on the BABEL site.",
            "Accept the academic license.",
            "Download labels and code.",
        ),
        tags=("mocap", "language", "whole-body"),
    ),
    Dataset(
        slug="lafan1",
        name="LaFAN1",
        modality="Third-person mocap in BVH format",
        relation="Directly used in SONIC scaling comparisons",
        access="external_public",
        homepage="https://github.com/ubisoft/ubisoft-laforge-animation-dataset",
        manual_steps=(
            "Install git-lfs.",
            "Clone the Ubisoft La Forge dataset repo.",
            "Extract lafan1/lafan1.zip if needed.",
        ),
        notes=("Public repo, but data is stored with Git LFS.",),
        tags=("mocap", "whole-body", "bvh"),
    ),
    Dataset(
        slug="interhand26m",
        name="InterHand2.6M",
        modality="Large-scale single-hand and interacting-hand 3D pose",
        relation="Useful public hand motion dataset for pretraining and evaluation",
        access="external_public",
        homepage="https://mks0601.github.io/InterHand2.6M/",
        manual_steps=(
            "Download the desired 5fps or 30fps image package from the official site.",
            "Download annotations and optional MANO fits.",
            "Follow the official structure described on the project page.",
        ),
        tags=("hand", "pose", "interacting-hands"),
    ),
    Dataset(
        slug="ho3d",
        name="HO-3D",
        modality="Hand-object 3D pose with occlusion",
        relation="Useful for hand-object interaction learning",
        access="external_public",
        homepage="https://github.com/shreyashampali/ho3d",
        manual_steps=(
            "Open the official HO-3D repo.",
            "Download the desired HO-3D release from the published external link.",
            "Use the provided visualization/evaluation scripts if needed.",
        ),
        tags=("hand", "object", "interaction"),
    ),
    Dataset(
        slug="h2o3d",
        name="H2O-3D",
        modality="Two-hand object interaction with 3D pose annotations",
        relation="Useful for bimanual hand-object learning",
        access="external_public",
        homepage="https://github.com/shreyashampali/ho3d",
        manual_steps=(
            "Open the official HO-3D repo.",
            "Use the H2O-3D download link listed in the README.",
            "Reuse the same repo scripts for visualization and evaluation.",
        ),
        tags=("hand", "object", "bimanual"),
    ),
    Dataset(
        slug="dexycb",
        name="DexYCB",
        modality="Hand grasping with object pose and robot handover relevance",
        relation="Useful for hand-object interaction and robotics",
        access="external_public",
        homepage="https://dex-ycb.github.io/",
        manual_steps=(
            "Download the monolithic archive or subject-wise archives from the official project page.",
            "Extract the dataset and fetch the toolkit from NVlabs/dex-ycb-toolkit.",
        ),
        tags=("hand", "object", "grasp"),
    ),
)


UNAVAILABLE_DATASETS: tuple[Dataset, ...] = (
    Dataset(
        slug="egoscale_pretraining_mixture",
        name="EgoScale 20k-hour Pretraining Mixture",
        modality="Large-scale egocentric human video mixture",
        relation="Core EgoScale pretraining source",
        access="unavailable",
        homepage="https://research.nvidia.com/labs/gear/egoscale/",
        notes=("Not publicly released as a downloadable dataset.",),
    ),
    Dataset(
        slug="egoscale_midtraining_data",
        name="EgoScale Aligned Human-Robot Mid-Training Data",
        modality="Aligned human and robot tabletop play data",
        relation="Core EgoScale Stage II data",
        access="unavailable",
        homepage="https://research.nvidia.com/labs/gear/egoscale/",
        notes=("Not publicly released as a downloadable dataset.",),
    ),
    Dataset(
        slug="sonic_inhouse_mocap",
        name="SONIC In-House Mocap Corpus",
        modality="700-hour whole-body mocap dataset",
        relation="Core SONIC training data",
        access="unavailable",
        homepage="https://nvlabs.github.io/SONIC/",
        notes=("Not publicly released as a downloadable dataset.",),
    ),
)


ALL_DATASETS: tuple[Dataset, ...] = DATASETS + UNAVAILABLE_DATASETS


def get_dataset(slug: str) -> Dataset:
    for dataset in ALL_DATASETS:
        if dataset.slug == slug:
            return dataset
    raise KeyError(f"Unknown dataset slug: {slug}")


def iter_by_access(access: str) -> tuple[Dataset, ...]:
    return tuple(dataset for dataset in ALL_DATASETS if dataset.access == access)