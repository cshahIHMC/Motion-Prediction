"""
DataLoader for CMU subset of AMASS dataset.

Each .npz file contains:
  poses          : (T, 156)  -- 52 joints x 3 axis-angle values (SMPL-H)
  trans          : (T, 3)    -- root XYZ translation (world frame)
  betas          : (16,)     -- body shape coefficients
  dmpls          : (T, 8)    -- soft-tissue deformations
  gender         : str
  mocap_framerate: float     -- 120 Hz for CMU
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Joint index reference (SMPL-H, 52 joints)
# ---------------------------------------------------------------------------
SMPL_H_JOINT_NAMES = [
    "pelvis",        # 0
    "left_hip",      # 1  ← left thigh
    "right_hip",     # 2  ← right thigh
    "spine1",        # 3
    "left_knee",     # 4  ← left shank
    "right_knee",    # 5  ← right shank
    "spine2",        # 6
    "left_ankle",    # 7  ← left foot
    "right_ankle",   # 8  ← right foot
    "spine3",        # 9
    "left_foot",     # 10
    "right_foot",    # 11
    "neck",          # 12
    "left_collar",   # 13
    "right_collar",  # 14
    "head",          # 15
    "left_shoulder", # 16
    "right_shoulder",# 17
    "left_elbow",    # 18
    "right_elbow",   # 19
    "left_wrist",    # 20
    "right_wrist",   # 21
    # Left hand (22-36)
    "left_index1","left_index2","left_index3",
    "left_middle1","left_middle2","left_middle3",
    "left_pinky1","left_pinky2","left_pinky3",
    "left_ring1","left_ring2","left_ring3",
    "left_thumb1","left_thumb2","left_thumb3",
    # Right hand (37-51)
    "right_index1","right_index2","right_index3",
    "right_middle1","right_middle2","right_middle3",
    "right_pinky1","right_pinky2","right_pinky3",
    "right_ring1","right_ring2","right_ring3",
    "right_thumb1","right_thumb2","right_thumb3",
]

# Lower body joints — all local rotations (pelvis excluded: global root).
#   spine1 + thigh (hip) + shank (knee) + foot (ankle)  →  7 joints × 3 = 21 values/frame
LOWER_BODY_JOINT_INDICES = [
    3,   # spine1       -- lumbar base (local, parent = pelvis)
    1,   # left_hip     -- left thigh
    2,   # right_hip    -- right thigh
    4,   # left_knee    -- left shank
    5,   # right_knee   -- right shank
    7,   # left_ankle   -- left foot
    8,   # right_ankle  -- right foot
]
LOWER_BODY_JOINT_NAMES = [SMPL_H_JOINT_NAMES[i] for i in LOWER_BODY_JOINT_INDICES]


# ---------------------------------------------------------------------------
# Motion-type → subject folder mapping
# A subject can belong to multiple categories.
# ---------------------------------------------------------------------------
# fmt: off
_MOTION_CATALOG: Dict[str, List[str]] = {
    "walk": [
        "01", "02", "06", "07", "08", "09",
        "28", "49", "54", "55", "56", "62", "70",
        "75", "76", "77", "78", "79", "80",
        "83", "84", "88",
        "102", "103", "105", "118", "122", "123", "137",
    ],
    "run": [
        "01", "02", "06", "07", "08", "09",
        "54", "55", "56", "62", "70",
        "75", "76", "77", "78", "79", "80",
        "83", "84", "88",
        "102", "103", "105", "118", "122", "123", "137",
    ],
    "jog": [
        "01", "02", "06",
        "54", "55", "56", "62", "70",
        "83", "84", "102", "103", "105", "118", "122", "123", "137",
    ],
    "skip": ["06"],
    "crawl": ["01"],
    "varied_terrain": ["49"],
    "obstacle_course": ["05", "13"],
    "sit_to_stand": ["40"],
}
# fmt: on

VALID_MOTION_TYPES = list(_MOTION_CATALOG.keys())

# ---------------------------------------------------------------------------
# Clip-level catalog — subjects whose ENTIRE folder is one activity type.
# Only subjects that appear in exactly ONE entry of _MOTION_CATALOG are
# included.  Each entry is (subject_folder, clip_filename).
#
# walk          → subject 28  (17 clips, all walk-only)
# obstacle_course → subjects 05 (20 clips) and 13 (42 clips), both exclusive
#
# run / jog / crawl / varied_terrain / skip have NO exclusive subjects in the
# CMU dataset — those subjects all perform multiple activity types and the
# individual clip files carry no activity label.
# ---------------------------------------------------------------------------
# fmt: off
_CLIP_CATALOG: dict = {
    "walk": [
        ("28", "28_01_poses.npz"), ("28", "28_02_poses.npz"), ("28", "28_03_poses.npz"),
        ("28", "28_04_poses.npz"), ("28", "28_05_poses.npz"), ("28", "28_06_poses.npz"),
        ("28", "28_07_poses.npz"), ("28", "28_09_poses.npz"), ("28", "28_10_poses.npz"),
        ("28", "28_11_poses.npz"), ("28", "28_12_poses.npz"), ("28", "28_13_poses.npz"),
        ("28", "28_14_poses.npz"), ("28", "28_15_poses.npz"), ("28", "28_16_poses.npz"),
        ("28", "28_17_poses.npz"), ("28", "28_19_poses.npz"),
    ],
    "obstacle_course": [
        ("05", "05_01_poses.npz"), ("05", "05_02_poses.npz"), ("05", "05_03_poses.npz"),
        ("05", "05_04_poses.npz"), ("05", "05_05_poses.npz"), ("05", "05_06_poses.npz"),
        ("05", "05_07_poses.npz"), ("05", "05_08_poses.npz"), ("05", "05_09_poses.npz"),
        ("05", "05_10_poses.npz"), ("05", "05_11_poses.npz"), ("05", "05_12_poses.npz"),
        ("05", "05_13_poses.npz"), ("05", "05_14_poses.npz"), ("05", "05_15_poses.npz"),
        ("05", "05_16_poses.npz"), ("05", "05_17_poses.npz"), ("05", "05_18_poses.npz"),
        ("05", "05_19_poses.npz"), ("05", "05_20_poses.npz"),
        ("13", "13_01_poses.npz"), ("13", "13_02_poses.npz"), ("13", "13_03_poses.npz"),
        ("13", "13_04_poses.npz"), ("13", "13_05_poses.npz"), ("13", "13_06_poses.npz"),
        ("13", "13_07_poses.npz"), ("13", "13_08_poses.npz"), ("13", "13_09_poses.npz"),
        ("13", "13_10_poses.npz"), ("13", "13_11_poses.npz"), ("13", "13_12_poses.npz"),
        ("13", "13_13_poses.npz"), ("13", "13_14_poses.npz"), ("13", "13_15_poses.npz"),
        ("13", "13_16_poses.npz"), ("13", "13_17_poses.npz"), ("13", "13_18_poses.npz"),
        ("13", "13_19_poses.npz"), ("13", "13_20_poses.npz"), ("13", "13_21_poses.npz"),
        ("13", "13_22_poses.npz"), ("13", "13_23_poses.npz"), ("13", "13_24_poses.npz"),
        ("13", "13_25_poses.npz"), ("13", "13_26_poses.npz"), ("13", "13_27_poses.npz"),
        ("13", "13_28_poses.npz"), ("13", "13_29_poses.npz"), ("13", "13_30_poses.npz"),
        ("13", "13_31_poses.npz"), ("13", "13_32_poses.npz"), ("13", "13_33_poses.npz"),
        ("13", "13_34_poses.npz"), ("13", "13_35_poses.npz"), ("13", "13_36_poses.npz"),
        ("13", "13_37_poses.npz"), ("13", "13_38_poses.npz"), ("13", "13_39_poses.npz"),
        ("13", "13_40_poses.npz"), ("13", "13_41_poses.npz"), ("13", "13_42_poses.npz"),
    ],
    # Subject 55: walk + run + jog mixed — 28 clips, 140,816 frames, 135,776 windows, ~19.6 min.
    # Activity labels are NOT known at clip level; included for full-subject manifold analysis.
    "walk_run_jog_s55": [
        ("55", "55_01_poses.npz"),  # 1,806 frames   15.1s
        ("55", "55_02_poses.npz"),  # 2,180 frames   18.2s
        ("55", "55_03_poses.npz"),  # 4,530 frames   37.8s
        ("55", "55_04_poses.npz"),  #   530 frames    4.4s
        ("55", "55_05_poses.npz"),  # 2,812 frames   23.4s
        ("55", "55_06_poses.npz"),  # 9,275 frames   77.3s
        ("55", "55_07_poses.npz"),  # 4,568 frames   38.1s
        ("55", "55_08_poses.npz"),  # 5,340 frames   44.5s
        ("55", "55_09_poses.npz"),  # 4,220 frames   35.2s
        ("55", "55_10_poses.npz"),  # 1,362 frames   11.3s
        ("55", "55_11_poses.npz"),  # 4,365 frames   36.4s
        ("55", "55_12_poses.npz"),  # 2,075 frames   17.3s
        ("55", "55_13_poses.npz"),  # 9,738 frames   81.2s
        ("55", "55_14_poses.npz"),  # 5,202 frames   43.4s
        ("55", "55_15_poses.npz"),  # 9,693 frames   80.8s
        ("55", "55_16_poses.npz"),  # 6,082 frames   50.7s
        ("55", "55_17_poses.npz"),  # 6,176 frames   51.5s
        ("55", "55_18_poses.npz"),  # 5,387 frames   44.9s
        ("55", "55_19_poses.npz"),  # 4,741 frames   39.5s
        ("55", "55_20_poses.npz"),  # 4,598 frames   38.3s
        ("55", "55_21_poses.npz"),  # 3,499 frames   29.2s
        ("55", "55_22_poses.npz"),  # 6,541 frames   54.5s
        ("55", "55_23_poses.npz"),  # 5,971 frames   49.8s
        ("55", "55_24_poses.npz"),  # 9,522 frames   79.3s
        ("55", "55_25_poses.npz"),  # 3,188 frames   26.6s
        ("55", "55_26_poses.npz"),  # 4,103 frames   34.2s
        ("55", "55_27_poses.npz"),  # 5,003 frames   41.7s
        ("55", "55_28_poses.npz"),  # 8,309 frames   69.2s
    ],
}
# fmt: on


def load_clean_clips(root_path: str, motion_type: str,
                     lower_body_only: bool = True,
                     euler_sequence: str = "ZYX",
                     n_windows: int = 500,
                     history_horizon: int = 121) -> List[dict]:
    """
    Load clips from _CLIP_CATALOG for motion_type.  Only clips long enough
    to yield at least n_windows consecutive sliding windows are returned.

    Returns a list of dicts:  { 'subject', 'clip_name', 'poses' (T, D) tensor }
    Returns [] if motion_type has no entry in _CLIP_CATALOG.
    """
    if motion_type not in _CLIP_CATALOG:
        return []

    min_frames = history_horizon + n_windows
    result = []

    for subj, fname in _CLIP_CATALOG[motion_type]:
        path = os.path.join(root_path, subj, fname)
        if not os.path.exists(path):
            continue
        raw      = np.load(path, allow_pickle=True)
        poses_raw = raw["poses"].astype(np.float32)   # (T, 156)
        fps      = float(raw["mocap_framerate"])

        if lower_body_only:
            body_pose = poses_raw[:, 3:3 + 52 * 3]
            T = body_pose.shape[0]
            angles_list = []
            for ji in LOWER_BODY_JOINT_INDICES:
                aa    = body_pose[:, ji * 3: ji * 3 + 3]
                euler = Rotation.from_rotvec(aa).as_euler(euler_sequence, degrees=True)
                angles_list.append(euler)
            poses = np.stack(angles_list, axis=1).reshape(T, -1).astype(np.float32)
        else:
            poses = poses_raw

        if min_frames > 0 and poses.shape[0] < min_frames:
            continue

        result.append({
            "subject":   subj,
            "clip_name": fname,
            "poses":     torch.from_numpy(poses),
            "fps":       fps,
        })

    return result


def _subjects_for_types(motion_types: List[str]) -> List[str]:
    """Return deduplicated subject folder names for the requested motion types."""
    subjects = set()
    for mt in motion_types:
        if mt not in _MOTION_CATALOG:
            raise ValueError(
                f"Unknown motion type '{mt}'. Valid options: {VALID_MOTION_TYPES}"
            )
        subjects.update(_MOTION_CATALOG[mt])
    return sorted(subjects)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CMUAMASSDataset(Dataset):
    """
    PyTorch Dataset for the CMU subset of AMASS.

    Parameters
    ----------
    root_path : str
        Path to the inner CMU folder that contains numbered subject sub-folders,
        e.g. ".../Data/AMASS Dataset/CMU/CMU".
    motion_types : list[str] | None
        Subset of motion categories to include. Pass None to load every
        available clip.  Valid values: walk, run, jog, skip, crawl,
        varied_terrain, obstacle_course, sit_to_stand.
    lower_body_only : bool
        When True each sample contains only the 7 local lower-body joints
        (spine1, L/R hip, L/R knee, L/R ankle — pelvis and trans excluded) →
        pose shape (T, 21).
        When False the full 52-joint body is returned → pose shape (T, 156).
    euler_sequence : str
        Euler angle convention passed to scipy Rotation.as_euler(), e.g. 'ZYX',
        'XYZ'. Angles are returned in degrees. Default 'ZYX'.
    """

    def __init__(
        self,
        root_path: str,
        motion_types: Optional[List[str]] = None,
        lower_body_only: bool = False,
        euler_sequence: str = "ZYX",
        subjects: Optional[List[str]] = None,
    ):
        """
        subjects : explicit list of subject folder names to load, e.g. ['15','91','127'].
                   When provided, motion_types is ignored.  Useful for loading
                   uncatalogued subjects (dance, sports, acrobatics, etc.) directly.
        """
        super().__init__()

        self.root_path = root_path
        self.lower_body_only = lower_body_only
        self.euler_sequence = euler_sequence
        self.subjects_override = subjects
        self.motion_types = motion_types if motion_types is not None else VALID_MOTION_TYPES

        # Build list of (subject, clip_path) tuples
        self.clips: List[Dict] = []
        self._build_clip_list()

        if len(self.clips) == 0:
            raise RuntimeError(
                f"No clips found in '{root_path}' for "
                f"subjects={subjects} / motion_types={self.motion_types}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_clip_list(self):
        if self.subjects_override is not None:
            target_subjects = self.subjects_override
        else:
            target_subjects = _subjects_for_types(self.motion_types)
        available = set(os.listdir(self.root_path))

        for subj in target_subjects:
            if subj not in available:
                continue
            subj_dir = os.path.join(self.root_path, subj)
            if not os.path.isdir(subj_dir):
                continue
            for fname in sorted(os.listdir(subj_dir)):
                if not fname.endswith(".npz"):
                    continue
                self.clips.append({
                    "subject": subj,
                    "clip_name": fname,
                    "path": os.path.join(subj_dir, fname),
                })

    @staticmethod
    def _load_npz(path: str) -> Dict:
        data = np.load(path, allow_pickle=True)
        return {
            "poses":          data["poses"].astype(np.float32),        # (T, 156)
            "trans":          data["trans"].astype(np.float32),         # (T, 3)
            "betas":          data["betas"].astype(np.float32),         # (16,)
            "mocap_framerate": float(data["mocap_framerate"]),
            "gender":         str(data["gender"]),
        }

    @staticmethod
    def _axis_angle_to_euler(poses: np.ndarray, sequence: str) -> np.ndarray:
        """
        Convert axis-angle pose array to Euler angles (degrees).

        poses    : (T, N*3)  axis-angle, N joints
        sequence : Euler convention, e.g. 'ZYX'
        returns  : (T, N*3)  Euler angles in degrees, same layout
        """
        T, D = poses.shape
        n_joints = D // 3
        aa = poses.reshape(T * n_joints, 3)         # (T*N, 3)
        # scipy interprets a (n,3) array as n rotvecs (axis * angle)
        euler = Rotation.from_rotvec(aa).as_euler(sequence, degrees=True)
        return euler.reshape(T, n_joints * 3).astype(np.float32)

    def _extract_lower_body(self, poses: np.ndarray) -> np.ndarray:
        """
        poses : (T, 156)  →  (T, 21)
        Selects the 7 local lower-body joints (3 axis-angle values each):
          spine1, L/R hip, L/R knee, L/R ankle  (pelvis excluded — global root).
        """
        idx = np.array([j * 3 + offset
                        for j in LOWER_BODY_JOINT_INDICES
                        for offset in range(3)])
        return poses[:, idx]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta   = self.clips[idx]
        raw    = self._load_npz(meta["path"])

        poses  = raw["poses"]                             # (T, 156)
        if self.lower_body_only:
            poses = self._extract_lower_body(poses)       # (T, 21)

        poses = self._axis_angle_to_euler(poses, self.euler_sequence)  # (T, D) degrees

        out = {
            "poses":           torch.from_numpy(poses),  # (T, 156) or (T, 21), Euler degrees
            "betas":           torch.from_numpy(raw["betas"]),  # (16,)
            "mocap_framerate": raw["mocap_framerate"],
            "gender":          raw["gender"],
            "subject":         meta["subject"],
            "clip_name":       meta["clip_name"],
        }
        # trans is global — only include when the caller wants full-body data
        if not self.lower_body_only:
            out["trans"] = torch.from_numpy(raw["trans"])  # (T, 3)
        return out

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Print a short summary of the loaded dataset."""
        src = f"subjects={self.subjects_override}" if self.subjects_override else f"motion_types={self.motion_types}"
        lines = [
            f"CMUAMASSDataset",
            f"  root         : {self.root_path}",
            f"  source       : {src}",
            f"  lower body   : {self.lower_body_only}",
            f"  pose dim     : {21 if self.lower_body_only else 156}  (Euler {self.euler_sequence}, degrees)",
            f"  clips        : {len(self.clips)}",
        ]
        if self.lower_body_only:
            lines.append(f"  joints       : {LOWER_BODY_JOINT_NAMES}")
        return "\n".join(lines)

    def subject_clip_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for c in self.clips:
            counts[c["subject"]] = counts.get(c["subject"], 0) + 1
        return counts

    def get_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load every clip and concatenate along the time axis.

        Returns
        -------
        poses : (N_total_frames, pose_dim)
        trans : (N_total_frames, 3)
        """
        all_poses, all_trans = [], []
        for i in range(len(self)):
            sample = self[i]
            all_poses.append(sample["poses"].numpy())
            all_trans.append(sample["trans"].numpy())
        return np.concatenate(all_poses, axis=0), np.concatenate(all_trans, axis=0)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ROOT = os.path.join(
        os.path.dirname(__file__),
        "..", "Data", "AMASS Dataset", "CMU", "CMU",
    )

    print("=== Full body, all locomotion types ===")
    ds_full = CMUAMASSDataset(
        root_path=ROOT,
        motion_types=["walk", "run", "jog", "skip", "crawl",
                      "varied_terrain", "obstacle_course", "sit_to_stand"],
        lower_body_only=False,
    )
    print(ds_full.summary())
    sample = ds_full[0]
    print(f"\n  clip  : {sample['subject']}/{sample['clip_name']}")
    print(f"  poses : {tuple(sample['poses'].shape)}  (T x 156)")
    print(f"  trans : {tuple(sample['trans'].shape)}")
    print(f"  fps   : {sample['mocap_framerate']}")

    print("\n=== Lower body only, walk + run + jog ===")
    ds_lower = CMUAMASSDataset(
        root_path=ROOT,
        motion_types=["walk", "run", "jog"],
        lower_body_only=True,
    )
    print(ds_lower.summary())
    sample_lb = ds_lower[0]
    print(f"\n  clip  : {sample_lb['subject']}/{sample_lb['clip_name']}")
    print(f"  poses : {tuple(sample_lb['poses'].shape)}  (T x 21)")
    print(f"  joints: {LOWER_BODY_JOINT_NAMES}")

    print("\n=== Clips per subject (lower body dataset) ===")
    for subj, count in sorted(ds_lower.subject_clip_counts().items()):
        print(f"  {subj:<20} {count} clips")
