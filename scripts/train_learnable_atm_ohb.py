import copy
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader, Dataset

try:
    import h5py
except ImportError:  # pragma: no cover - runtime dependency
    h5py = None

from gr00t.atm.dit_atm import enable_trainable_dit_atm_ohb, collect_learnable_atm_ohb_params
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import GR00TTransform, collate as gr00t_collate


@dataclass(frozen=True)
class LiberoDatasetSpec:
    state_horizon: int = 1
    action_horizon: int = 16
    max_state_dim: int = 64
    max_action_dim: int = 64


@dataclass(frozen=True)
class LiberoStateSource:
    kind: str
    keys: tuple[str, ...]


@dataclass(frozen=True)
class LiberoDemoRecord:
    file_path: str
    demo_key: str
    image_key: str
    state_source: LiberoStateSource
    language: str
    num_steps: int


@dataclass(frozen=True)
class LiberoSampleIndex:
    demo_index: int
    timestep: int
    valid_action_steps: int


@dataclass
class Args:
    libero_root: str = "/home/xinyu/QuantVLA/LIBERO/datasets/libero_10"
    base_model_path: str = "youliangtan/gr00t-n1.5-libero-long-posttrain"
    embodiment_tag: str = EmbodimentTag.NEW_EMBODIMENT.value
    output_dir: str = "outputs/learnable_atm_ohb"
    atm_json_init: str = ""
    denoising_steps: int = 8
    batch_size: int = 4
    max_steps: int = 500
    save_every: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 4
    tune_visual: bool = False
    tune_llm: bool = False
    tune_projector: bool = False
    tune_diffusion_model: bool = False
    lambda_act: float = 3.0
    lambda_atm: float = 0.2
    lambda_ohb: float = 0.2
    lambda_reg: float = 1e-7
    log_clip: float = 0.3
    num_gpus: int = 1
    seed: int = 42
    teacher_device: str = "cuda:0"
    student_device: str = "cuda:1"


class LiberoHDF5Dataset(Dataset):
    _RGB_CANDIDATES = (
        "agentview_rgb",
        "rgb",
        "image",
        "front_rgb",
        "camera_rgb",
        "agentview_image",
    )

    def __init__(self, libero_root: str, embodiment_tag: str, spec: LiberoDatasetSpec | None = None):
        if h5py is None:
            raise ImportError(
                "h5py is required to read LIBERO HDF5 demos. Install it in the current environment first."
            )

        self.libero_root = Path(libero_root)
        self.spec = spec or LiberoDatasetSpec()
        self.embodiment_tag = EmbodimentTag(embodiment_tag)

        self.file_paths = sorted(self.libero_root.glob("*.hdf5"))
        if not self.file_paths:
            raise FileNotFoundError(f"No .hdf5 files found under {self.libero_root}")

        self.demo_records: list[LiberoDemoRecord] = []
        self.samples: list[LiberoSampleIndex] = []
        self.state_min: np.ndarray | None = None
        self.state_max: np.ndarray | None = None
        self.action_min: np.ndarray | None = None
        self.action_max: np.ndarray | None = None

        self._scan()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        record = self.demo_records[sample.demo_index]

        with h5py.File(record.file_path, "r") as f:
            demo_group = self._get_demo_group(f, record.demo_key)
            obs_group = demo_group["obs"]
            frame = self._prepare_frame(np.asarray(obs_group[record.image_key][sample.timestep]))
            state_seq = self._read_state_sequence(demo_group, record.state_source)
            actions = np.asarray(demo_group["actions"], dtype=np.float32)

        state = state_seq[sample.timestep : sample.timestep + self.spec.state_horizon]
        valid_action_steps = sample.valid_action_steps
        action_dim = actions.shape[-1]
        padded_actions = np.zeros((self.spec.action_horizon, action_dim), dtype=np.float32)
        padded_actions[:valid_action_steps] = actions[
            sample.timestep : sample.timestep + valid_action_steps
        ]

        return {
            "video": frame[None, None, ...],  # [T=1, V=1, H, W, C]
            "state": state.astype(np.float32),
            "action": padded_actions,
            "valid_action_steps": valid_action_steps,
            "annotation.human.action.task_description": record.language,
        }

    def build_collate_fn(self) -> "LiberoHDF5Collator":
        assert self.state_min is not None and self.state_max is not None
        assert self.action_min is not None and self.action_max is not None
        return LiberoHDF5Collator(
            spec=self.spec,
            embodiment_tag=self.embodiment_tag,
            state_min=self.state_min,
            state_max=self.state_max,
            action_min=self.action_min,
            action_max=self.action_max,
        )

    def _scan(self) -> None:
        skipped: list[str] = []

        for file_path in self.file_paths:
            with h5py.File(file_path, "r") as f:
                data_group = f["data"] if "data" in f else f
                language = self._extract_language(data_group, file_path)

                for demo_key in self._sorted_demo_keys(data_group):
                    demo_group = data_group[demo_key]
                    if not isinstance(demo_group, h5py.Group):
                        continue

                    image_key = self._select_image_key(demo_group)
                    state_source = self._select_state_source(demo_group)
                    if image_key is None or state_source is None or "actions" not in demo_group:
                        skipped.append(f"{file_path.name}:{demo_key}")
                        continue

                    states = self._read_state_sequence(demo_group, state_source)
                    actions = np.asarray(demo_group["actions"], dtype=np.float32)
                    image_len = int(demo_group["obs"][image_key].shape[0])
                    num_steps = min(image_len, states.shape[0], actions.shape[0])
                    if num_steps <= 0:
                        skipped.append(f"{file_path.name}:{demo_key}")
                        continue

                    states = states[:num_steps]
                    actions = actions[:num_steps]
                    self._update_stats("state", states)
                    self._update_stats("action", actions)

                    record = LiberoDemoRecord(
                        file_path=str(file_path),
                        demo_key=demo_key,
                        image_key=image_key,
                        state_source=state_source,
                        language=language,
                        num_steps=num_steps,
                    )
                    self.demo_records.append(record)
                    demo_index = len(self.demo_records) - 1

                    for timestep in range(num_steps):
                        valid_action_steps = min(self.spec.action_horizon, num_steps - timestep)
                        self.samples.append(
                            LiberoSampleIndex(
                                demo_index=demo_index,
                                timestep=timestep,
                                valid_action_steps=valid_action_steps,
                            )
                        )

        if not self.demo_records:
            raise RuntimeError(
                f"Failed to build any LIBERO demos from {self.libero_root}. "
                f"Skipped entries: {skipped[:5]}"
            )
        if self.state_min is None or self.action_min is None:
            raise RuntimeError("Failed to compute LIBERO state/action statistics during scan")

        print(
            f"Loaded {len(self.demo_records)} demos from {len(self.file_paths)} HDF5 files "
            f"with {len(self.samples)} trainable timesteps"
        )
        print(
            f"Raw state dim={self.state_min.shape[0]}, raw action dim={self.action_min.shape[0]}, "
            f"action_horizon={self.spec.action_horizon}"
        )
        if skipped:
            print(f"Skipped {len(skipped)} demo groups without usable RGB/state/action triples")

    @staticmethod
    def _sorted_demo_keys(data_group) -> list[str]:
        def key_fn(name: str) -> tuple[int, str]:
            try:
                return (int(name.split("_")[-1]), name)
            except ValueError:
                return (10**9, name)

        return sorted(data_group.keys(), key=key_fn)

    @staticmethod
    def _decode_attr(value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.bytes_):
            return value.tobytes().decode("utf-8")
        return value

    def _extract_language(self, data_group, file_path: Path) -> str:
        problem_info = data_group.attrs.get("problem_info")
        language = ""
        if problem_info is not None:
            try:
                payload = json.loads(self._decode_attr(problem_info))
                language = payload.get("language_instruction", "")
                if isinstance(language, (list, tuple)):
                    language = "".join(str(item) for item in language)
                language = str(language)
            except Exception:
                language = ""

        language = language.strip().strip('"')
        if language:
            return language

        stem = file_path.stem.replace("_demo", "")
        return stem.replace("_", " ")

    def _select_image_key(self, demo_group) -> str | None:
        if "obs" not in demo_group:
            return None
        obs_group = demo_group["obs"]

        for key in self._RGB_CANDIDATES:
            if key in obs_group:
                dataset = obs_group[key]
                if isinstance(dataset, h5py.Dataset) and dataset.ndim >= 4:
                    return key

        for key, dataset in obs_group.items():
            if isinstance(dataset, h5py.Dataset) and dataset.ndim == 4:
                shape = dataset.shape
                if shape[-1] == 3 or (len(shape) >= 4 and shape[1] == 3):
                    return key
        return None

    def _select_state_source(self, demo_group) -> LiberoStateSource | None:
        if "obs" in demo_group:
            obs_group = demo_group["obs"]

            if "ee_states" in obs_group:
                ee_states = obs_group["ee_states"]
                if ee_states.ndim >= 2 and ee_states.shape[-1] >= 7:
                    return LiberoStateSource(kind="direct", keys=("obs/ee_states",))
                if "gripper_states" in obs_group:
                    return LiberoStateSource(
                        kind="concat",
                        keys=("obs/ee_states", "obs/gripper_states"),
                    )
                return LiberoStateSource(kind="direct", keys=("obs/ee_states",))

            if all(key in obs_group for key in ("ee_pos", "ee_ori", "gripper_states")):
                return LiberoStateSource(
                    kind="concat",
                    keys=("obs/ee_pos", "obs/ee_ori", "obs/gripper_states"),
                )

            if all(key in obs_group for key in ("ee_pos", "ee_ori")):
                return LiberoStateSource(kind="concat", keys=("obs/ee_pos", "obs/ee_ori"))

        if "robot_states" in demo_group:
            return LiberoStateSource(kind="direct", keys=("robot_states",))
        if "states" in demo_group:
            return LiberoStateSource(kind="direct", keys=("states",))
        return None

    @staticmethod
    def _get_demo_group(f, demo_key: str):
        data_group = f["data"] if "data" in f else f
        return data_group[demo_key]

    @staticmethod
    def _read_path(group, path: str):
        node = group
        for part in path.split("/"):
            node = node[part]
        return node

    def _read_state_sequence(self, demo_group, source: LiberoStateSource) -> np.ndarray:
        pieces: list[np.ndarray] = []

        for path in source.keys:
            values = np.asarray(self._read_path(demo_group, path), dtype=np.float32)
            if values.ndim == 1:
                values = values[:, None]
            elif values.ndim > 2:
                values = values.reshape(values.shape[0], -1)

            if path.endswith("gripper_states") and values.shape[-1] > 1:
                values = values[:, :1]
            pieces.append(values)

        if not pieces:
            raise RuntimeError(f"Unable to build state sequence from source={source}")
        return np.concatenate(pieces, axis=-1).astype(np.float32)

    @staticmethod
    def _prepare_frame(frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3:
            raise ValueError(f"Expected a single RGB frame with 3 dims, got shape={frame.shape}")

        if frame.shape[0] == 3 and frame.shape[-1] != 3:
            frame = np.moveaxis(frame, 0, -1)
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        if frame.shape[-1] > 3:
            frame = frame[..., :3]
        if frame.shape[-1] != 3:
            raise ValueError(f"Unsupported image shape for LIBERO sample: {frame.shape}")

        if frame.dtype != np.uint8:
            frame = frame.astype(np.float32)
            if frame.size > 0 and float(frame.max()) <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
        return frame

    def _update_stats(self, prefix: str, values: np.ndarray) -> None:
        values = values.astype(np.float32)
        if values.ndim == 1:
            values = values[:, None]

        current_min = getattr(self, f"{prefix}_min")
        current_max = getattr(self, f"{prefix}_max")
        dim = values.shape[-1]

        if current_min is None or current_max is None:
            current_min = np.full((dim,), np.inf, dtype=np.float32)
            current_max = np.full((dim,), -np.inf, dtype=np.float32)
        elif current_min.shape[0] < dim:
            pad = dim - current_min.shape[0]
            current_min = np.pad(current_min, (0, pad), constant_values=np.inf)
            current_max = np.pad(current_max, (0, pad), constant_values=-np.inf)

        current_min[:dim] = np.minimum(current_min[:dim], values.min(axis=0))
        current_max[:dim] = np.maximum(current_max[:dim], values.max(axis=0))

        setattr(self, f"{prefix}_min", current_min)
        setattr(self, f"{prefix}_max", current_max)


class LiberoHDF5Collator:
    def __init__(
        self,
        spec: LiberoDatasetSpec,
        embodiment_tag: EmbodimentTag,
        state_min: np.ndarray,
        state_max: np.ndarray,
        action_min: np.ndarray,
        action_max: np.ndarray,
    ):
        self.spec = spec
        self.embodiment_tag = embodiment_tag
        self.state_min = state_min.astype(np.float32)
        self.state_max = state_max.astype(np.float32)
        self.action_min = action_min.astype(np.float32)
        self.action_max = action_max.astype(np.float32)

        self.transform = GR00TTransform(
            state_horizon=spec.state_horizon,
            action_horizon=spec.action_horizon,
            max_state_dim=spec.max_state_dim,
            max_action_dim=spec.max_action_dim,
        )
        self.transform.embodiment_tag = embodiment_tag

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        processed = []

        for feature in features:
            sample = dict(feature)
            valid_action_steps = int(sample.pop("valid_action_steps"))
            sample["state"] = self._normalize(
                sample["state"],
                self.state_min,
                self.state_max,
            )

            action = sample["action"].copy()
            if valid_action_steps > 0:
                action[:valid_action_steps] = self._normalize(
                    action[:valid_action_steps],
                    self.action_min,
                    self.action_max,
                )
            if valid_action_steps < self.spec.action_horizon:
                action[valid_action_steps:] = 0.0
            sample["action"] = action

            transformed = self.transform.apply_single(sample)
            if valid_action_steps < self.spec.action_horizon:
                transformed["action_mask"][valid_action_steps:] = False
                transformed["action"][valid_action_steps:] = 0.0
            processed.append(transformed)

        return gr00t_collate(processed, self.transform.eagle_processor)

    @staticmethod
    def _normalize(values: np.ndarray, min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32)
        dim = values.shape[-1]
        min_slice = min_values[:dim]
        max_slice = max_values[:dim]
        denom = max_slice - min_slice

        normalized = np.zeros_like(values, dtype=np.float32)
        mask = np.abs(denom) > 1e-6
        normalized[..., mask] = (
            2.0 * (values[..., mask] - min_slice[mask]) / denom[mask] - 1.0
        )
        normalized[..., ~mask] = 0.0
        return np.clip(normalized, -1.0, 1.0)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    return x


def _build_dataset(cfg: Args) -> tuple[LiberoHDF5Dataset, LiberoHDF5Collator, LiberoDatasetSpec]:
    spec = LiberoDatasetSpec()
    dataset = LiberoHDF5Dataset(
        libero_root=cfg.libero_root,
        embodiment_tag=cfg.embodiment_tag,
        spec=spec,
    )
    collator = dataset.build_collate_fn()
    return dataset, collator, spec


def _copy_action_head_weights_with_resize(
    old_state: dict[str, torch.Tensor],
    new_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    copied_tensors = 0
    partial_tensors = 0
    skipped_tensors = 0

    for key, old_tensor in old_state.items():
        if key not in new_state:
            skipped_tensors += 1
            continue

        new_tensor = new_state[key]
        if old_tensor.shape == new_tensor.shape:
            new_tensor.copy_(old_tensor)
            copied_tensors += 1
            continue

        if old_tensor.ndim != new_tensor.ndim:
            skipped_tensors += 1
            continue

        slices = tuple(slice(0, min(old_dim, new_dim)) for old_dim, new_dim in zip(old_tensor.shape, new_tensor.shape))
        if any(s.stop == 0 for s in slices):
            skipped_tensors += 1
            continue

        new_tensor[slices].copy_(old_tensor[slices])
        partial_tensors += 1

    print(
        "Action head weight transfer: "
        f"{copied_tensors} exact, {partial_tensors} partial, {skipped_tensors} skipped"
    )
    return new_state


def _maybe_recreate_action_head(model: GR00T_N1_5, spec: LiberoDatasetSpec) -> None:
    data_action_horizon = spec.action_horizon
    data_max_action_dim = spec.max_action_dim

    action_horizon_mismatch = data_action_horizon != model.action_head.config.action_horizon
    action_dim_mismatch = data_max_action_dim != model.action_head.config.action_dim
    if not (action_horizon_mismatch or action_dim_mismatch):
        return

    from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead

    print(
        f"Recreating action head: horizon {model.action_head.config.action_horizon} -> {data_action_horizon}, "
        f"action_dim {model.action_head.config.action_dim} -> {data_max_action_dim}"
    )
    old_state = copy.deepcopy(model.action_head.state_dict())
    new_cfg = copy.deepcopy(model.action_head.config)
    new_cfg.action_horizon = data_action_horizon
    new_cfg.action_dim = data_max_action_dim
    new_head = FlowmatchingActionHead(new_cfg)
    new_state = _copy_action_head_weights_with_resize(old_state, new_head.state_dict())
    new_head.load_state_dict(new_state, strict=True)
    model.action_head = new_head
    model.config.action_horizon = data_action_horizon
    model.action_horizon = data_action_horizon
    model.config.action_head_cfg["action_horizon"] = data_action_horizon
    model.config.action_head_cfg["action_dim"] = data_max_action_dim
    model.config.action_dim = data_max_action_dim
    model.action_dim = data_max_action_dim


def _freeze_all_params(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _load_teacher(cfg: Args, device: torch.device, spec: LiberoDatasetSpec) -> GR00T_N1_5:
    teacher_dtype = torch.float16 if device.type == "cuda" else torch.float32
    teacher = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=cfg.base_model_path,
        tune_llm=False,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
        torch_dtype=teacher_dtype,
    )
    _maybe_recreate_action_head(teacher, spec)
    _freeze_all_params(teacher)
    teacher.action_head.num_inference_timesteps = cfg.denoising_steps
    teacher.eval()
    teacher.to(device)
    return teacher


def _load_student(cfg: Args, device: torch.device, spec: LiberoDatasetSpec) -> GR00T_N1_5:
    from gr00t.atm.dit_atm import ensure_dit_attention_patch
    from gr00t.quantization import enable_duquant_if_configured

    student_dtype = torch.float16 if device.type == "cuda" else torch.float32
    student = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=cfg.base_model_path,
        tune_llm=cfg.tune_llm,
        tune_visual=cfg.tune_visual,
        tune_projector=cfg.tune_projector,
        tune_diffusion_model=cfg.tune_diffusion_model,
        torch_dtype=student_dtype,
    )
    _maybe_recreate_action_head(student, spec)
    ensure_dit_attention_patch(student)
    enable_duquant_if_configured(student)
    _freeze_all_params(student)
    enable_trainable_dit_atm_ohb(
        student,
        init_from_json_path=cfg.atm_json_init if cfg.atm_json_init else None,
        scope="dit",
        train_atm=True,
        train_ohb=True,
    )
    student.action_head.num_inference_timesteps = cfg.denoising_steps
    student.train()
    student.to(device)
    return student


def export_json(model: torch.nn.Module, output_path: str, log_clip: float) -> None:
    export = {}
    for name, module in model.named_modules():
        has_alpha = hasattr(module, "atm_log_alpha")
        has_beta = hasattr(module, "ohb_log_beta")
        if not (has_alpha or has_beta):
            continue
        item = {}
        if has_alpha:
            alpha = torch.exp(module.atm_log_alpha.detach().float().cpu().clamp(-log_clip, log_clip)).tolist()
            item["all"] = alpha
        if has_beta:
            beta = float(torch.exp(module.ohb_log_beta.detach().float().cpu().clamp(-log_clip, log_clip)).item())
            item["beta"] = beta
        export[name] = item
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)
    print(f"[export] wrote learned ATM/OHB JSON to {output_path}")


def maybe_strip_labels(batch: dict) -> dict:
    return dict(batch)


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _extract_tensor(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            if torch.is_tensor(item):
                return item
    if isinstance(output, dict):
        for item in output.values():
            if torch.is_tensor(item):
                return item
    return None


def _register_attn_output_hooks(model: torch.nn.Module, store: dict[str, torch.Tensor], detach: bool):
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            tensor = _extract_tensor(output)
            if tensor is None:
                return
            tensor = tensor.float()
            store[name] = tensor.detach() if detach else tensor
        return hook

    for name, module in model.named_modules():
        if name.startswith("action_head.model.transformer_blocks.") and name.endswith(".attn1"):
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles


def _flatten_feature(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x.reshape(-1, x.shape[-1])


def _compute_direct_feature_mse(
    teacher_store: dict[str, torch.Tensor],
    student_store: dict[str, torch.Tensor],
    student_device: torch.device,
) -> torch.Tensor:
    keys = sorted(set(teacher_store.keys()) & set(student_store.keys()))
    if not keys:
        dev = next(iter(student_store.values())).device if student_store else student_device
        return torch.zeros((), device=dev)

    total = None
    count = 0
    for key in keys:
        t = _flatten_feature(teacher_store[key].to(student_device, non_blocking=True))
        s = _flatten_feature(student_store[key])
        n = min(t.shape[0], s.shape[0])
        if n == 0:
            continue
        t = t[:n]
        s = s[:n]
        loss = F.mse_loss(s, t)
        total = loss if total is None else total + loss
        count += 1
    if count == 0:
        return torch.zeros((), device=student_device)
    return total / count


def _compute_direct_feature_cosine_loss(
    teacher_store: dict[str, torch.Tensor],
    student_store: dict[str, torch.Tensor],
    student_device: torch.device,
    eps: float = 1e-8,
) -> torch.Tensor:
    keys = sorted(set(teacher_store.keys()) & set(student_store.keys()))
    if not keys:
        dev = next(iter(student_store.values())).device if student_store else student_device
        return torch.zeros((), device=dev)

    total = None
    count = 0
    for key in keys:
        t = _flatten_feature(teacher_store[key].to(student_device, non_blocking=True))
        s = _flatten_feature(student_store[key])
        n = min(t.shape[0], s.shape[0])
        if n == 0:
            continue
        t = t[:n]
        s = s[:n]
        t = F.normalize(t, dim=-1, eps=eps)
        s = F.normalize(s, dim=-1, eps=eps)
        loss = (1.0 - (s * t).sum(dim=-1)).mean()
        total = loss if total is None else total + loss
        count += 1
    if count == 0:
        return torch.zeros((), device=student_device)
    return total / count


def main(cfg: Args):
    set_seed(cfg.seed)
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        teacher_device = torch.device(cfg.teacher_device)
        student_device = torch.device(cfg.student_device)
    else:
        teacher_device = torch.device("cpu")
        student_device = torch.device("cpu")

    print(f"teacher_device={teacher_device}, student_device={student_device}")

    dataset, collator, spec = _build_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collator,
    )

    teacher = _load_teacher(cfg, teacher_device, spec)
    student = _load_student(cfg, student_device, spec)

    learnable_params = collect_learnable_atm_ohb_params(student)
    if not learnable_params:
        raise RuntimeError("No learnable ATM/OHB parameters found on student model")

    print(f"Found {len(learnable_params)} learnable ATM/OHB tensors")
    optimizer = torch.optim.AdamW(learnable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    teacher_attn_store: dict[str, torch.Tensor] = {}
    student_attn_store: dict[str, torch.Tensor] = {}
    teacher_hooks = _register_attn_output_hooks(teacher, teacher_attn_store, detach=True)
    student_hooks = _register_attn_output_hooks(student, student_attn_store, detach=False)

    best_loss = float("inf")
    step = 0

    try:
        while step < cfg.max_steps:
            for batch in loader:
                if step >= cfg.max_steps:
                    break
                step += 1

                teacher_inputs = move_to_device(maybe_strip_labels(batch), teacher_device)
                student_inputs = move_to_device(maybe_strip_labels(batch), student_device)

                if step == 1:
                    print("teacher batch keys:", sorted(teacher_inputs.keys()))
                    for k, v in teacher_inputs.items():
                        if torch.is_tensor(v):
                            print(f"  teacher {k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
                    for k, v in student_inputs.items():
                        if torch.is_tensor(v):
                            print(f"  student {k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")

                teacher_attn_store.clear()
                student_attn_store.clear()

                with torch.no_grad():
                    with _autocast_context(teacher_device, torch.float16):
                        teacher_out = teacher.get_action(teacher_inputs)
                        teacher_action = teacher_out["action_pred"].float().to(student_device, non_blocking=True)

                with _autocast_context(student_device, torch.float16):
                    student_out = student.get_action(student_inputs)
                    student_action = student_out["action_pred"].float()

                    loss_act = F.mse_loss(student_action, teacher_action)
                    # Direct feature-to-feature training, no std/RMS proxy.
                    loss_atm = _compute_direct_feature_mse(teacher_attn_store, student_attn_store, student_device)
                    loss_ohb = _compute_direct_feature_cosine_loss(teacher_attn_store, student_attn_store, student_device)

                    loss_reg = torch.zeros((), device=student_action.device)
                    for module in student.modules():
                        if hasattr(module, "atm_log_alpha"):
                            loss_reg = loss_reg + (module.atm_log_alpha ** 2).mean()
                        if hasattr(module, "ohb_log_beta"):
                            loss_reg = loss_reg + (module.ohb_log_beta ** 2).mean()

                    loss = (
                        cfg.lambda_act * loss_act
                        + cfg.lambda_atm * loss_atm
                        + cfg.lambda_ohb * loss_ohb
                        + cfg.lambda_reg * loss_reg
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if step % 10 == 0 or step == 1:
                    print(
                        f"step={step} "
                        f"loss={loss.item():.6f} "
                        f"loss_act={loss_act.item():.6f} "
                        f"loss_atm={loss_atm.item():.6f} "
                        f"loss_ohb={loss_ohb.item():.6f} "
                        f"loss_reg={loss_reg.item():.6f}"
                    )

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    export_json(student, str(outdir / "atm_alpha_beta_learned_best.json"), cfg.log_clip)
                    torch.save(
                        {"step": step, "best_loss": best_loss, "optimizer": optimizer.state_dict()},
                        outdir / "train_state_best.pt",
                    )

                if step % cfg.save_every == 0:
                    export_json(student, str(outdir / f"atm_alpha_beta_step{step}.json"), cfg.log_clip)
                    torch.save(
                        {"step": step, "optimizer": optimizer.state_dict()},
                        outdir / f"train_state_step{step}.pt",
                    )
    finally:
        for h in teacher_hooks + student_hooks:
            h.remove()

    export_json(student, str(outdir / "atm_alpha_beta_learned_final.json"), cfg.log_clip)
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "best_loss": best_loss},
        outdir / "train_state_final.pt",
    )
    print("Training complete.")


if __name__ == "__main__":
    cfg = tyro.cli(Args)
    print("=" * 80)
    print(cfg)
    print("=" * 80)
    main(cfg)
