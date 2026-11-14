"""Side-effect-free R2 helpers, preserving the original vision BC contract.

Legacy scripts run experiments on import, so their model/dynamics are copied
here verbatim in behavior and checked against those sources by regression tests.
"""
from pathlib import Path
import hashlib
import math

import cv2
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTORCH_ROOT = PROJECT_ROOT.parent / "pytorch-learning"
MODEL_PATH = PROJECT_ROOT / "models" / "two_link_arm.xml"
L1, L2 = 1.0, 0.8
m1, m2, lc1, lc2 = 1.0, 0.8, 0.5, 0.4
I1, I2, gravity = 0.0833333333, 0.0426666667, 9.81
Kp = np.array([25.0, 25.0])
Kd = np.array([10.0, 10.0])


class VisionBCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.image_fc = nn.Linear(64 * 7 * 10, 128)
        self.proprio_fc1 = nn.Linear(4, 32)
        self.proprio_fc2 = nn.Linear(32, 32)
        self.fusion_fc1 = nn.Linear(160, 128)
        self.output_layer = nn.Linear(128, 2)

    def forward(self, image, proprio):
        x = self.pool(self.relu(self.conv1(image)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        image_feature = self.relu(self.image_fc(torch.flatten(x, start_dim=1)))
        p = self.relu(self.proprio_fc1(proprio))
        proprio_feature = self.relu(self.proprio_fc2(p))
        combined = torch.cat([image_feature, proprio_feature], dim=1)
        return self.output_layer(self.relu(self.fusion_fc1(combined)))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("image_width", 80), checkpoint.get("image_height", 60)) != (80, 60):
        raise ValueError("Expected the existing 80x60 vision policy")
    for key, size in (("proprio_mean", 4), ("proprio_std", 4),
                      ("tau_mean", 2), ("tau_std", 2)):
        value = checkpoint[key]
        if tuple(value.shape) != (size,) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid normalization: {key}")
        if key.endswith("std") and not (value > 0).all():
            raise ValueError(f"Non-positive normalization: {key}")
    policy = VisionBCPolicy()
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()
    return policy, checkpoint


def preprocess_rgb(rgb):
    image = cv2.resize(rgb, (80, 60), interpolation=cv2.INTER_AREA)
    return np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))


@torch.no_grad()
def policy_action(policy, checkpoint, rgb, q, dq):
    image = torch.tensor(preprocess_rgb(rgb), dtype=torch.float32).unsqueeze(0)
    proprio = np.concatenate([q, dq])
    proprio = (proprio - checkpoint["proprio_mean"].numpy()) / checkpoint["proprio_std"].numpy()
    state = torch.tensor(proprio, dtype=torch.float32).unsqueeze(0)
    normalized = policy(image, state).numpy()[0]
    return normalized * checkpoint["tau_std"].numpy() + checkpoint["tau_mean"].numpy()


def forward_kinematics(q):
    q1, q2 = q
    return np.array([L1 * math.cos(q1) + L2 * math.cos(q1 + q2),
                     L1 * math.sin(q1) + L2 * math.sin(q1 + q2)])


def dynamics_terms(q, dq):
    q1, q2 = q
    dq1, dq2 = dq
    cos_q2, sin_q2 = math.cos(q2), math.sin(q2)
    M11 = I1 + I2 + m1 * lc1**2 + m2 * (L1**2 + lc2**2 + 2.0 * L1 * lc2 * cos_q2)
    M12 = I2 + m2 * (lc2**2 + L1 * lc2 * cos_q2)
    M22 = I2 + m2 * lc2**2
    M = np.array([[M11, M12], [M12, M22]])
    h = m2 * L1 * lc2 * sin_q2
    c = np.array([-h * (2.0 * dq1 * dq2 + dq2**2), h * dq1**2])
    g = np.array([m1 * gravity * lc1 * math.cos(q1)
                  + m2 * gravity * (L1 * math.cos(q1) + lc2 * math.cos(q1 + q2)),
                  m2 * gravity * lc2 * math.cos(q1 + q2)])
    return M, c, g


def expert_ctc(q, dq, q_goal):
    desired_acceleration = Kp * (q_goal - q) - Kd * dq
    M, c, g = dynamics_terms(q, dq)
    return M @ desired_acceleration + c + g


def make_simulation():
    # The learning project's environment does not need MuJoCo to train.
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    if not np.isclose(model.opt.timestep, 0.002, rtol=0, atol=1e-12):
        raise ValueError("Model timestep changed; expected the original 0.002 seconds")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.5, 0.0, 0.0]
    camera.distance, camera.azimuth, camera.elevation = 4.0, 90, -90
    return model, data, renderer, camera


def sample_evaluation_episode(rng):
    """Keep draw order, distribution and number of draws exactly as the old test."""
    q_start = rng.uniform(low=[-0.7, -0.8], high=[0.7, 0.8])
    dq_start = rng.uniform(low=-0.3, high=0.3, size=2)
    q_goal = np.array([rng.uniform(-0.8, 0.8), rng.uniform(0.2, 1.2)])
    return q_start, dq_start, q_goal


def episode_success(tip_error, final_dq):
    return bool(tip_error <= 0.03 and np.all(np.abs(final_dq) <= 0.05))
