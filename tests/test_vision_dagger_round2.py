"""Guard benchmark parity, checkpoint compatibility and label/split isolation."""
import ast
import csv
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import vision_dagger_common as common
import collect_dagger_round2 as collector
from collect_dagger_round2 import sample_training_episode
from train_dagger_round2 import hard_weights, original_split, read_source, weighted_mse


def legacy_definitions():
    """Execute only pure definitions from legacy source; never run its experiment."""
    tree = ast.parse((ROOT / "scripts/evaluate_vision_bc_generalization.py").read_text(encoding="utf-8"))
    wanted = {"VisionBCPolicy", "forward_kinematics", "dynamics_terms", "expert_ctc", "vision_policy_action"}
    constants = {"L1", "L2", "m1", "m2", "lc1", "lc2", "I1", "I2", "gravity", "Kp", "Kd"}
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
             or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)]
    namespace = {"np": np, "torch": torch, "nn": nn, "math": math, "cv2": cv2}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "legacy_definitions", "exec"), namespace)
    return namespace, tree


class VisionDAggerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.old, cls.tree = legacy_definitions()

    def test_architecture_and_forward_match_legacy(self):
        torch.manual_seed(10)
        original = self.old["VisionBCPolicy"]().eval()
        new = common.VisionBCPolicy().eval()
        new.load_state_dict(original.state_dict(), strict=True)
        images, proprio = torch.rand(3, 3, 60, 80), torch.rand(3, 4)
        with torch.no_grad():
            torch.testing.assert_close(new(images, proprio), original(images, proprio), rtol=0, atol=0)

    def test_ctc_and_fk_match_legacy(self):
        rng = np.random.default_rng(12)
        for _ in range(100):
            q, dq, goal = rng.uniform(-2, 2, size=(3, 2))
            np.testing.assert_array_equal(common.forward_kinematics(q), self.old["forward_kinematics"](q))
            np.testing.assert_array_equal(common.expert_ctc(q, dq, goal), self.old["expert_ctc"](q, dq, goal))

    def test_image_normalization_and_action_match_legacy(self):
        path = common.PYTORCH_ROOT / "checkpoints/vision_dagger_r1_policy.pt"
        if not path.exists():
            self.skipTest("Local R1 checkpoint unavailable")
        policy, checkpoint = common.load_policy(path)
        old = dict(self.old)
        old["policy"] = policy
        for key in ("proprio_mean", "proprio_std", "tau_mean", "tau_std"):
            old[key] = checkpoint[key].numpy()
        function = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "vision_policy_action")
        exec(compile(ast.Module(body=[function], type_ignores=[]), "legacy_action", "exec"), old)
        rgb = np.random.default_rng(5).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        q, dq = np.array([0.3, -0.4]), np.array([0.2, -0.1])
        np.testing.assert_array_equal(common.policy_action(policy, checkpoint, rgb, q, dq),
                                      old["vision_policy_action"](rgb, q, dq))
        small = cv2.resize(rgb, (80, 60), interpolation=cv2.INTER_AREA)
        encoded = cv2.imencode(".png", cv2.cvtColor(small, cv2.COLOR_RGB2BGR))[1]
        decoded = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(common.preprocess_rgb(rgb), common.preprocess_rgb(decoded))

    def test_all_fifty_evaluation_draws_match_actual_legacy_ast(self):
        loop = next(n for n in self.tree.body if isinstance(n, ast.For))
        names = {"q_start", "dq_start", "q_goal"}
        draws = [n for n in loop.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
        self.assertEqual(len(draws), 3)
        code = compile(ast.Module(body=draws, type_ignores=[]), "legacy_draws", "exec")
        old = {"rng": np.random.default_rng(123), "np": np}
        rng = np.random.default_rng(123)
        for _ in range(50):
            exec(code, old)
            for name, value in zip(("q_start", "dq_start", "q_goal"), common.sample_evaluation_episode(rng)):
                np.testing.assert_array_equal(value, old[name])

    def test_success_boundaries(self):
        self.assertTrue(common.episode_success(0.03, np.array([0.05, -0.05])))
        self.assertFalse(common.episode_success(np.nextafter(0.03, np.inf), np.zeros(2)))
        for joint in range(2):
            dq = np.zeros(2)
            dq[joint] = np.nextafter(0.05, np.inf)
            self.assertFalse(common.episode_success(0, dq))
            self.assertFalse(common.episode_success(0, -dq))
        self.assertFalse(common.episode_success(float("nan"), np.zeros(2)))

    def test_training_distribution_is_wider_and_separate(self):
        rng = np.random.default_rng(789)
        samples = [sample_training_episode(rng, i) for i in range(50)]
        q, dq, goal = (np.array([s[i] for s in samples]) for i in range(3))
        self.assertTrue(np.any(np.abs(q) > [0.7, 0.8]))
        self.assertTrue(np.any(np.abs(dq) > 0.3))
        self.assertTrue(np.any(goal[:, 1] > 1.2))
        self.assertTrue((goal[:, 1] > 0).all())
        for start, _, target, stratum in samples:
            distance = np.linalg.norm(target - start)
            if stratum == "short_move":
                self.assertTrue(0.15 <= distance <= 0.75)
            elif stratum == "long_move":
                self.assertTrue(1.6 <= distance <= 2.6)

    def test_episode_split_keeps_all_original_test_rows_out(self):
        episodes = np.repeat(np.arange(100), 50)
        train, test = original_split(episodes)
        np.testing.assert_array_equal(np.flatnonzero(train), np.arange(4000))
        np.testing.assert_array_equal(np.flatnonzero(test), np.arange(4000, 5000))

    def test_weights_are_bounded_and_affect_loss(self):
        np.testing.assert_allclose(hard_weights(np.array([0, 1, 10, 1e9]), 10), [1, 1.2, 3, 3], rtol=1e-7)
        prediction = torch.tensor([[0., 0.], [2., 2.]], requires_grad=True)
        loss = weighted_mse(prediction, torch.zeros_like(prediction), torch.tensor([1., 3.]))
        self.assertEqual(loss.item(), 3.0)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_teacher_columns_are_labels_and_student_only_affects_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            cv2.imwrite(str(root / "images/test.png"), np.zeros((60, 80, 3), dtype=np.uint8))
            row = dict(episode=0, filename="test.png", q1=0, q2=0, dq1=0, dq2=0,
                       tau1=1, tau2=2, student_tau1=4, student_tau2=6, action_error=5)
            with (root / "labels.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            data = read_source(root, dagger=True)
            np.testing.assert_array_equal(data["targets"], [[1, 2]])
            np.testing.assert_array_equal(data["errors"], [5])

    def test_rollout_applies_student_at_every_step_and_saves_expert(self):
        student, expert = np.array([1., 2.]), np.array([11., 22.])
        model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        data = SimpleNamespace(qpos=np.zeros(2), qvel=np.zeros(2),
                               mocap_pos=np.zeros((1, 3)), qfrc_applied=np.zeros(2))
        renderer = Mock()
        renderer.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        applied = []
        def step(*_):
            applied.append(data.qfrc_applied.copy())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "round2"
            argv = ["collector", "--policy", str(Path(__file__)), "--episodes", "1", "--output-dir", str(output)]
            with patch.object(sys, "argv", argv), \
                 patch.object(collector, "load_policy", return_value=(Mock(), {})), \
                 patch.object(collector, "make_simulation", return_value=(model, data, renderer, Mock())), \
                 patch.object(collector, "policy_action", return_value=student), \
                 patch.object(collector, "expert_ctc", return_value=expert), \
                 patch.object(collector.mujoco, "mj_resetData"), \
                 patch.object(collector.mujoco, "mj_forward"), \
                 patch.object(collector.mujoco, "mj_step", side_effect=step), \
                 patch("builtins.print"):
                collector.main()
            np.testing.assert_array_equal(applied, np.tile(student, (2000, 1)))
            with (output / "labels.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 100)
            for row in rows:
                self.assertEqual([float(row["tau1"]), float(row["tau2"])], expert.tolist())
                self.assertEqual([float(row["student_tau1"]), float(row["student_tau2"])], student.tolist())
                self.assertAlmostEqual(float(row["action_error"]), np.linalg.norm(student-expert))
            renderer.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
