# lerobot/scripts/eval_with_rerun.py
import argparse
import numpy as np
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
import rerun as rr

from lerobot.dataset.pusht import PushTEnv
from lerobot.policy.diffusion import DiffusionPolicy

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, required=True, help="Path to pretrained_model directory")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # rerun 初期化
    rr.init("eval_with_rerun", spawn=True)

    # 環境構築
    dataset_path = snapshot_download("lerobot/pusht")
    env = PushTEnv(dataset_path=dataset_path)

    # モデル読み込み
    weights = load_file(f"{args.policy_path}/model.safetensors")
    policy = DiffusionPolicy.load_from_pretrained(args.policy_path)
    policy.eval().to(args.device)

    for ep in range(args.n_episodes):
        obs = env.reset()
        rr.set_time_sequence("step")
        step_idx = 0

        while True:
            obs_tensor = {k: torch.tensor(v[None], device=args.device, dtype=torch.float32)
                          for k, v in obs.items() if isinstance(v, np.ndarray)}
            action = policy.predict_action(obs_tensor)
            action_np = action.detach().cpu().numpy()[0]

            obs, reward, done, info = env.step(action_np)

            # rerun にログを送る（手先位置・物体位置など）
            ee_pos = obs["ee_pos"]
            obj_pos = obs["obj_pos"]
            rr.set_time_sequence("step")
            rr.set_time("step", step_idx)

            rr.log("robot/ee", rr.Points3D([ee_pos], colors=[255, 0, 0]))
            rr.log("object", rr.Transform3D(translation=obj_pos))

            if done:
                break
            step_idx += 1

    print("評価と可視化が完了しました。rerun Viewer に表示されています。")

if __name__ == "__main__":
    main()
