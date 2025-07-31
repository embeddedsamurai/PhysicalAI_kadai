import os
import numpy as np
import torch
from pathlib import Path
from tqdm import trange
from types import SimpleNamespace

from lerobot.common.envs.factory import make_env_config, make_env
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

# ========= 設定 ========= #
SAVE_PATH = Path("outputs/generated_npz_episodes")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

POLICY_PATH = "outputs/train/lerobot_pusht/checkpoints/003000/pretrained_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPISODES = 1
MAX_EPISODE_LENGTH = 300
# ======================== #

# 1. ポリシー読み込み
policy = DiffusionPolicy.from_pretrained(POLICY_PATH).to(DEVICE)
policy.eval()

# 2. 環境構築
env_cfg = make_env_config("pusht")
env = make_env(env_cfg, n_envs=1)

for ep in range(NUM_EPISODES):
    obs_list = []
    act_list = []

    obs, _ = env.reset()
    obs = {k: v[0] for k, v in obs.items()}  # VectorEnvから1つ分を抽出
    done = False
    t = 0

    policy.reset()

    while not done and t < MAX_EPISODE_LENGTH:
        obs_proc = {
            k: torch.from_numpy(np.expand_dims(v, 0)).to(DEVICE)
            for k, v in obs.items() if isinstance(v, np.ndarray)
        }

        # Add image input for DiffusionPolicy
        if "pixels" in obs:
            image = obs["pixels"][..., ::-1] / 255.0  # BGR to RGB
            image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
            obs_proc["observation.image"] = image
		
		# Add observation.state
        if "agent_pos" in obs:
            obs_proc["observation.state"] = torch.from_numpy(
            np.expand_dims(obs["agent_pos"], 0)).float().to(DEVICE)
        action = policy.select_action(obs_proc).cpu().numpy()[0]
        obs_list.append(obs)
        act_list.append(action)

        obs, _, terminated, truncated, _ = env.step([action])
        obs = {k: v[0] for k, v in obs.items()}
        done = terminated[0] or truncated[0]
        t += 1

    # 保存
    np.savez_compressed(
        SAVE_PATH / f"episode-{ep:06d}.npz",
        observations=obs_list,
        actions=np.stack(act_list),
    )
    print(f"✅ Saved: episode-{ep:06d}.npz")

env.close()
