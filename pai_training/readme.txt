0. 訓練
#$ pixi run jupyter notebook

:~/pai_training$ pixi run python -m lerobot.scripts.train \
  --dataset.repo_id='lerobot/pusht' \
  --policy.type=diffusion \
  --steps=3000 \
  --batch_size=32 \
  --output_dir=outputs/train/lerobot_pusht \
  --job_name=lerobot_pusht \
  --policy.device=cuda \
  --wandb.enable=false

1. npzファイルの作成
:~/pai_training$ pixi run python lerobot/lerobot/scripts/save_rollout_as_npz.py
2. 1つめのコマンドプロンプト
:~/pai_training$ pixi run rerun --serve-web --web-viewer --web-viewer-port 9090
3. 2つめのコマンドプロンプト
:~/pai_training$ pixi run python lerobot/lerobot/scripts/visualize_dataset_npz.py   --npz-path outputs/generated_npz_episodes/episode-000000.npz
4. Microsoft Edgeで、
　http://localhost:9090/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9876%2Fproxy
　にアクセス
