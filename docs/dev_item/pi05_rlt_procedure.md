# pi05_rlt 実行手順書（再現用）

ブランチ: `feature/pi05-rlt`
マシン前提: CUDA GPU（開発時: DGX Spark GB10, unified 128GB）, Linux

## 0. セットアップ

```bash
cd ~/lerobot
uv sync --extra pi --extra libero        # pi05 + LIBERO シミュレータ
export MUJOCO_GL=egl                      # ヘッドレス描画
# （metaworld 検証時のみ: uv sync --extra pi --extra libero --extra metaworld）
```

チェックポイント / データセット（初回に自動DL）:

- ベースモデル: `lerobot/pi05_libero_finetuned`（LIBERO SFT済み pi05。RLT論文の「base VLA policy＝タスクSFT済みモデル」に相当）
- Stage 1 データ: `HuggingFaceVLA/libero`（LIBERO teleopデモ）

## 1. ユニットテスト（CPU）

```bash
uv run pytest tests/policies/pi05_rlt/test_rlt_units.py -v
```

対象: RLTokenEncoder/Decoder（pad不変性・自己回帰再構成・勾配）、RLTActor（full chunk出力・**reference dropout時の独立性**）、TwinCritic、target soft update、ChunkTransitionAssembler（stride-2・γ割引・終端処理・ref窓スライス）、RLTReplayBuffer、TD3更新則（critic毎回・actor遅延・done時bootstrap遮断）、policy登録。

## 2. 統合テスト（GPU・チェックポイントDL）

```bash
RLT_INTEGRATION=1 uv run pytest tests/policies/pi05_rlt/test_pi05_rlt_integration.py -v -x
```

対象: バックボーン重み同一性・freeze、**rlt_enabled=false での pi05 完全一致（atol=0）**、referenceモードのpi05一致（atol=1e-4）、z_rl抽出、actorモードの影響範囲（先頭C=10のみ）、Stage 1再構成lossの減少、Stage 1後のpi05重み不変。

## 3. Stage 1: RL Token 学習（lerobot-train を使用）

```bash
uv run lerobot-train \
  --policy.type=pi05_rlt \
  --policy.pretrained_path=lerobot/pi05_libero_finetuned \
  --policy.train_stage=rl_token \
  --policy.device=cuda \
  --policy.optimizer_lr=1e-4 \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.episodes="$(python -c 'print(list(range(200)))')" \
  --batch_size=8 \
  --steps=3000 \
  --save_freq=1000 \
  --log_freq=25 \
  --eval_freq=0 \
  --output_dir=outputs/pi05_rlt_stage1 \
  --job_name=pi05_rlt_stage1 \
  --wandb.enable=false
```

- `train_stage=rl_token` により `PI05RLTPolicy.forward()` は再構成損失 L_ro（式(2)、自己回帰・sgターゲット）を返し、optimizer は encoder/decoder のみ更新（pi05はfreeze）。
- 論文の目安: 2000〜10000 steps。まず3000で損失曲線を確認。
- 判定基準: reconstruction loss が単調減少し、初期値から1桁以上下がること。
- 成果物: `outputs/pi05_rlt_stage1/checkpoints/last/pretrained_model`

## 4. Stage 2: オンラインRL（TD3系）

```bash
MUJOCO_GL=egl uv run python -m lerobot.scripts.rlt.train_pi05_rlt_online \
  --policy.type=pi05_rlt \
  --policy.pretrained_path=outputs/pi05_rlt_stage1/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --policy.bc_beta=1.0 \
  --policy.rlt_fixed_std=0.05 \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[0]' \
  --episodes=120 \
  --warmup_episodes=15 \
  --stride=2 \
  --utd=5 \
  --batch_size=256 \
  --eval_freq_episodes=25 \
  --eval_episodes=10 \
  --save_freq_episodes=50 \
  --output_dir=outputs/pi05_rlt_stage2 \
  --seed=42
```

- warmup 15エピソード: VLA reference をそのまま実行し replay buffer を事前充填（source=0）
- 以降: stochastic actor（固定σ=0.05）で収集、遷移×UTD=5 の勾配更新（critic:actor=2:1、twin-min、γ^C bootstrap、reference dropout 50%、β‖a−ã‖²）
- 報酬: sparse binary（成功ステップで+1）
- ログ: `outputs/pi05_rlt_stage2/log.jsonl`（episode/eval/updateレコード）
- 成果物: `outputs/pi05_rlt_stage2/checkpoints/last`

## 5. 評価（LIBERO）

3条件を同一タスク・同一シードで比較する:

```bash
# (a) ベースライン: pi05そのもの
MUJOCO_GL=egl uv run lerobot-eval \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_libero_finetuned \
  --policy.device=cuda \
  --env.type=libero --env.task=libero_object --env.task_ids='[0]' \
  --eval.n_episodes=50 --eval.batch_size=1 \
  --output_dir=outputs/eval_pi05_base --seed=1000

# (b) pi05_rlt referenceモード（(a)と統計的に同等であること＝非破壊性の確認）
MUJOCO_GL=egl uv run lerobot-eval \
  --policy.type=pi05_rlt \
  --policy.pretrained_path=outputs/pi05_rlt_stage1/checkpoints/last/pretrained_model \
  --policy.rlt_actor_mode=reference \
  --policy.device=cuda \
  --env.type=libero --env.task=libero_object --env.task_ids='[0]' \
  --eval.n_episodes=50 --eval.batch_size=1 \
  --output_dir=outputs/eval_pi05_rlt_reference --seed=1000

# (c) pi05_rlt Stage 2後（actorモード・決定論）
MUJOCO_GL=egl uv run lerobot-eval \
  --policy.path=outputs/pi05_rlt_stage2/checkpoints/last \
  --policy.device=cuda \
  --env.type=libero --env.task=libero_object --env.task_ids='[0]' \
  --eval.n_episodes=50 --eval.batch_size=1 \
  --output_dir=outputs/eval_pi05_rlt_stage2 --seed=1000
```

記録する指標:

- success rate（主指標）
- 平均エピソード長（速度・throughputの代理。Stage 2 trainer の eval ログにも `mean_episode_steps` あり）
- (b) vs (a) の差（非破壊性: 差は乱数誤差の範囲であること）

## 6. LIBERO以外の環境

- **metaworld**（`--env.type=metaworld`）: pi05のmetaworld SFT checkpointが存在しないため性能評価は不可。`pi05_rlt` がenv非依存に動くこと（ロード・rollout・action shape・NaNなし・steps/sec）のパイプライン検証として実施。
- **libero_plus**（`--env.type=libero_plus`）: LIBEROの摂動版（視点・配置・光源等）。同一checkpointで (a) vs (c) を評価しロバスト性を比較。ただしLIBERO-plusはgit fork のclone+PYTHONPATH導入が必要（`docker/Dockerfile.benchmark.libero_plus` 参照）。導入できない場合はその旨を記録して省略。

## 7. 判定基準まとめ

| 項目 | 基準 |
|---|---|
| ユニットテスト | 全パス |
| 統合テスト | 全パス（特に parity atol=0） |
| Stage 1 | recon loss が >10x 減少 |
| 非破壊性 | eval (b) ≒ (a)（±数%） |
| Stage 2 | (c) の success rate ≥ (a)、または mean_episode_steps < (a)（速度改善）。学習曲線（log.jsonl の eval レコード）が改善傾向 |
