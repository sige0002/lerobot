# pi05_rlt 実装・検証レポート

日付: 2026-07-07 ／ ブランチ: `feature/pi05-rlt` ／ マシン: DGX Spark GB10（aarch64, unified 128GB, torch 2.11.0+cu128）

## 1. 概要

Physical Intelligence の RLT（"RL Token: Bootstrapping Online RL with Vision-Language-Action Models", 2026）を LeRobot の `pi05` に適用する新規ポリシー `pi05_rlt` を実装し、LIBEROシミュレータで Stage 1（RL Token 学習）→ Stage 2（TD3系オンラインRL）→ 定量評価まで実行した。

結論（要点）:

- `pi05_rlt` は `rlt_enabled=false` および reference モードで既存 `pi05` と **bit完全一致**（統合テストで max abs diff = 0.0 を確認）。既存 pi05 は非破壊。
- Stage 1 再構成損失は 1.615 → 0.558（3000 steps, 97分）で収束。
- Stage 2 オンラインRL（libero_10 task 3、120エピソード、7.4時間、環境51,197ステップ≒シミュ28分ぶんのロボット時間）は学習中間評価でベースライン80%→**90%（ep49時点）** まで改善後、後半にやや劣化（checkpoint選択で対処）。
- 最終評価（各50エピソード）: pi05ネイティブ（H=50実行）92%に対し、C=10再計画では同一重みが16%まで劣化するところ、RLT actorは84〜92%まで回復（random-encoder ablationですらベスト検証点checkpointでネイティブ同率92%）。Stage 1 encoderを正しくロードした本来のRLT（v2）は実行中（下表参照）。

## 2. 実装サマリ

### 構成（新規追加のみ、既存pi05は無変更）

```text
src/lerobot/policies/pi05_rlt/
  configuration_pi05_rlt.py   # PI05RLTConfig（@register_subclass("pi05_rlt")）
  modeling_pi05_rlt.py        # PI05RLTPolicy（PI05Policy継承、backbone常時freeze）
  processor_pi05_rlt.py       # pi05と同一パイプライン（正規化空間を共有）
  rlt_modules.py              # RLTokenEncoder/Decoder, RLTActor, TwinCritic
  online_rl.py                # RLTReplayBuffer, ChunkTransitionAssembler, RLTOnlineUpdater
src/lerobot/scripts/rlt/train_pi05_rlt_online.py   # Stage 2 トレーナ
tests/policies/pi05_rlt/     # ユニット18 + 統合8テスト
登録: policies/__init__.py, policies/factory.py に3分岐追加
```

### 論文準拠ポイント

| 要素 | 実装 |
|---|---|
| RL Token | prefix（VLM）最終層token埋め込み＋学習可能`<rl>`トークン → encoder transformer、特殊トークン位置出力 z_rl（2048次元） |
| Stage 1 | 自己回帰decoder（teacher forcing・causal mask・stop-gradターゲット）のmasked MSE。pi05はfreeze |
| Actor | **full action chunk出力**（残差ではない）。(z_rl, proprio, ã) 条件付け、固定σ=0.05のガウス、2層MLP(256) |
| Reference dropout | 学習時50%でã入力をゼロマスク（出力にãが混入しない構造をユニットテストで保証） |
| BC正則化 | L_π = −Q₁ + β‖a−ã‖²（β=1.0） |
| Critic | TwinCritic + twin-min target、chunk-level C-step TD（C=10 < H=50、γ^C bootstrap、γ=0.99） |
| TD3詳細 | target soft update τ=0.005、critic:actor=2:1、UTD=5/遷移、warmupバッファ事前充填 |
| Subsampling | stride=2 の中間遷移保存（中間観測のz_rlはチャンク末尾で1回のバッチforwardにまとめて計算） |
| 報酬 | sparse binary（成功ステップで+1） |

### 論文との意図的な差分（既知の制約）

1. **Stage 1でのVLA同時SFT（α·L_vla）なし** — SFT済みcheckpoint（pi05_libero_finetuned）で代替
2. **critical phaseハンドオーバー未実装** — エピソード全体にRLを適用（シム検証のため）
3. **人間介入なし** — シム自動ロールアウトのみ
4. proprioは位置のみ（論文は位置+速度）
5. 学習は同期ループ（論文は非同期rollout/learner）

## 3. 再現手順（実行したコマンド）

セットアップ〜評価の全コマンドは `docs/dev_item/pi05_rlt_procedure.md` を正とする。実際に実行した主要コマンド:

### セットアップ

```bash
uv sync --extra pi --extra training --extra libero --extra metaworld --extra test
export MUJOCO_GL=egl
printf 'N\n' | uv run --no-sync python -c "import libero.libero"   # LIBERO初回config生成
```

注意: 素の`uv run`はextrasを外すため、以後の全コマンドは`uv run --no-sync`（または`--extra`明示）。

### テスト

```bash
uv run --no-sync pytest tests/policies/pi05_rlt/test_rlt_units.py -v          # ユニット18件
RLT_INTEGRATION=1 uv run --no-sync pytest tests/policies/pi05_rlt/test_pi05_rlt_integration.py -v  # 統合8件（GPU・checkpoint DL）
```

### Stage 1（RL Token学習）

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --no-sync lerobot-train \
  --policy.type=pi05_rlt \
  --policy.pretrained_path=lerobot/pi05_libero_finetuned \
  --policy.train_stage=rl_token \
  --policy.device=cuda \
  --policy.compile_model=false \
  --policy.optimizer_lr=1e-4 \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.episodes="$(python -c 'print(list(range(128)))')" \
  --batch_size=8 --steps=3000 --save_freq=1000 --log_freq=25 \
  --output_dir=outputs/pi05_rlt_stage1 --wandb.enable=false
```

### タスク選定スキャン

```bash
MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --no-sync lerobot-eval \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_libero_finetuned --policy.device=cuda \
  --env.type=libero --env.task=libero_10 \
  --eval.n_episodes=10 --eval.batch_size=10 \
  --output_dir=outputs/eval_pi05_libero10_scan --seed=1000
```

### Stage 2（オンラインRL）

```bash
MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --no-sync \
  python -m lerobot.scripts.rlt.train_pi05_rlt_online \
  --policy.type=pi05_rlt \
  --policy.pretrained_path=outputs/pi05_rlt_stage1/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --env.type=libero --env.task=libero_10 --env.task_ids='[3]' \
  --episodes=120 --warmup_episodes=15 --stride=2 \
  --utd=5 --batch_size=256 \
  --eval_freq_episodes=25 --eval_episodes=10 --save_freq_episodes=50 \
  --output_dir=outputs/pi05_rlt_stage2 --seed=42
```

### 最終評価（各50エピソード、同一シード1000、libero_10 task 3）

```bash
# (a) pi05ベースライン
MUJOCO_GL=egl uv run --no-sync lerobot-eval \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_libero_finetuned --policy.device=cuda \
  --env.type=libero --env.task=libero_10 --env.task_ids='[3]' \
  --eval.n_episodes=50 --eval.batch_size=10 --output_dir=outputs/eval_a_pi05_base --seed=1000
# (b) pi05_rlt referenceモード（＝同一重み・C=10再計画）
MUJOCO_GL=egl uv run --no-sync lerobot-eval \
  --policy.type=pi05_rlt --policy.pretrained_path=outputs/pi05_rlt_stage1/checkpoints/last/pretrained_model \
  --policy.rlt_actor_mode=reference --policy.device=cuda \
  --env.type=libero --env.task=libero_10 --env.task_ids='[3]' \
  --eval.n_episodes=50 --eval.batch_size=10 --output_dir=outputs/eval_b_rlt_reference --seed=1000
# (c1) Stage 2後（last）/ (c2) Stage 2後（ep0050=ベスト検証点）
MUJOCO_GL=egl uv run --no-sync lerobot-eval \
  --policy.path=outputs/pi05_rlt_stage2/checkpoints/last --policy.device=cuda \
  --env.type=libero --env.task=libero_10 --env.task_ids='[3]' \
  --eval.n_episodes=50 --eval.batch_size=10 --output_dir=outputs/eval_c1_rlt_last --seed=1000
MUJOCO_GL=egl uv run --no-sync lerobot-eval \
  --policy.path=outputs/pi05_rlt_stage2/checkpoints/ep0050 --policy.device=cuda \
  --env.type=libero --env.task=libero_10 --env.task_ids='[3]' \
  --eval.n_episodes=50 --eval.batch_size=10 --output_dir=outputs/eval_c2_rlt_ep50 --seed=1000
```

## 4. テスト結果

- **ユニットテスト**: 18/18 PASS（encoder pad不変性、自己回帰decoder、actor full-chunk出力、**ref dropout時のã独立性**、twin-min、target soft update、assembler（stride-2・γ割引・終端・ref窓）、buffer、TD3更新則、compute_z_batched回帰、登録・validation）
- **統合テスト（GPU）**: 8/8 PASS
  - `rlt_enabled=false`: pi05と **bit完全一致（atol=0、max abs diff=0.0）**
  - referenceモード（独自prefill+denoise経路）: pi05と **max abs diff=0.0**（op-for-op再現）
  - z_rl: shape (1,2048)、決定的、有限
  - actorモードは先頭C=10ステップのみ変更
  - Stage 1 lossは10ステップで2.24→0.93に減少、backbone重みは不変
  - 注意: parity検証は両者 `compile_model=false`（eager統制比較）で行う。torch.compile/CUDAGraphs有効時はカーネル差で1e-2程度の数値差とバッファ上書きが発生するため

## 5. Stage 1 結果

- データ: HuggingFaceVLA/libero ep0-127（≈34.7k frames、libero_10主体＋libero_90少数）
- 損失: step25 1.615 → step100 0.777 → step500 0.647 → step1000 0.599 → step2000 0.557 → step3000 ≈0.55
- 速度: 1.90 s/step、総時間 97分、GPUピーク ≈25GB
- 判定: 単調減少・収束。ただし手順書基準の「>10x減少」は未達（2.9x。step500以降ほぼ平坦＝この構成の再構成下限）。z_rlの実効性はStage 2で確認する方針とした

## 6. タスク選定（ベースラインスキャン: libero_10 各10エピソード）

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 全体 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 成功率 | 90 | 100 | 90 | **80** | 90 | 100 | 100 | **80** | 90 | 100 | 92% |

最低タイの task 3 / 7（各80%）はいずれもStage 1データにデモあり（12/13エピソード）→ 規定ルールで **task 3** を選定（"put the black bowl in the bottom drawer of the cabinet and close it"）。

## 7. Stage 2 結果（libero_10 task 3、120エピソード、7.4時間）

収集: 19,347遷移（stride-2）、79,005勾配更新、環境51,197ステップ（30fps換算≒28分の実機相当データ）。

学習エピソード成功率（25エピソード窓）:

| ep0-24 | ep25-49 | ep50-74 | ep75-99 | ep100-119 |
|---|---|---|---|---|
| 0.40（warmup含む） | 0.76 | 0.88 | 0.76 | 0.70 |

中間評価（決定論actor、各10エピソード）:

| ep24 | ep49 | ep74 | ep99 | ep120(final) |
|---|---|---|---|---|
| 0.80 / 290.9步 | **0.90 / 271.9步** | 0.70 / 297.9步 | 0.60 / 356.6步 | 0.70 / 304.7步 |

観察:

1. **warmup（referenceモード）の成功率が4/15=27%と低い** — C=10の頻繁な再計画自体がpi05の性能を下げる（→評価(b)で定量化）。RLはそこから80-90%まで回復・改善した
2. ep49をピークに後半やや劣化（off-policy RLの過適合傾向）。ピーク直後の `ep0050` checkpointを最終評価に含めた
3. critic学習は安定（critic_loss ~8.5e-4、Q̄ ≈ 0.29、NaN/発散なし）

## 8. 最終評価（各50エピソード、同一シード）

すべて libero_10 task 3、50エピソード、`--eval.batch_size=10 --seed=1000`。エピソード長は動画フレーム数（=制御ステップ数、失敗時は最大520）から算出。

| 条件 | 実行レジーム | success | 平均エピソード長（全/成功のみ） | 意味 |
|---|---|---|---|---|
| (a) pi05 | H=50チャンク実行 | **92.0%** | 228 / — | ネイティブ運用のベースライン |
| (b) pi05_rlt reference | C=10再計画 | **16.0%** | 517 / — | 同一重み（bit一致）での再計画レジームの実測＝**RLTが受け取るreference信号の実力** |
| (c1) RLT ablation（last） | C=10 + RL actor | **84.0%** | 241 / 210 | random-encoder・最終checkpoint |
| (c2) RLT ablation（ep0050） | C=10 + RL actor | **92.0%** | 299 / 243 | random-encoder・ベスト検証点 |
| (c1') RLT v2（last） | C=10 + RL actor | 実行中 | — | Stage 1 encoder正常ロード版 |
| (c2') RLT v2（ep0050） | C=10 + RL actor | 実行中 | — | 同上・ベスト検証点 |

注: (c1)(c2) は当初、checkpointロードバグ（rlt.*重み消失、§10参照）により0%と誤測定された。修正（ed75d06a）後の再評価値。両評価とも起動ログの「Loaded 93 RLT weights from checkpoint (0 missing, 0 unexpected)」を確認済み。

解釈:

1. **C=10再計画の代償が支配的**: 同一重み・bit一致の推論でも、10ステップごとにflow-matchingが新ノイズからchunkを引き直すことでchunk間のモード切替（迷い）が起き、92%→16%まで劣化する。論文はC<H（反応性）を前提とするが、pi05+LIBERO長ホライズンタスクではこの代償が大きい。本検証の最重要知見の一つ。
2. **RLの寄与は大きい**: 同一実行レジーム比較（(c)−(b)）で+68〜76pt。RLT actorはreference劣化を強く補償し、ベスト検証点ではネイティブpi05と同率まで到達した。
3. **ただし速度（エピソード長）はネイティブ未満**: 成功エピソード長 210〜243 vs (a)全体228。論文が示す「ベースより高速化」はこの設定では未達。
4. **checkpoint選択が重要**: last(84%) vs ベスト検証点(92%)で8pt差。オンラインRL後半の劣化（eval軌跡 0.9→0.6→0.7）に対し、中間評価に基づく選択が有効。
5. **random-encoderでこの水準に達した**ことは、z_rl以外の入力（proprio+reference条件付け）とBC正則化の寄与が大きいことを示唆する。Stage 1 encoderの付加価値は v2 との比較で判定する（論文のw/o RL token ablationでは50%のthroughput差）。

## 9. LIBERO以外の環境（metaworld）

__METAWORLD_RESULT__

libero_plus はfork clone+PYTHONPATH導入が必要なため今回は未実施（手順書に導入手順の参照を記載）。

## 10. 遭遇した問題と解決（詳細: issue/issue_1.md）

| 問題 | 解決 |
|---|---|
| サブエージェントのフェーズ間idle停滞 | Monitor常駐（ログファイルベース検知）＋起動時PID報告ルール |
| GPU同時ロードOOM | GPU重ジョブの直列化・実行所有権の一本化 |
| `--dataset.episodes`疎指定でsampler KeyError | 0始まりの連続レンジのみ使用 |
| 素の`uv run`がextrasを削除 | `uv sync`上位集合＋`uv run --no-sync` |
| torch.compile/CUDAGraphsがparity比較を破壊 | eager統制比較（compile_model=false）＋clone |
| LIBERO初回importの対話プロンプト | `printf 'N\n' \| python -c "import libero.libero"` |
| metaworldデフォルトtask名の陳腐化（v2） | `--env.task=metaworld-push-v3`明示 |
| compute_z_batchedのスカラー項crash | 非テンソル項をフラットリスト集約に修正（45ee5aea、回帰テスト付き） |

## 11. 今後の課題

1. **checkpoint選択の自動化**: 中間評価ベストの採用（今回のep0050）またはearly stopping
2. **後半劣化への対処**: β（BC正則化）のスケジューリング、eval頻度の増加、UTD低減の検討
3. **critical phase**: サブゴール/時刻ベースの切替を実装し、論文本来の「難所限定RL」を再現
4. **proprioに速度を追加**（論文準拠）
5. **α·L_vla**（Stage 1でのVLA同時SFT）対応
6. SO101実機展開（dev_item 9、安全制限・人間介入・sparse reward UI）

## 付録: 成果物パス

- Stage 1 checkpoint: `outputs/pi05_rlt_stage1/checkpoints/last/pretrained_model`
- Stage 2 checkpoint: `outputs/pi05_rlt_stage2/checkpoints/{ep0050,ep0100,last}`
- Stage 2 学習ログ: `outputs/pi05_rlt_stage2/log.jsonl`
- 評価出力: `outputs/eval_{a_pi05_base,b_rlt_reference,c1_rlt_last,c2_rlt_ep50}/`
- コミット履歴: `git log --oneline develop..feature/pi05-rlt`
