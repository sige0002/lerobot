# pi05にRLT適用

## 目的

Physical Intelligence の RLT（RL Token）を LeRobot 上の `pi05` に適用する。

対象研究:

* 研究ページ: https://pi.website/research/rlt
* 論文: "RL Token: Bootstrapping Online RL with Vision-Language-Action Models"（PDF: https://pi.website/download/rlt.pdf）

本取り組みでは、既存の `pi05` ポリシーを直接改造するのではなく、`pi05_rlt` のような新しいポリシーとして実装する。
これにより、既存の `pi05` の推論・学習機能を保持しつつ、RLTによるオンライン強化学習機能を追加できる構成を目指す。

> **改訂履歴（2026-07-07）**: 論文全文・参考実装2件・LeRobotコードベースとの照合検証を実施し、以下を修正した。
>
> 1. Actorを残差方式（`final = ref + α·delta`）から論文準拠の **full action chunk出力＋BC正則化＋reference-action dropout** に変更（残差方式は論文が比較で劣ると報告した Policy Decorator / PLD 側の設計だった）
> 2. Stage 1 を論文準拠に詳細化（自己回帰transformer decoder、最終層token埋め込み、任意のVLA同時SFT）
> 3. Stage 2 を TD3系に固定し、chunk-level TD・UTD比・warmup 等の論文パラメータを明記
> 4. 報酬設計（sparse binary）と critical phase 運用を追加
> 5. LeRobotコードベース調査で判明した実装上の注意（hidden state取得、strictロード、ポリシー登録、正規化境界）を追加
> 6. 記号を論文に合わせた（**β = BC正則化係数**。論文のαはStage 1のVLA-SFT重みであり、residual scaleではない）

---

## 背景

`pi05` は画像・言語・ロボット状態を入力として、action chunkを生成するVLA系ポリシーである。
一方で、精密な接触作業や最後の位置合わせでは、事前学習済みポリシーだけでは十分に安定しない可能性がある。

RLTでは、VLAの最終層token埋め込みを encoder-decoder transformer のボトルネックで圧縮した **RL Token** を状態表現とし、その上で軽量なActor-CriticをオンラインRL（TD3系）で学習する。
Actorは VLA の出す reference action chunk を**入力条件**として受け取り、**最終action chunkそのものを直接出力**する。reference への近さは BC正則化と reference-action dropout で担保する。
これにより、VLA本体を更新せずに、タスク固有の精密フェーズを数分〜数時間の実データで改善することを狙う。

本プロジェクトでは、この考え方をLeRobotの`pi05`に適用し、以下のような構成を目指す。

```text
observation + language
        ↓
      pi05（frozen）
        ↓
reference action chunk ã + 最終層hidden state
        ↓                        ↓
        ↓                 RL Token Encoder（frozen after Stage 1）
        ↓                        ↓
        └────→ RLT Actor ←── z_rl + proprio state
                  ↓
        final action chunk（full chunk を直接出力）
```

注意: 論文のVLAは π0.6 である。本プロジェクトは π0.5（LeRobotの`pi05`、flow-matching action expert）への移植であり、reference action のサンプルには denoise ループの実行コストがかかる点が異なる。

---

## 論文の手法要約（設計の基準）

実装判断に迷ったら本節と論文本文を正とする。

### Stage 1: RL Token の学習

* VLA最終層のtoken埋め込み列 `z_{1:M}`（画像トークン。固定命令タスクでは言語埋め込みは落としてよい＝論文脚注1）に、学習可能な特殊トークン `e_rl` を末尾に付加し、**軽量encoder transformer** に通す。特殊トークン位置の出力が `z_rl`（論文では 1×2048）。
* **decoder transformer** が `z_rl` から元の埋め込み列を**自己回帰的に再構成**（teacher forcing、ターゲットは stop-gradient `sg(z_i)`、損失はMSE。式(2)）。単一ベクトルの単純MSE再構成ではない。
* VLAは再構成損失に対してはfreeze。ただし論文の実験では **タスクdemoでのVLA SFT（α·L_vla）を併走**させて「base VLAポリシー」を作っている（demo 1〜10時間、2000〜10000 gradient steps）。
* Stage 1完了後、VLAとRL Token encoderの両方をfreeze。

### Stage 2: オンラインRL（TD3系）

* 状態: `x = (z_rl, s^p)`。`s^p` は proprio（**位置＋速度**）。
* Actor: `π_θ(a_{1:C} | x, ã_{1:C}) = N(μ_θ(x, ã), σ²I)`。**固定の小さいσ**（論文実装は0.05以下）。出力はfull action chunk。2層MLP(256)、難タスクは3層MLP(512)。
* **chunk長 C=10 < H=50**（VLAはH=50を予測、RLは先頭C stepを使い反応性を確保）。
* Actor損失: `L_π = E[ −Q_ψ(x, a) + β‖a − ã‖² ]`（式(5)）。**BC正則化βはablationで単独最大の効果**。
* **Reference-action dropout**: 学習バッチの50%で ã をゼロマスクし、独立した行動生成経路を維持（copy-collapse防止）。**推論時は常に ã を渡す**。
* Critic: Twin Q（2つのQのminでtarget計算）、chunk-level TD:
  `Q̂ = Σ_{t'=1}^{C} γ^{t'−1} r_{t'} + γ^C min_j Q_{ψ'_j}(x', a'∼π_θ)`（式(3)）。target networkはTD3準拠のsoft update（τ=0.005）。
* **Warmup**: 学習開始前に VLA の ã をそのまま実行して replay buffer を事前充填。
* **Subsampling**: 中間観測を stride 2 で保存（`<x_0, a_{0:C}>, <x_2, a_{2:C+2}>, ...`）しデータ効率を向上。
* 更新: **UTD比5**（環境遷移あたり5更新）、**critic 2回 : actor 1回**、rollout と learner は非同期。
* 報酬: **sparse binary**（成功時に+1、オペレータ/成功判定器が付与）。
* **Critical phase 限定**: タスク全体ではなく最難関フェーズのみRLを適用。人間がbase→RLの切替と成功/失敗判定を行う（test時は切替予測をVLAにSFDして自動化可能）。
* 人間介入: 介入action `a^h` が actor 出力を上書きし、**replay 内では reference も `ã ← a^h` に置換**（Algorithm 1, L12）。

SACのエントロピー目的・学習σは論文にない。アルゴリズムは**TD3系に固定**する。

---

## 前提

* LeRobot上で実行可能であること
* まずはLIBEROなどのシミュレーション環境で検証する
* その後、SO101への展開を検討する
* SO101への展開方法は `dev_item` に記載する
* 参考実装は添付するが、実装内容が正しいとは限らない（→下記監査結果）
* Codexなどのコードエージェントと連携しながら、設計・実装・検証を進める
* 既存の`pi05`を壊さないため、新規ポリシーとして実装する
* 論文はπ0.6、本計画はπ0.5への移植であることを認識して進める

---

## 参考実装（監査済み）

2026-07-07 に両リポジトリを論文と照合監査した。**どちらも第三者による非公式再現**である。使い分けを誤らないこと。

* `Yyshadow/openpi-RLT`（JAX/Flax、Ethernet 1タスクの定性デモあり）

  * **Stage 2（actor-critic）は論文にほぼ忠実**: full chunk出力＋reference条件付け＋50% dropout＋BC正則化＋twin-min＋γ^C TD＋stride-2＋UTD5＋2:1＋非同期＋人間介入。`rlt_online_rl/src/rlt_online_rl/{networks,trainer,replay}.py` を **Stage 2 の設計参照にする**
  * してはいけない: decoderが**非自己回帰**（論文Eq.2と別物）なのでStage 1はコピーしない。Ethernet設定のloss重み（Q×0.1, BC×5 ≒ 実効β50の過剰アンカー）と `delta_penalty`（step間平滑化）は論文にない独自改変なので持ち込まない

* `yknxh/rlt-openpi`（PyTorch、実VLAでの動作実績なし）

  * **Stage 1（RL Token）は論文に忠実**: 自己回帰decoder・teacher forcing・stop-grad・特殊token（`src/rlt_openpi/models/rl_token.py`）。**Stage 1 の設計参照にする**。`td3_utils.py`/`TwinQCritic` の数式部分も式(3)一致
  * してはいけない: **Actorは参照禁止**。残差方式（論文が否定した設計）であり、かつ残差加算により reference dropout が無効化されるバグを持つ。他にも、rollout時の探索ノイズ欠如、UTDの誤解釈（エピソードあたり5更新＝実効UTD≪1）、stride-2未結線、正規化空間の不整合、介入時に ã 未置換、進捗+0.5の独自報酬シェイピングがあるため、Stage 2は全体的に信用しない

* 共通: 埋め込み抽出はどちらもopenpi内部構造へのパッチに依存しており直接移植不可。LeRobot `pi05` 用に新規実装する。

---

## 実装方針

### 基本方針

既存の`pi05`を直接変更するのではなく、以下のような新規ポリシーを追加する。

```text
src/lerobot/policies/pi05_rlt/
  ├── __init__.py
  ├── configuration_pi05_rlt.py
  ├── modeling_pi05_rlt.py
  ├── processor_pi05_rlt.py
  ├── rlt_modules.py        # RLTokenEncoder / RLTokenDecoder / RLTActor / TwinCritic
  └── online_rl.py          # TD3系更新則（RLAlgorithm継承）
```

ポリシー名は以下を想定する。

```text
policy.type = "pi05_rlt"
```

リポジトリ名やブランチ名としては `pi05-RLT` でもよいが、Pythonモジュール名やLeRobotのpolicy typeでは `_` を使い、`pi05_rlt` とする。

### LeRobotコードベース上の実装注意（調査済み）

1. **ポリシー登録**: ファイル作成だけでは `--policy.type=pi05_rlt` は通らない。
   `@PreTrainedConfig.register_subclass("pi05_rlt")` を付けた上で、`policies/__init__.py` に import を追加する（デコレータはimport時に実行されるため）。命名規約（`PI05RLTConfig` / `modeling_pi05_rlt.py` / `PI05RLTPolicy` / `make_pi05_rlt_pre_post_processors`）を守れば `factory.py` のフォールバック解決にも乗る。
2. **checkpoint読み込み**: `PI05Policy.from_pretrained` は **strict=True が既定**（`modeling_pi05.py:956`）。RLTモジュールを追加すると missing keys で失敗するため、`PI05RLTPolicy.from_pretrained` で `strict=False` にするか、pi05重みを `self.model` に限定ロードする。
3. **hidden state取得**: prefix/suffix の hidden state は既存の関数境界では取得できない（`sample_actions` は prefix_output を破棄、`denoise_step` は射影後のvelocityのみ返す）。`sample_actions` / `denoise_step` を**オーバーライド**して捕捉する（RTCが`denoise_step`を差し替えるパターンが参考になる）。`compile_model=True` だと forward hook は壊れやすいのでオーバーライド方式を採る。
4. **正規化境界**: 正規化はポリシー内ではなく**プロセッサパイプライン側**（QUANTILES）。`predict_action_chunk` の返り値は正規化空間である。**RLのactor・critic・replay bufferはすべて正規化空間で統一**し、生空間への変換は既存の後処理（Unnormalize）に任せる。`use_relative_actions` 使用時の相対/絶対×正規化/生の二軸に注意。
5. **RL基盤の再利用**: `src/lerobot/rl/buffer.py` の `ReplayBuffer`（HIL-SERL由来）は dev_item 6 でほぼ流用可。ただし **TD3は未実装**、SACは `gaussian_actor` に密結合なので、RLT用は `RLAlgorithm` を継承した新規アルゴリズムとして実装し `rl/algorithms/factory.py` に登録する。

---

## 全体構成

### 通常のpi05

```text
observation
  ├── image
  ├── state
  └── language
        ↓
      pi05
        ↓
reference action chunk ã (H=50)
```

### pi05_rlt

```text
observation
  ├── image
  ├── state
  └── language
        ↓
      pi05（frozen）
        ├── reference action chunk ã（先頭C stepを使用, C=10想定）
        └── 最終層 token embeddings
                  ↓
          RL Token Encoder（frozen after Stage 1）
                  ↓
                z_rl
                  ↓
   RLT Actor(z_rl, proprio, ã) → final action chunk a_{1:C}
   （学習時: aは N(μ_θ, σ²I) からサンプル、ãは50%でゼロマスク）
   （推論時: a = μ_θ、ãは常に付与）
```

行動生成は以下のとおり（**残差加算ではない**）。

```text
a_{1:C} = μ_θ(z_rl, proprio, ã_{1:C}) + ε,  ε ~ N(0, σ²I)（固定小σ、学習時のみ）
```

reference への近さは以下の3点で担保する。

```text
1. Actor損失のBC正則化: L_π = −Q(x, a) + β‖a − ã‖²   ← βが主要ハイパーパラメータ
2. Reference-action dropout: 学習時50%で ã をゼロマスク（推論時は常時付与）
3. Warmup: 学習開始前は ã をそのまま実行
```

`rlt_enabled=false` のときは actor を完全にバイパスし、pi05 の ã をそのまま返す（既存pi05と同一出力の保証はこのフラグで行う）。

---

## 実装ステップ

### Step 1: pi05_rltポリシーの追加

`PI05Policy`を継承した`PI05RLTPolicy`を作成する。

この段階では、RLTの処理はまだ有効化せず（`rlt_enabled=false`）、既存の`pi05`と同じ出力を返すことを目標にする。

確認項目:

* `policy.type=pi05_rlt` でLeRobotから読み込めること（`policies/__init__.py` への登録を含む）
* 既存の`pi05` checkpointを読み込めること（**strict=False オーバーライドが必要**）
* `predict_action_chunk()` が既存`pi05`と同じ形状・同じ値のaction chunkを返すこと
* LIBEROなどの環境で推論が動作すること

---

### Step 2: RL Token抽出機能の追加

`pi05`の内部hidden stateからRL Tokenを抽出する機能を追加する。

対象は**backbone（prefix）最終層のtoken埋め込み列**とする（論文で確定済み。他の層は将来のablation扱い）。固定命令タスクでは言語埋め込みは落としてよい。

`sample_actions` / prefix prefill をオーバーライドして埋め込みを捕捉し、以下を実装する。

```python
def extract_rl_token(self, batch):
    # prefix最終層 token embeddings [B, M, D] を捕捉
    # → encoder transformer（末尾に学習可能<rl>トークン付加）
    # → 特殊トークン位置の出力を z_rl として返す
    return z_rl  # [batch_size, rlt_token_dim]  (論文は D=2048, token数1)
```

この段階では、`z_rl` はまだ行動生成には使わない。

確認項目:

* `z_rl` のshapeが安定していること
* batch処理に対応していること
* GPU上で動作すること
* `pi05`本体の推論結果が変わらないこと
* `compile_model=True` でも動作すること（hookでなくオーバーライドで実装していれば問題ない）

---

### Step 3: RLT moduleの追加

`rlt_modules.py` に以下のモジュールを実装する。

```text
RLTokenEncoder:  軽量transformer。prefix最終層埋め込み列 + <rl>トークン → z_rl
RLTokenDecoder:  軽量transformer。z_rl から埋め込み列を自己回帰再構成（teacher forcing, causal mask, 線形出力射影）
RLTActor:        MLP（2層×256、難タスクは3層×512）。(z_rl, proprio, ã_{1:C}) → μ_θ ∈ R^{C×d}（full chunk）
TwinCritic:      MLP×2。(z_rl, proprio, a_{1:C}) → (Q1, Q2)
```

* Encoder/DecoderはMLPではなく**transformer**とする（ボトルネックの質が論文と変わるため）。参考: `rlt-openpi/src/rlt_openpi/models/rl_token.py`
* Actorは**残差ではなくfull chunkを出力**する。参考: `openpi-RLT/rlt_online_rl/src/rlt_online_rl/networks.py`
* Actorには reference-action dropout（学習時50%ゼロマスク）を実装する。**ゼロマスク時に出力へ ã が混入しない構造であること**（残差加算だとdropoutが無効化される — rlt-openpiのバグ）
* 既存pi05との出力一致は Step 1 の `rlt_enabled=false` バイパスで担保する（Actorのゼロ初期化には依存しない）

---

### Step 4: Stage 1学習

Stage 1では、RL Token Encoder/Decoderを学習する。

```text
pi05 prefix最終層 token embeddings z_{1:M}
        ↓ （+ 学習可能<rl>トークン）
RLTokenEncoder → z_rl
        ↓
RLTokenDecoder（自己回帰、teacher forcing）
        ↓
reconstructed embeddings ẑ_{1:M}
```

損失関数（論文 式(2)）:

```text
L_ro = Σ_i ‖ẑ_i − sg(z_i)‖²    # ターゲットはstop-gradient
（＋ 任意で α·L_vla: タスクdemoによるpi05のSFTを併走）
```

方針:

* ターゲット埋め込みには **stop-gradient** をかける（pi05はL_roに対してfreeze）
* まずは α=0（pi05完全freeze）で開始してよいが、**論文の「base VLAポリシー」はタスクSFT済みモデル**である。LIBEROではSFT済みcheckpoint（例: pi05_libero_finetuned）を前提にすることで代替する
* データはLeRobot dataset（LIBEROのdemo）を使用。報酬は不要
* 学習量の目安: 論文はdemo 1〜10時間、2000〜10000 gradient steps
* checkpoint保存後、Encoder/Decoderともfreeze（Decoderは以後不要）

---

### Step 5: Actor統合とpass-through検証

Stage 1で学習した`RLTokenEncoder`を使い、Actorを推論経路に統合する。

```text
ã = pi05(obs)             # H=50、先頭C=10を使用
z_rl = RLTokenEncoder(prefix embeddings)
a = μ_θ(z_rl, proprio, ã_{1:C})   # 学習前はwarmupモードで a = ã をそのまま実行
```

段階的に確認する:

1. `rlt_enabled=false` → 既存pi05と完全一致
2. `rlt_enabled=true, warmup` → ã をそのまま実行（pi05と同等挙動のままreplay bufferが埋まる）
3. actor有効 → N(μ_θ, σ²I) からサンプルして実行

確認項目:

* `final_action` のshapeが既存`pi05`と同じであること（C stepぶん）
* warmup時に既存`pi05`と同じ動作になること
* actionのスケールが壊れないこと（**actor入出力・bufferは正規化空間で統一**）
* normalization / unnormalization がプロセッサ境界で正しく処理されること

---

### Step 6: Stage 2オンラインRL

Stage 2では、Actor-CriticをTD3系オンラインRLで学習する。

基本方針:

* `pi05`本体・`RLTokenEncoder`はfreeze
* `RLTActor`と`TwinCritic`を学習
* アルゴリズムは**TD3系に固定**（twin-min target、target network soft update τ=0.005、γ=0.99。SACのエントロピー目的は使わない）
* replay bufferは`src/lerobot/rl/buffer.py`をベースに、**chunk単位遷移 ⟨x, a_{1:C}, ã, r, x'⟩** を保存
* **stride-2 subsampling** で中間観測も保存
* 報酬は**sparse binary**（成功+1）。simでは環境の成功判定を使う
* まずはシミュレーションで検証する

学習ループの概略:

```text
0. Warmup: ã をそのまま実行して replay buffer を N_warm 遷移ぶん事前充填
1. 環境からobservationを取得
2. pi05で ã（H step）を生成、先頭C stepを切り出し
3. z_rl を抽出、x = (z_rl, proprio) を構成
4. a ~ N(μ_θ(x, ã), σ²I) をサンプル（探索ノイズはここ。決定論で回さない）
5. 環境で a_{1:C} を実行し、r, next_observation, done を取得
6. ⟨x, a, ã, r, x'⟩ を stride-2 subsample で replay bufferに保存
7. 遷移ごとに UTD=5 で更新（critic 2回につき actor 1回）
   - critic: Q̂ = Σ γ^{t'−1} r + γ^C min_j Q_{ψ'_j}(x', a'~π_θ) へのTD回帰
   - actor:  L_π = −Q(x, a) + β‖a − ã‖²（ãは50%ゼロマスク）
8. target networkをsoft update
```

実装上の注意:

* UTDは「**環境遷移あたり**5更新」。エピソードあたり5更新（rlt-openpiの誤り）にしないこと
* rollout と learner は非同期が理想（`lerobot.rl` の分散learner/actor骨格を流用可）。simでは同期でもよいがUTDの意味を守る
* 人間介入を扱う場合、介入時は `a ← a^h` かつ **`ã ← a^h` に置換**して保存する

---

## Critical phase の扱い

論文はタスク全体ではなく**最難関フェーズのみ**にRLを適用する（サンプル効率の中核）。

* LIBEROでは、まず critical phase を模擬する: エピソードの前半はbase pi05で実行し、所定の時刻またはサブゴール到達でRL policyに切り替える。あるいはタスクを短いreach/insert相当の区間に切って初期状態をランダム化する
* 難しければ第一段階はエピソード全体へのRL適用でパイプラインを検証し、critical phase切替は第二段階とする（その場合、論文よりhorizonが長くなり学習が遅い可能性を認識しておく）
* 実機（SO101）では人間が切替と成功判定を行う。将来的に切替予測をVLAにSFTして自動化できる

---

## LIBEROでの検証

まずはLIBEROなどのシミュレーション環境で検証する（`lerobot-eval` + `--env.type=libero` の既存パスに乗せる）。

検証目的:

* LeRobot上で`pi05_rlt`が動作すること
* 既存`pi05`と同じ推論ができること（`rlt_enabled=false`）
* RLT module追加後もaction shapeやnormalizationが壊れないこと
* Stage 1学習が可能であること（reconstruction lossが下がる）
* Stage 2のオンラインRLループが動作すること

最初の評価では、RLTによる性能向上よりも、以下を優先する。

```text
動くこと
壊れないこと
既存pi05と比較できること
ログが取れること
```

---

## 評価項目

### 機能確認

* `policy.type=pi05_rlt` でロードできるか
* 既存`pi05` checkpointを読み込めるか
* `predict_action_chunk()` が動作するか
* `select_action()` が動作するか
* `rlt_enabled=false` のとき、既存`pi05`と同一の出力になるか

### Stage 1評価

* RL Token Encoderのlossが下がるか
* hidden stateを再構成できるか
* `z_rl` の次元を変えたときの性能差（論文既定は2048）
* （ablation）他layerのhidden stateとの比較

### Stage 2評価

* rewardが改善するか
* success rateが改善するか
* 既存`pi05`より失敗が減るか・速くなるか（論文の主目的は速度改善）
* actionが不安定にならないか
* `‖a − ã‖` が過大にならないか（βの調整指標）

### 比較対象

```text
pi05
pi05 + supervised fine-tuning
pi05_rlt（rlt_enabled=false）
pi05_rlt with Stage 1 only（warmup運転）
pi05_rlt with Stage 1 + Stage 2
```

### 論文ablation対応の回帰試験（実装が「RLTになっているか」の検証）

```text
β=0（BC正則化なし）        → 論文では最大の性能低下。壊れなければ実装を疑う
reference dropoutなし       → copy-collapse（ãの複製）が起きるか確認
C=1（single-step）          → 学習が大きく劣化するはず
RL Tokenなし（ResNet等で代替）→ z_rlの寄与を確認
```

---

## dev_item

### dev_item 1: pi05_rlt policy追加

* `configuration_pi05_rlt.py` を作成（`@PreTrainedConfig.register_subclass("pi05_rlt")`、`rlt_enabled`/`β`/`C`/`σ`/dropout率等のconfig）
* `modeling_pi05_rlt.py` を作成（`PI05RLTPolicy`、`from_pretrained` の strict=False オーバーライド）
* `processor_pi05_rlt.py` を作成（`make_pi05_rlt_pre_post_processors`）
* `policies/__init__.py` へ import 追加し `policy.type=pi05_rlt` で読み込めるようにする
* `rlt_enabled=false` で既存`pi05`と同一出力を返す状態まで実装する

---

### dev_item 2: pi05 hidden state取得

* `sample_actions` / prefix prefill のオーバーライドで prefix最終層 token embeddings を捕捉する
* `extract_rl_token()` を実装
* 使用するhidden stateは**backbone最終層のtoken埋め込み**を既定とする（論文準拠）。他の候補（action expert側、複数layer）はablation項目として残す

---

### dev_item 3: RLT module実装

* `RLTokenEncoder`（軽量transformer + 学習可能`<rl>`トークン）
* `RLTokenDecoder`（自己回帰transformer、teacher forcing、線形出力射影）
* `RLTActor`（MLP、full chunk出力、reference条件付け、50% dropout対応）
* `TwinCritic`（MLP×2、chunk-level Q）

を実装する。Encoder/Decoderはtransformer必須。Actor/CriticはMLPでよい（論文もMLP）。

---

### dev_item 4: Stage 1 trainer作成

* `train_rlt_token.py` を作成
* datasetからobservationを読み込む
* frozen pi05からprefix最終層埋め込みを取得
* RLTokenEncoder/Decoderを自己回帰再構成（stop-gradターゲット）で学習
* SFT済みpi05 checkpointを入力の前提とする（α·L_vla併走は将来対応）
* checkpointを保存する

---

### dev_item 5: Actor統合とpass-through

* `rlt_enabled` フラグによるバイパス経路（既存pi05と同一出力の保証）
* warmupモード（ã をそのまま実行）
* actor経路（N(μ_θ, σ²I) サンプル、推論時は μ_θ）
* reference-action dropout（学習時50%、推論時は常時付与）
* **actor入出力・bufferの行動空間を正規化空間に統一**し、境界をテストする

---

### dev_item 6: Replay buffer実装

`src/lerobot/rl/buffer.py` の `ReplayBuffer` をベースに、**chunk単位遷移**を保存できるよう拡張する。

保存項目:

* x = (z_rl, proprio)（または再計算用のobservation）
* action chunk a_{1:C}
* reference chunk ã_{1:C}（**介入時は a^h に置換**）
* reward（chunk内の割引和計算に必要な粒度）
* next x
* done
* source タグ（warmup / RL / human）

stride-2 subsampling をロールアウト経路に**結線**する（rlt-openpiは未結線だった）。

---

### dev_item 7: Stage 2 online RL trainer作成

* `train_pi05_rlt_online.py` を作成（`RLAlgorithm` 継承のTD3系アルゴリズムを新設し `rl/algorithms/factory.py` に登録）
* 環境とのrollout（**探索はガウスサンプリング**。決定論で収集しない）
* warmupフェーズ（ã実行でbuffer事前充填）
* replay bufferへの保存（stride-2）
* critic update（twin-min、γ^C chunk-level TD）
* actor update（−Q + β‖a−ã‖²、50% dropout、critic 2回に1回）
* target network soft update（τ=0.005）
* UTD=5（環境遷移あたり）
* logging（reward、success、‖a−ã‖、Q値、loss）

---

### dev_item 8: LIBERO検証

* LIBERO環境で`pi05_rlt`を実行
* 既存`pi05`との比較
* Stage 1の学習確認
* Stage 2のオンラインRL確認
* success rate / reward / action stability / ‖a−ã‖ を記録する
* critical phase模擬（時刻/サブゴール切替）の検証

---

### dev_item 9: SO101展開

シミュレーションで動作確認後、SO101への展開を行う。

SO101展開時の確認項目:

* SO101のstate dimension（proprioに**速度**を含めるか検討。論文は位置+速度）
* SO101のaction dimension
* action normalization / unnormalization（正規化空間統一の確認）
* control frequency（論文は50Hz・C=10。SO101の周波数に合わせてCを再設計）
* action chunkの実行方法
* 安全停止処理・関節制限・action clipping（‖a−ã‖の上限クランプ等の安全弁はここで入れる）
* gripper制御
* カメラ入力
* 言語指示入力
* rollout保存形式
* reward設計（オペレータによるsparse binary。ボタン等のインターフェース）
* 人間介入時の処理（a と ã の両方を a^h に置換して保存）
* critical phase切替インターフェース（人間によるbase→RL切替）

SO101では、最初からオンラインRLを行わず、以下の順番で進める。

```text
1. pi05単体で推論
2. pi05_rltをrlt_enabled=falseで推論
3. RL Token抽出のみ実行
4. warmupモード（ã実行）で推論・buffer保存
5. 安全なタスクで短時間rollout（小さい固定σ）
6. offline update（bufferからcritic先行学習）
7. online RL（actor更新開始、βは大きめから徐々に調整）
```

---

## リスク

### 既存pi05の動作を壊すリスク

対策:

* `pi05`を直接変更しない
* `pi05_rlt`として新規policyを作る
* `rlt_enabled=false` で既存`pi05`と同一出力になるバイパスを持つ

---

### Actorが不安定なactionを出すリスク

対策:

* **BC正則化βを大きめから始める**（アンカーの主役はβ。ただしopenpi-RLTのEthernet設定のような実効β≈50の過剰アンカーはRLの改善余地を潰すので避ける）
* 固定σを小さくする（論文実装は0.002〜0.05）
* warmupで ã 実行から開始し、criticが立ち上がってからactorを更新する（critic 2 : actor 1）
* 実機ではaction clipping / ‖a−ã‖クランプを安全弁として入れる（学習則ではなく安全機構として）

---

### Actorが ã をコピーするだけになるリスク（copy-collapse）

対策:

* reference-action dropout（学習時50%）を必ず入れる
* dropout時に出力へ ã が混入しない構造にする（残差加算は禁止）
* ‖a−ã‖ とQ値をロギングし、criticが立ち上がった後に逸脱が増えることを確認する

---

### RL Tokenが有効な特徴にならないリスク

対策:

* Stage 1でreconstruction lossを確認する
* z_rl の次元・使用layerを比較する
* RL Tokenなし（ResNet等）のbaselineと比較する（論文ablationでは50%のthroughput低下）

---

### オンラインRLが難しいリスク

対策:

* まずはシミュレーションで検証する
* 報酬はsparse binaryに固定し、独自シェイピングを安易に足さない
* chunk-level TD（C=10）でhorizonを短縮する（single-stepにしない）
* warmup → critic先行 → actor更新の順で立ち上げる
* 実機では必ず安全制限を入れる

---

## 最初の到達目標

最初の到達目標は、RLTで性能を上げることではなく、以下を達成することである。

```text
policy.type=pi05_rlt
pretrained_path=既存pi05 checkpoint（strict=Falseで読み込み）
rlt_enabled=false で既存pi05と同一の推論ができる
内部でRL Token（prefix最終層埋め込み→z_rl）を抽出できる
```

この状態を作ることで、その後のStage 1学習、Actor統合、オンラインRLに進める。

---

## 最終目標

最終的には、`pi05`が出すreference actionをRLT actorが（入力条件として受け取り）洗練し、精密作業や接触作業において成功率・安定性・速度を改善することを目指す。

特に以下のようなタスクへの適用を想定する。

* 挿入作業
* 位置合わせ
* 把持後の微調整
* 接触を伴う操作
* 失敗後の再試行
* SO101での実ロボット操作

---
