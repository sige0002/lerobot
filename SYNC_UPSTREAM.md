# Upstream同期手順

## リモート構成

| リモート名 | URL | 用途 |
|-----------|-----|------|
| origin | https://github.com/sige0002/lerobot.git | 自分のフォーク |
| upstream | https://github.com/huggingface/lerobot.git | 本家リポジトリ（`main` のみ取得する設定） |

> upstream は `main` だけを追跡する。設定:
> `git config remote.upstream.fetch "+refs/heads/main:refs/remotes/upstream/main"`

## ブランチ構成

| ブランチ | 用途 |
|---------|------|
| main | upstream/mainと同期するブランチ。直接の開発は行わない |
| develop | 開発用ブランチ。mainからの変更をマージして使う |

## upstream からの更新手順（GitHub Actions不要）

### 1. upstreamの最新を取得

```bash
cd lerobot
git fetch upstream
```

### 2. mainブランチを更新

```bash
git checkout main
git merge upstream/main --ff-only
```

> `--ff-only` で fast-forward のみ許可。mainに独自コミットがなければ常に成功する。

### 3. originに反映

```bash
git push origin main
```

### 4. developブランチにmainの変更を取り込む

```bash
git checkout develop
git merge main
```

> コンフリクトが発生した場合は手動で解決してコミットする。

### 5. developをoriginに反映

```bash
git push origin develop
```

## 一括実行スクリプト

```bash
cd lerobot
git fetch upstream \
  && git checkout main \
  && git merge upstream/main --ff-only \
  && git push origin main \
  && git checkout develop \
  && git merge main \
  && git push origin develop
```

## 注意事項

- mainブランチには直接コミットしない（upstream追従専用）
- 開発はすべてdevelopブランチ（またはdevelopから切ったfeatureブランチ）で行う
- upstream更新時にコンフリクトが起きるのはdevelopへのmerge時のみ

## DGX Spark (GB10) 環境メモ

| 項目 | 値 |
|------|-----|
| アーキテクチャ | aarch64 |
| GPU | NVIDIA GB10 (Blackwell) |
| ドライバ / CUDA | 580.95.05 / CUDA 13.0 |
| PyTorch | torch 2.11.0+cu128 / torchvision 0.26.0+cu128（動作確認済み） |

- lerobot 本体の `pyproject.toml` が既に torch/torchvision を **cu128** インデックス（`[[tool.uv.index]] pytorch-cu128`）に固定しているため、**DGX Spark 向けの独自 CUDA 指定は不要**。
- ドライバは CUDA 13.0 だが、PyTorch は **cu128 wheel** で前方互換的に動作する（本体の driver floor 570.86 < 580.95）。`torch.cuda.is_available() == True` / device = `NVIDIA GB10` で確認済み。
- ⚠️ `pyproject.toml` に cu130 等の `[tool.uv.sources]` を**追記しないこと**。本体に既存の `[tool.uv.sources]`（cu128）があるため二重定義になり、`uv sync`/`uv lock` が TOML パースエラー（`duplicate key`）で失敗する。CUDA 版を変えたい場合は既存の 367〜374 行のブロックを直接編集する。

### セットアップ

```bash
uv sync --locked                                # ベース依存（cu128 の torch/torchvision が入る）
uv sync --locked --extra pi --extra training    # pi0/pi0.5 学習環境
```

### 動作確認

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期待: 2.11.0+cu128 True NVIDIA GB10
```
