# Vendored Wan 2.2 (in this repository)

This tree is **part of the Wan Service git repo**, not a separate clone to refresh on each GPU deploy.

## On RunPod / Linux GPU

```bash
cd /workspace/Wan   # or your clone path
git pull origin dev
test -f third_party/Wan2.2/generate.py && echo OK
```

Do **not** run `git clone https://github.com/Wan-Video/Wan2.2.git third_party/Wan2.2` if this folder already exists — that replaces **your vendored copy** (including local patches).

## Upstream reference

- https://github.com/Wan-Video/Wan2.2

When syncing upstream intentionally, merge or cherry-pick into this tree and commit in the **parent** repo.

## Patches in this copy (vs stock upstream)

- `generate.py`: CLI flag `--init_on_cpu` passed into `WanTI2V` (for `wan_service` GPU / memory modes).

Record upstream revision when you merge:

```bash
git log -1 --oneline -- third_party/Wan2.2
```
