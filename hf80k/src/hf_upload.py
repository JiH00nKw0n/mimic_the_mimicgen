#!/usr/bin/env python3
"""Push a finished LeRobot dataset directory to the Hugging Face Hub.

WHY a dedicated script instead of `huggingface-cli upload`: this run produces
160 chunk datasets over days on a spot instance, so the upload has to be
(a) restartable without duplicating anything, (b) tolerant of the transient
5xx/timeout the Hub throws under load, and (c) machine-readable, because the
chunk driver records `uploaded` in MANIFEST.json from its exit code.

Idempotency comes from the path prefix. Chunk N always lands under
`chunks/chunk_0000N/`, and `upload_folder` addresses files by repo path, so
re-uploading the same chunk overwrites the same paths and creates an empty (or
partial) commit instead of a second copy. Nothing is ever appended blindly.

At the end of the whole run `aggregate_and_upload()` merges every local chunk
with lerobot's own `aggregate_datasets` and uploads the merged dataset at the
repo root, so the repository is a single readable LeRobot v3 dataset with the
per-chunk originals kept beside it under `chunks/`.

HF_TOKEN and HF_REPO_ID come from the environment (INTERFACE.md section 1) and
the token is never printed -- it is scrubbed out of error messages too.

Progress goes to stderr; the LAST line of stdout is the JSON summary.

    python3 hf_upload.py --folder /work/chunks/chunk_00000/lerobot
    python3 hf_upload.py --mode aggregate --work-dir /work
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

# lerobot writes temporary per-frame PNGs under images/ while encoding; they are
# rebuilt from the mp4s and must never reach the Hub. The rest is junk.
DEFAULT_IGNORE = ("images/**", ".cache/**", "**/.DS_Store", "*.tmp", "*.lock")
SUMMARY_SCHEMA = "fr3_cube.hf80k.upload.v1"
# HTTP codes that will never succeed on retry (bad token, bad name, too large)
FATAL_STATUS = {400, 401, 403, 404, 413, 422}


def log(message: str) -> None:
    """Progress on stderr so stdout stays a parseable JSON document."""
    print(message, file=sys.stderr, flush=True)


def scrub(text: str, token: str) -> str:
    """Remove the token from anything we are about to print."""
    if not text:
        return ""
    if token:
        text = text.replace(token, "<HF_TOKEN>")
    return re.sub(r"hf_[A-Za-z0-9]{8,}", "<HF_TOKEN>", text)


def fail(message: str, code: int = 2, **extra):
    """Print the JSON summary of a failure and leave with a nonzero code."""
    summary = {"schema_version": SUMMARY_SCHEMA, "ok": False, "error": message}
    summary.update(extra)
    print(json.dumps(summary))
    sys.exit(code)


def env_settings(require_token: bool = True):
    """(token, repo_id, private) from the environment, failing fast and clearly."""
    token = os.environ.get("HF_TOKEN", "").strip()
    repo_id = os.environ.get("HF_REPO_ID", "").strip()
    missing = []
    if require_token and not token:
        missing.append("HF_TOKEN")
    if not repo_id:
        missing.append("HF_REPO_ID")
    if missing:
        fail(f"missing required environment variable(s): {', '.join(missing)}. "
             f"Set them in the .env that is passed to the container.")
    # 빈 문자열은 "설정하지 않음"으로 보고 기본값인 비공개를 쓴다. 여기에 ""를 거짓으로
    # 넣어 두면 HF_PRIVATE= 로 비워 둔 순간 8만 에피소드가 공개로 올라간다.
    private = os.environ.get("HF_PRIVATE", "1").strip().lower() not in (
        "0", "false", "no", "off")
    return token, repo_id, private


def import_hub():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        fail("huggingface_hub is not installed in this interpreter")
    # HfHubHTTPError는 쓰지 않는다. 판본에 따라 위치가 바뀌어서, 가져오려다 실패하면
    # "huggingface_hub가 없다"는 엉뚱한 메시지를 내며 업로드가 통째로 막힌다.
    return HfApi


def import_lerobot():
    """Import lerobot >= 0.4, honouring an out-of-tree site dir (see writer)."""
    import importlib.metadata

    site = os.environ.get("LEROBOT_SITE") or os.environ.get("UWLAB_LEROBOT_SITE", "")
    if site and os.path.isdir(site) and site not in sys.path:
        sys.path.insert(0, site)
    version = importlib.metadata.version("lerobot")
    from lerobot.datasets.aggregate import aggregate_datasets

    numbers = re.findall(r"\d+", version)
    if len(numbers) >= 2 and (int(numbers[0]), int(numbers[1])) < (0, 4):
        raise RuntimeError(f"aggregate_datasets needs lerobot>=0.4, found {version}")
    return version, aggregate_datasets


def is_transient(exc) -> bool:
    """Retry anything that is not an outright rejection by the Hub."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status is None or int(status) not in FATAL_STATUS


def with_retries(call, attempts: int, token: str, label: str, base_delay: float = 4.0):
    """Run `call`, retrying transient Hub failures with exponential backoff."""
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return call()
        except Exception as exc:
            message = scrub(f"{type(exc).__name__}: {exc}", token)
            if attempt >= attempts or not is_transient(exc):
                log(f"[upload] {label} failed permanently: {message}")
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, base_delay)
            log(f"[upload] {label} attempt {attempt}/{attempts} failed: {message}; "
                f"retrying in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError(f"{label}: retries exhausted")  # unreachable, keeps linters calm


def ensure_repo(api, repo_id: str, private: bool, token: str, attempts: int) -> str:
    """Create the dataset repo if it is not there yet. Existing repo = no-op."""
    url = with_retries(
        lambda: api.create_repo(repo_id=repo_id, repo_type="dataset",
                                private=private, exist_ok=True, token=token),
        attempts, token, f"create_repo {repo_id}")
    return str(url)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def count_files(path: Path) -> int:
    return sum(1 for item in Path(path).rglob("*") if item.is_file())


def upload_folder(api, folder: Path, repo_id: str, path_in_repo: str, token: str,
                  *, allow_patterns=None, ignore_patterns=DEFAULT_IGNORE,
                  attempts: int = 4, commit_message: str = "") -> dict:
    """upload_folder into a path prefix, retried. Same prefix = overwrite, not copy."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"not a directory: {folder}")
    if not (folder / "meta").is_dir():
        raise FileNotFoundError(f"{folder} has no meta/ -- not a LeRobot dataset root")
    message = commit_message or f"hf80k: {path_in_repo or 'merged dataset'}"
    log(f"[upload] {folder} -> {repo_id}:{path_in_repo or '/'} "
        f"({count_files(folder)} files, {directory_bytes(folder) / 1e6:.1f} MB)")
    info = with_retries(
        lambda: api.upload_folder(
            repo_id=repo_id, repo_type="dataset", folder_path=str(folder),
            path_in_repo=path_in_repo, allow_patterns=allow_patterns,
            ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
            commit_message=message, token=token),
        attempts, token, f"upload_folder {path_in_repo or '/'}")
    return {
        "folder": str(folder),
        "path_in_repo": path_in_repo,
        # counted on disk before ignore_patterns are applied, so these are an
        # upper bound on what the commit actually carries
        "local_files": count_files(folder),
        "local_bytes": directory_bytes(folder),
        "commit_url": scrub(str(getattr(info, "commit_url", "") or ""), token),
        "commit_oid": str(getattr(info, "oid", "") or ""),
    }


def shard_id() -> str:
    """이 컨테이너가 저장소 안에서 쓰는 자기 몫의 이름.

    왜 필요한가. 청크 번호는 컨테이너마다 0부터 다시 시작한다. GPU 4장에 컨테이너를
    네 개 띄우면 네 개가 모두 `chunks/chunk_00000`에 올리고, 먼저 올린 것을 나중에
    올린 것이 덮는다. 오류도 나지 않고 로그도 정상이며, 다 끝난 뒤 저장소에 8만 편이
    아니라 2만 편만 남는다. 8일을 쓰고 나서야 알게 되는 종류의 결함이다.

    그래서 컨테이너마다 다른 이름을 앞에 붙인다. 기본값은 `SEED_BASE`에서 만든다.
    컨테이너마다 SEED_BASE가 달라야 한다는 것은 이미 규칙이라(같으면 네 대가 같은
    데이터를 만든다) 그것을 그대로 쓰면 설정을 하나 더 늘리지 않고도 겹치지 않는다.
    사람이 읽기 좋은 이름을 원하면 `SHARD_ID=gpu0`처럼 직접 준다. Makefile의
    run-4gpu가 그렇게 한다.
    """
    explicit = os.environ.get("SHARD_ID", "").strip().strip("/")
    if explicit:
        return re.sub(r"[^A-Za-z0-9._-]", "_", explicit)
    seed = os.environ.get("SEED_BASE", "").strip()
    return f"seed{seed}" if seed else ""


def chunk_path_in_repo(chunk_name: str) -> str:
    """청크 하나가 저장소에서 차지할 경로. 몫 이름이 있으면 그 아래에 둔다."""
    shard = shard_id()
    return f"chunks/{shard}/{chunk_name}" if shard else f"chunks/{chunk_name}"


def default_path_in_repo(folder: Path) -> str:
    """/work/chunks/chunk_00007/lerobot -> chunks/<몫>/chunk_00007.

    The chunk directory name is the only thing that keeps two chunks from
    overwriting each other, so it is taken from the layout rather than invented.
    """
    folder = Path(folder).resolve()
    parent = folder.parent.name
    return chunk_path_in_repo(parent if parent.startswith("chunk_") else folder.name)


def work_dirs(spec: str) -> list[Path]:
    """작업 디렉터리 목록. 쉼표로 여러 개를 주거나 별표를 쓸 수 있다.

    GPU 4장으로 돌리면 한 대의 기계에 작업 디렉터리가 네 개 생긴다
    (/data/hf80k/gpu0부터 gpu3까지). 마지막 합치기는 네 개를 전부 읽어야 8만 편이
    나온다. 하나만 읽으면 2만 편짜리 데이터셋이 조용히 만들어진다.
    """
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    found: list[Path] = []
    for part in parts:
        matches = sorted(Path(p) for p in glob.glob(part)) if any(
            ch in part for ch in "*?[") else [Path(part)]
        for match in matches:
            if match.is_dir() and match.resolve() not in [f.resolve() for f in found]:
                found.append(match)
    return found


def chunk_dataset_dirs(work_dir) -> list[Path]:
    """끝난 청크 데이터셋 전부. 작업 디렉터리를 여러 개 줘도 된다."""
    roots: list[Path] = []
    for directory in work_dirs(work_dir) if isinstance(work_dir, str) else [Path(work_dir)]:
        roots.extend(sorted((Path(directory) / "chunks").glob("chunk_*/lerobot")))
    return [root for root in roots if (root / "meta" / "info.json").is_file()]


def read_total_episodes(root: Path) -> int:
    with open(root / "meta" / "info.json") as handle:
        return int(json.load(handle).get("total_episodes", 0))


def aggregate_and_upload(work_dir, repo_id: str, token: str, *, merged_dir=None,
                         private: bool = True, path_in_repo: str = "",
                         attempts: int = 4, upload: bool = True,
                         overwrite: bool = False) -> dict:
    """Merge every local chunk with lerobot's aggregate_datasets, then upload it.

    Run once at the end of the whole job. The merged dataset goes to the repo
    ROOT so that `LeRobotDataset(repo_id)` just works for the RL team, while the
    per-chunk copies stay under chunks/ as the raw record.

    video_files_size_in_mb=0.0 makes the aggregator copy and re-index each chunk
    video instead of concatenating into one growing mp4 -- the concatenating
    path is quadratic in the number of sources and 160 chunks would never
    finish (same reason the RL team's collector passes it).
    """
    directories = work_dirs(work_dir) if isinstance(work_dir, str) else [Path(work_dir)]
    if not directories:
        raise FileNotFoundError(f"작업 디렉터리를 찾지 못했다: {work_dir}")
    merged = (Path(merged_dir).resolve() if merged_dir
              else directories[0].resolve() / "merged")
    roots = chunk_dataset_dirs(work_dir)
    if not roots:
        raise FileNotFoundError(
            "청크 데이터셋이 없다. 찾아본 곳: "
            + ", ".join(f"{d}/chunks" for d in directories))
    log(f"[aggregate] 작업 디렉터리 {len(directories)}개에서 청크 {len(roots)}개를 찾았다: "
        + ", ".join(str(d) for d in directories))
    if merged.exists():
        # 비어 있어도 지운다. lerobot의 aggregate_datasets가 이 디렉터리를 자기가
        # 만들면서 이미 있으면 FileExistsError를 낸다. 오케스트레이터가 작업 디렉터리를
        # 만들 때 빈 merged/를 함께 만들어 두므로, "비었으면 그냥 둔다"로 두면 합치기가
        # 항상 그 오류로 죽는다.
        if any(merged.iterdir()) and not overwrite:
            raise FileExistsError(f"{merged} exists and is not empty; pass --overwrite")
        shutil.rmtree(merged)

    version, aggregate_datasets = import_lerobot()
    # the source repo ids are labels only (roots decide what is read), but they
    # must be unique. 청크 이름만 쓰면 작업 디렉터리를 여러 개 합칠 때 gpu0의
    # chunk_00000과 gpu1의 chunk_00000이 같은 이름이 되어 겹친다. 작업 디렉터리
    # 이름까지 넣어야 네 대분을 한 번에 합칠 수 있다.
    source_ids = [f"{repo_id}-{root.parents[2].name}-{root.parent.name}" for root in roots]
    if len(set(source_ids)) != len(source_ids):
        raise RuntimeError("청크 이름이 겹친다. 작업 디렉터리 이름이 서로 달라야 한다: "
                           + ", ".join(str(d) for d in directories))
    expected = sum(read_total_episodes(root) for root in roots)
    log(f"[aggregate] lerobot {version}: merging {len(roots)} chunks "
        f"({expected} episodes) into {merged}")
    aggregate_datasets(
        repo_ids=source_ids,
        aggr_repo_id=repo_id,
        roots=roots,
        aggr_root=merged,
        video_files_size_in_mb=0.0,
    )
    merged_total = read_total_episodes(merged)
    if merged_total != expected:
        raise RuntimeError(f"merged episode count mismatch: chunks hold {expected}, "
                           f"merged dataset reports {merged_total}")

    result = {
        "schema_version": SUMMARY_SCHEMA,
        "mode": "aggregate",
        "repo_id": repo_id,
        "lerobot_version": version,
        "chunks": len(roots),
        "chunk_roots": [str(root) for root in roots],
        "episodes": merged_total,
        "merged_path": str(merged),
        "bytes_on_disk": directory_bytes(merged),
        "uploaded": False,
        "ok": True,
    }
    if upload:
        HfApi = import_hub()
        api = HfApi()
        result["repo_url"] = scrub(
            ensure_repo(api, repo_id, private, token, attempts), token)
        result["upload"] = upload_folder(api, merged, repo_id, path_in_repo, token,
                                         attempts=attempts,
                                         commit_message="hf80k: merged dataset")
        result["uploaded"] = True
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("chunk", "aggregate"), default="chunk",
                        help="chunk: upload one lerobot/ directory; aggregate: "
                             "merge every local chunk and upload the result")
    # underscore spellings and --dataset/--chunk_index exist because
    # src/orchestrate.py calls this script with those names
    parser.add_argument("--folder", "--dataset", dest="folder", default="lerobot",
                        help="chunk mode: the LeRobot dataset directory to upload")
    parser.add_argument("--path-in-repo", "--path_in_repo", dest="path_in_repo",
                        default="",
                        help="chunk mode: default chunks/<chunk dir name>; "
                             "aggregate mode: default the repo root")
    parser.add_argument("--chunk-index", "--chunk_index", dest="chunk_index",
                        type=int, default=-1,
                        help="chunk mode: name the prefix chunks/chunk_<index> "
                             "instead of deriving it from the directory")
    parser.add_argument("--work-dir", "--work_dir", dest="work_dir",
                        default=os.environ.get("WORK_DIR", "/work"),
                        help="aggregate mode: chunks/와 merged/를 담은 디렉터리. "
                             "쉼표로 여러 개를 주거나 '/data/hf80k/gpu*'처럼 별표를 "
                             "쓸 수 있다. GPU 여러 장으로 돌렸으면 반드시 전부 준다")
    parser.add_argument("--merged", default="",
                        help="aggregate mode: default $WORK_DIR/merged")
    parser.add_argument("--repo-id", "--repo_id", dest="repo_id", default="",
                        help="default HF_REPO_ID from the environment")
    parser.add_argument("--private", default="", choices=("", "0", "1"),
                        help="override HF_PRIVATE for this call")
    parser.add_argument("--retries", type=int, default=4,
                        help="attempts per Hub call, backoff 4s doubling")
    parser.add_argument("--allow-patterns", default="",
                        help="comma-separated globs; empty uploads everything")
    parser.add_argument("--ignore-patterns", default=",".join(DEFAULT_IGNORE))
    parser.add_argument("--commit-message", default="")
    parser.add_argument("--no-upload", action="store_true",
                        help="aggregate mode: merge locally and stop")
    parser.add_argument("--overwrite", action="store_true",
                        help="aggregate mode: replace a non-empty merged directory")
    return parser.parse_args()


def main():
    args = parse_args()
    needs_token = not (args.mode == "aggregate" and args.no_upload)
    token, env_repo_id, private = env_settings(require_token=needs_token)
    repo_id = args.repo_id.strip() or env_repo_id
    if args.private:
        private = args.private == "1"
    attempts = max(3, args.retries)  # the spec floor is three attempts
    allow = [p.strip() for p in args.allow_patterns.split(",") if p.strip()] or None
    ignore = [p.strip() for p in args.ignore_patterns.split(",") if p.strip()]

    if args.mode == "aggregate":
        try:
            summary = aggregate_and_upload(
                args.work_dir, repo_id, token, merged_dir=args.merged or None,
                private=private, path_in_repo=args.path_in_repo, attempts=attempts,
                upload=not args.no_upload, overwrite=args.overwrite)
        except Exception as exc:
            fail(scrub(f"{type(exc).__name__}: {exc}", token), code=1,
                 mode="aggregate", repo_id=repo_id)
        print(json.dumps(summary))
        return 0

    folder = Path(args.folder).resolve()
    path_in_repo = args.path_in_repo
    if not path_in_repo and args.chunk_index >= 0:
        path_in_repo = chunk_path_in_repo(f"chunk_{args.chunk_index:05d}")
    path_in_repo = path_in_repo or default_path_in_repo(folder)
    HfApi = import_hub()
    api = HfApi()
    try:
        repo_url = ensure_repo(api, repo_id, private, token, attempts)
        uploaded = upload_folder(api, folder, repo_id, path_in_repo, token,
                                 allow_patterns=allow, ignore_patterns=ignore,
                                 attempts=attempts,
                                 commit_message=args.commit_message)
    except Exception as exc:
        fail(scrub(f"{type(exc).__name__}: {exc}", token), code=1,
             mode="chunk", repo_id=repo_id, folder=str(folder),
             path_in_repo=path_in_repo)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "mode": "chunk",
        "repo_id": repo_id,
        "repo_url": scrub(repo_url, token),
        "private": private,
        "ok": True,
    }
    summary.update(uploaded)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
