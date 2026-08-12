#!/usr/bin/env python3
"""격리 설치 디렉터리에서 Isaac이 이미 가진 큰 꾸러미를 지운다.

왜 필요한가. lerobot을 `pip install --target`으로 별도 디렉터리에 넣으면, pip은 그
디렉터리만 보고 의존성을 푼다. Isaac 인터프리터가 torch를 이미 들고 있다는 사실을
모르기 때문에 torch와 nvidia CUDA 꾸러미를 한 벌 더 받는다. 압축 기준 2.8 GB다.

어떻게 안전하게 지우는가. 후보를 지우기 전에 그 이름을 Isaac 인터프리터에서 실제로
불러올 수 있는지 확인하고, 확인된 것만 지운다. 확인할 때는 대상 디렉터리를 검색 경로에서
빼야 "Isaac이 가지고 있는가"를 묻는 것이 된다. 지운 뒤에는 Dockerfile의 다음 단계가
LeRobotDataset을 실제로 불러 보므로, 잘못 지웠다면 빌드가 거기서 실패한다.

numpy와 pyarrow는 건드리지 않는다. 두 꾸러미는 판본에 따라 이진 호환이 깨지는데,
아낄 수 있는 용량이 작아 위험에 비해 얻는 것이 없다.

    dedupe_site.py <설치_디렉터리>
"""
import importlib.util
import os
import shutil
import sys

# Isaac Sim이 확실히 들고 있고 용량이 큰 것들이다.
CANDIDATES = ["torch", "torchvision", "torchaudio", "triton", "nvidia", "cv2"]


def directory_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            if os.path.exists(full):
                total += os.path.getsize(full)
    return total


def main() -> int:
    if len(sys.argv) != 2:
        print("사용법: dedupe_site.py <설치_디렉터리>", file=sys.stderr)
        return 2
    target = sys.argv[1]
    if not os.path.isdir(target):
        print(f"[dedupe] 디렉터리가 없다: {target}", file=sys.stderr)
        return 2

    # 대상 디렉터리를 검색 경로에서 빼야 "Isaac에 있는가"를 묻는 것이 된다.
    real_target = os.path.realpath(target)
    sys.path = [p for p in sys.path if os.path.realpath(p) != real_target]

    freed = 0
    for name in CANDIDATES:
        here = os.path.join(target, name)
        if not os.path.exists(here):
            continue
        if importlib.util.find_spec(name) is None:
            print(f"[dedupe] 남긴다 {name}: Isaac 인터프리터에 없다")
            continue
        siblings = [
            os.path.join(target, entry)
            for entry in os.listdir(target)
            if entry.startswith(name + "-") and entry.endswith((".dist-info", ".libs"))
        ]
        for path in [here] + siblings:
            freed += directory_bytes(path)
            shutil.rmtree(path, ignore_errors=True)
        print(f"[dedupe] 지운다 {name}: Isaac 것을 쓴다")

    print(f"[dedupe] 되찾은 용량 {freed / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
