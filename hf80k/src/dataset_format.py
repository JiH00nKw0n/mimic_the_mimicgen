"""HDF5 시연 파일의 사원수 규약을 검사한다.

이 파일이 있는 이유는 하나다. Isaac Lab의 HDF5DatasetFileHandler는 파일 루트에
format_version 속성이 없으면 그 파일을 옛 형식으로 판단하고, 읽는 동안 root_pose의
사원수를 WXYZ 순서에서 XYZW 순서로 바꾼다. Isaac Lab 3.0이 기록하는 파일은 이미 XYZW
순서이므로, 속성이 빠진 파일을 읽으면 있지도 않은 변환이 한 번 더 걸린다.

이 파이프라인에서 그 결과는 다음과 같았다. 로봇 받침에 기록된 사원수는 (0, 0, 1, 0)이고
XYZW로 읽으면 z축 180도 회전이다. 같은 네 숫자를 WXYZ로 읽으면 y축 180도 회전이 된다.
받침 링크에 매달린 카메라 세 대가 전부 그만큼 돌아서 책상 밑을 보게 되고, 영상은 책상
아랫면으로 가득 찬다. 관절 각도와 물체 위치는 그대로라 재생 성공 판정은 통과하므로,
수율이나 성공률 검사로는 절대 잡히지 않는다.

속성이 빠지는 경로는 파일을 새로 만드는 자리다. h5py로 새 파일을 열면 루트 속성은
따라오지 않으므로, 조각을 묶는 병합 함수가 손으로 옮겨야 한다.
"""

from __future__ import annotations

import os

# Isaac Lab 3.0의 현재 형식 번호다. 이 값보다 작으면 옛 WXYZ 형식으로 취급된다.
CURRENT_FORMAT_VERSION = 1


def read_format_version(path: str) -> int:
    """파일 루트의 format_version을 읽는다. 없으면 0을 돌려준다.

    0은 Isaac Lab이 옛 WXYZ 형식으로 판단하는 값이며, 속성이 아예 없을 때와 같다.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        value = handle.attrs.get("format_version")
    return int(value) if value is not None else 0


def assert_modern_quaternion_format(path: str, stage: str, allow_legacy: bool = False) -> int:
    """옛 형식으로 읽히는 파일이면 그 자리에서 멈춘다.

    allow_legacy를 참으로 주면 검사를 건너뛴다. Isaac Lab 2.x가 실제로 기록한 파일을
    일부러 읽을 때만 쓴다. 이 파이프라인이 만드는 파일은 전부 XYZW이므로 기본값은 거짓이다.

    읽은 형식 번호를 돌려준다.
    """
    version = read_format_version(path)
    if allow_legacy or version >= CURRENT_FORMAT_VERSION:
        return version
    raise SystemExit(
        f"[{stage}] {os.path.basename(path)}의 루트에 format_version 속성이 없거나 값이 "
        f"{version}이다. Isaac Lab은 이 파일을 옛 WXYZ 사원수 형식으로 보고 root_pose를 "
        f"한 번 더 변환한다. 그러면 로봇 받침의 z축 180도 회전이 y축 180도 회전으로 바뀌고 "
        f"카메라 세 대가 전부 책상 밑을 향한다. 관절과 물체는 멀쩡해서 성공 판정은 통과하므로 "
        f"영상만 조용히 망가진다. 이 파일을 만든 병합 단계가 루트 속성을 옮기지 않은 것이니 "
        f"그쪽을 고쳐야 한다. 정말로 Isaac Lab 2.x가 기록한 옛 파일을 읽는 것이라면 "
        f"검사를 끄는 인자를 명시로 켜라.")
