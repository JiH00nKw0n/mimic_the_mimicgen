#!/usr/bin/env python3
"""병합한 HDF5가 사원수 규약을 잃지 않는지 확인한다.

왜 필요한가. Isaac Lab의 데이터 적재기는 파일 루트에 format_version 속성이 없으면 그
파일을 옛 형식으로 판단하고, 읽는 동안 root_pose의 사원수를 WXYZ 순서에서 XYZW 순서로
바꾼다. 우리가 기록하는 파일은 이미 XYZW다. 그래서 속성이 빠지면 있지도 않은 변환이
한 번 더 걸린다.

실제로 한 번 그렇게 됐다. 로봇 받침에 기록된 사원수 (0, 0, 1, 0)은 XYZW로 읽으면 z축
180도 회전이고, 같은 네 숫자를 WXYZ로 읽으면 y축 180도 회전이 된다. 받침 링크에 매달린
카메라 세 대가 전부 그만큼 돌아 책상 밑을 보게 되어, 업로드한 44편 가운데 42편의 영상이
책상 아랫면으로 가득 찼다. 관절 각도와 물체 위치는 그대로였고 재생 성공 판정도 통과해서,
수율로도 성공률로도 잡히지 않았다.

속성이 빠진 자리는 조각을 묶어 새 파일을 만드는 병합 함수 두 개였다. h5py로 새 파일을
열면 루트 속성이 따라오지 않기 때문이다. 여기서 막는다.

    python3 src/tests/test_dataset_format.py
"""
import os
import sys
import tempfile

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, SRC)

import dataset_format                                        # noqa: E402
from orchestrate import merge_hdf5_shards, merge_sart_into_gen  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  통과  {name}")
    else:
        print(f"  실패  {name}  {detail}")
        FAILURES.append(name)


def write_shard(path, names, samples=5, success=True, stamp_version=True):
    """Isaac Lab 3.0이 기록한 것과 같은 모양의 작은 시연 파일을 만든다."""
    with h5py.File(path, "w") as handle:
        if stamp_version:
            handle.attrs["format_version"] = dataset_format.CURRENT_FORMAT_VERSION
        data = handle.create_group("data")
        data.attrs["env_args"] = '{"env_name": "test", "type": 2}'
        data.attrs["total"] = str(samples * len(names))
        for name in names:
            demo = data.create_group(name)
            demo.attrs["num_samples"] = samples
            demo.attrs["success"] = True if success else False
            demo.create_dataset("actions", data=np.zeros((samples, 7), dtype=np.float32))
            root = demo.create_group("states/articulation/robot")
            # 받침의 z축 180도 회전을 XYZW 순서로 적는다. 실제 기록 파일과 같은 값이다.
            pose = np.tile(np.array([0.72, 0.138, 0.722, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
                           (samples, 1))
            root.create_dataset("root_pose", data=pose)


def main():
    print(__doc__.splitlines()[0])
    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "shard_a.hdf5")
        b = os.path.join(tmp, "shard_b.hdf5")
        write_shard(a, ["demo_0", "demo_1"])
        write_shard(b, ["demo_0"])

        print("\n[1] 속성을 읽는 함수")
        check("적힌 파일에서 1을 읽는다", dataset_format.read_format_version(a) == 1)
        bare = os.path.join(tmp, "bare.hdf5")
        write_shard(bare, ["demo_0"], stamp_version=False)
        check("속성이 없으면 0을 돌려준다", dataset_format.read_format_version(bare) == 0)

        print("\n[2] 검사 함수")
        try:
            dataset_format.assert_modern_quaternion_format(a, "test")
            ok = True
        except SystemExit:
            ok = False
        check("적힌 파일은 통과시킨다", ok)
        try:
            dataset_format.assert_modern_quaternion_format(bare, "test")
            raised = False
        except SystemExit:
            raised = True
        check("속성이 없는 파일에서 멈춘다", raised)
        try:
            dataset_format.assert_modern_quaternion_format(bare, "test", allow_legacy=True)
            allowed = True
        except SystemExit:
            allowed = False
        check("옛 형식을 일부러 허용하면 통과시킨다", allowed)

        print("\n[3] 생성 조각 병합")
        merged = os.path.join(tmp, "gen.hdf5")
        count = merge_hdf5_shards([a, b], merged, renumber=True, log=lambda *_: None)
        check("세 편이 묶인다", count == 3, f"실제 {count}")
        check("병합본이 format_version을 가진다",
              dataset_format.read_format_version(merged) == 1,
              f"실제 {dataset_format.read_format_version(merged)}")

        print("\n[4] 조각이 하나일 때")
        single_src = os.path.join(tmp, "single.hdf5")
        single_out = os.path.join(tmp, "gen_single.hdf5")
        write_shard(single_src, ["demo_0"])
        merge_hdf5_shards([single_src], single_out, renumber=True, log=lambda *_: None)
        check("이름만 바꾸는 경로에서도 속성이 남는다",
              dataset_format.read_format_version(single_out) == 1)

        print("\n[5] SART 병합")
        gen = os.path.join(tmp, "gen2.hdf5")
        write_shard(gen, ["demo_0", "demo_1"])
        sart = os.path.join(tmp, "sart_00.hdf5")
        write_shard(sart, ["demo_0"], samples=6)
        out = os.path.join(tmp, "gen_sart.hdf5")
        total, accepted = merge_sart_into_gen(gen, [sart], out, log=lambda *_: None)
        check("생성 2편과 증강 1편이 묶인다", (total, accepted) == (3, 1), f"실제 {(total, accepted)}")
        check("SART 병합본이 format_version을 가진다",
              dataset_format.read_format_version(out) == 1,
              f"실제 {dataset_format.read_format_version(out)}")

        print("\n[6] 렌더 결과처럼 표시가 없는 것이 정상인 파일")
        # 렌더 산출물은 카메라 영상이고 기록 단계가 h5py로 직접 읽는다. 로봇 자세를
        # 해석하지 않으므로 표시가 없는 것이 정상이며, 여기서 멈추면 파이프라인이
        # 렌더 다음 단계로 가지 못한다. 실제로 한 번 그렇게 다섯 청크가 전부 멈췄다.
        rgb_a = os.path.join(tmp, "rgb.part00.hdf5")
        rgb_b = os.path.join(tmp, "rgb.part01.hdf5")
        write_shard(rgb_a, ["demo_0", "demo_1"], stamp_version=False)
        write_shard(rgb_b, ["demo_2"], stamp_version=False)
        rgb_out = os.path.join(tmp, "rgb.hdf5")
        try:
            merged_n = merge_hdf5_shards([rgb_a, rgb_b], rgb_out, renumber=False,
                                         log=lambda *_: None,
                                         require_format_version=False)
            ok = merged_n == 3
            detail = f"실제 {merged_n}편"
        except Exception as exc:                                  # noqa: BLE001
            ok, detail = False, repr(exc)
        check("표시를 요구하지 않으면 렌더 조각도 묶인다", ok, detail)

        print("\n[7] 속성이 없는 조각은 생성 병합을 멈춘다")
        halted_out = os.path.join(tmp, "x.hdf5")
        try:
            merge_hdf5_shards([bare, b], halted_out,
                              renumber=True, log=lambda *_: None)
            stopped = False
        except Exception:
            stopped = True
        check("속성 없는 조각에서 멈춘다", stopped)
        # 멈춘 자리에 반쯤 쓰인 파일이 남으면 다음 실행이 그것을 완성본으로 오인할 수 있다.
        leftovers = [f for f in os.listdir(tmp) if f.startswith("x.hdf5")]
        check("멈췄을 때 산출물을 남기지 않는다", leftovers == [], f"남은 것 {leftovers}")

    print()
    if FAILURES:
        print(f"실패 {len(FAILURES)}건: {', '.join(FAILURES)}")
        return 1
    print("전부 통과했다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
