#!/usr/bin/env python3
"""쿼터니언 순서가 XYZW라는 전제가 코드 곳곳에서 지켜지는지 확인한다.

왜 이 검사가 있는가. 이 컨테이너(Isaac Lab 3.0)는 자산의 ``.data.root_quat_w``와
``isaaclab.utils.math``의 회전 함수가 모두 XYZW 순서를 쓴다. 회전이 없으면 ``(0,0,0,1)``
이다. 한때 이것을 WXYZ로 잘못 보고 세 곳을 고친 적이 있고, 그 결과 핀이 x축으로 180도
돌아간 채 놓였다. MimicGen은 사람 시연의 핀 자세와 지금 핀 자세의 차이로 손 궤적을
옮기는데, 그 180도가 궤적 전체를 엉뚱한 곳으로 보냈다. 697번 시도 동안 로봇이 핀을 한 번도
건드리지 못했다. 같은 착오가 다시 들어오지 않게 여기서 막는다.

    python3 src/tests/test_quat_convention.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_PEG = os.path.join(HERE, "..", "env_peg")
sys.path.insert(0, ENV_PEG)

failures = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(("  통과  " if ok else "  실패  ") + label + ((" | " + detail) if detail else ""))
    if not ok:
        failures.append(label)


# 1) 수직도 식이 XYZW를 전제로 하는지 순수 계산으로 확인한다.
#    peg_mdp는 불러오는 순간 Isaac Lab을 요구하므로, 계산만 담은 peg_geom에서 가져온다.
#    같은 식을 검사에 다시 쓰지 않고 실제로 쓰이는 함수를 그대로 부른다.
try:
    import torch

    from peg_geom import upright_z_from_quat

    ident = torch.tensor([[0.0, 0.0, 0.0, 1.0]])          # XYZW 항등 회전. 똑바로 선 상태
    flip_x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])         # XYZW로 x축 180도. 거꾸로 선 상태
    yaw90 = torch.tensor([[0.0, 0.0, 0.7071068, 0.7071068]])   # z축 90도. 여전히 똑바로 섬
    check("XYZW 항등 회전의 수직도가 1이다",
          abs(float(upright_z_from_quat(ident)[0]) - 1.0) < 1e-5,
          "값 %.4f" % float(upright_z_from_quat(ident)[0]))
    check("x축 180도의 수직도가 -1이다",
          abs(float(upright_z_from_quat(flip_x)[0]) + 1.0) < 1e-5,
          "값 %.4f" % float(upright_z_from_quat(flip_x)[0]))
    check("z축 90도의 수직도가 1이다(요 회전은 수직도를 바꾸지 않는다)",
          abs(float(upright_z_from_quat(yaw90)[0]) - 1.0) < 1e-5,
          "값 %.4f" % float(upright_z_from_quat(yaw90)[0]))
except ImportError as exc:
    print("  건너뜀  수직도 계산 검사 (torch를 불러오지 못했다: %s)" % exc)

# 2) 핀을 놓을 때 쓰는 자세가 XYZW 항등 회전인지 원문에서 확인한다.
#    시뮬레이터 없이 확인할 수 있는 유일한 방법이라 원문을 읽는다.
src = open(os.path.join(ENV_PEG, "peg_mdp.py"), encoding="utf-8").read()
spawn_block = src[src.index("def randomize_peg_xy("):]
check("핀 스폰 자세가 quat[:, 3] = 1.0이다(XYZW 항등 회전)",
      re.search(r"quat\[:, 3\]\s*=\s*1\.0", spawn_block) is not None)
check("핀 스폰 자세에 quat[:, 0] = 1.0이 남아 있지 않다(WXYZ 착오의 흔적)",
      re.search(r"quat\[:, 0\]\s*=\s*1\.0", spawn_block) is None)

# 3) MimicGen에 넘기는 핀 자세에 순서 바꾸기가 끼어 있지 않은지 확인한다.
env_src = open(os.path.join(ENV_PEG, "peg_mimic_env.py"), encoding="utf-8").read()
poses_block = env_src[env_src.index("def get_object_poses("):]
poses_code = "\n".join(ln for ln in poses_block.splitlines() if not ln.strip().startswith("#"))
check("핀 자세를 root_quat_w 그대로 matrix_from_quat에 넘긴다",
      "PoseUtils.matrix_from_quat(peg.data.root_quat_w[env_ids])" in poses_code)
check("핀 자세에 WXYZ에서 XYZW로 자리를 옮기는 코드가 없다",
      "torch.cat([quat_wxyz[:, 1:], quat_wxyz[:, :1]]" not in poses_code)

# 4) 성공 판정 훅도 같은 순서를 쓰는지 확인한다.
hook_src = open(os.path.join(ENV_PEG, "peg_success_hook.py"), encoding="utf-8").read()
check("성공 판정 훅이 쿼터니언을 x, y, z, w 순으로 푼다",
      "x, y, z, w = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))" in hook_src)

print()
if failures:
    print("실패 %d건: %s" % (len(failures), ", ".join(failures)))
    sys.exit(1)
print("모두 통과했다.")
