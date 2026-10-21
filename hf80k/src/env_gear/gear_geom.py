"""기어 조립 중 시뮬레이터가 없어도 돌릴 수 있는 계산만 모은 모듈.

여기에 따로 두는 이유는 검사 때문이다. gear_mdp는 불러오는 순간 Isaac Lab을 요구해서
노트북에서는 돌지 않는다. 쿼터니언 순서를 잘못 잡으면 생성이 통째로 무너지므로 그
부분만은 시뮬레이터 없이도 검사할 수 있어야 한다. 핀 꽂기의 peg_geom과 같은 구실이다.

쿼터니언 순서는 XYZW다. 이 컨테이너의 ``.data.root_quat_w``와 ``isaaclab.utils.math``의
회전 함수가 모두 XYZW를 쓴다. 두 가지로 확인했다. 첫째로 컨테이너 안에서
``quat_from_matrix(단위행렬)``이 ``(0,0,0,1)``을 돌려주었다. 둘째로 사람 시연 파일의
``initial_state``에 기록된 로봇 받침 자세가 ``(0,0,1,0)``인데, 받침이 180도 돌아가 있다는
사실과 맞으려면 XYZW여야 한다. 2026-10-21에 기어 시연의 명령을 이 순서로 다시 계산해
핀 꽂기의 기록된 명령과 최대 오차 1.8e-7로 재현되는 것도 확인했다.
"""
from __future__ import annotations

import torch

# 기어마다 축 구멍이 물체 원점에서 x축으로 얼마나 떨어져 있는지, 미터.
# 받침의 세 축도 같은 값만큼 떨어져 있어서, 기어가 제 축에 꽂히면 두 값이 겹친다.
# 시연을 만든 코드(multigear_cap_demos.py)의 AXIS_X를 그대로 가져왔다.
AXIS_X = {"large": -0.03025, "small": 0.05075, "medium": 0.02025}

# 축 맞음을 볼 높이, 미터. 시연을 만든 코드가 축 오차를 이 높이에서 재고 판정했다.
AXIS_Z = 0.005

# 조립 순서. 시연이 이 순서로만 만들어져 있다.
GEARS = ("large", "small", "medium")

# 성공 판정의 기준값 네 개. 시연을 만든 코드의 assembly_status를 그대로 가져왔다.
# 그 코드가 20편 60개 기어 전부를 이 기준으로 다시 확인했다고 적고 있다.
AXIS_XY_TOL_M = 0.001        # 축 중심이 받침 축에서 가로세로로 벗어난 거리
Z_TOL_M = 0.0025             # 기어 원점과 받침 원점의 높이 차이
TILT_TOL_RAD = 0.12          # 기어 몸통 z축이 월드 z축에서 기울어진 각
SPEED_TOL_M_S = 0.025        # 기어가 아직 움직이는 속도


def quat_apply_xyzw(quat, vec):
    """XYZW 순서 쿼터니언으로 벡터를 회전한다. quat (..., 4), vec (..., 3).

    라이브러리를 거치지 않고 직접 계산한다. 회전 규약을 한 군데서만 정하기 위해서다.
    식은 v' = v + 2w(q x v) + 2(q x (q x v))이고 q는 쿼터니언의 벡터 성분이다.
    """
    q = quat[..., :3]
    w = quat[..., 3:4]
    t = 2.0 * torch.cross(q, vec, dim=-1)
    return vec + w * t + torch.cross(q, t, dim=-1)


def upright_z_from_quat(quat) -> "torch.Tensor":
    """물체의 몸통 z축이 월드 z축과 얼마나 같은 방향인지를 -1에서 1 사이로 돌려준다.

    1이면 똑바로 서 있고 0이면 옆으로 누웠고 -1이면 거꾸로다. XYZW에서 회전행렬의
    (2,2) 성분이 ``1 - 2(x^2 + y^2)``이고 x가 0번, y가 1번 자리다.
    """
    return 1.0 - 2.0 * (quat[..., 0] ** 2 + quat[..., 1] ** 2)


def tilt_rad_from_quat(quat) -> "torch.Tensor":
    """몸통 z축이 월드 z축에서 기울어진 각을 라디안으로 돌려준다."""
    return torch.acos(torch.clamp(upright_z_from_quat(quat), -1.0, 1.0))


def axis_xy_error(gear_pos, gear_quat, base_pos, base_quat, gear_name: str):
    """기어의 축 중심과 받침의 해당 축 중심이 가로세로로 얼마나 벗어났는지, 미터.

    두 물체의 원점이 아니라 축 중심을 비교하는 이유는, 기어와 받침의 USD 원점이 축에서
    떨어져 있어서 원점끼리 비교하면 기어가 제 축 위에 있어도 오차가 크게 나오기 때문이다.
    기어가 축 주위로 돌아가 있어도 축 중심은 제자리이므로 돌아간 각과 무관하게 잰다.
    """
    offset = torch.zeros_like(gear_pos)
    offset[..., 0] = AXIS_X[gear_name]
    offset[..., 2] = AXIS_Z
    gear_axis = gear_pos + quat_apply_xyzw(gear_quat, offset)
    base_axis = base_pos + quat_apply_xyzw(base_quat, offset)
    return torch.linalg.vector_norm((gear_axis - base_axis)[..., :2], dim=-1)
