import sys
# 1) base_config: nerfstudio CONSOLE -> rich Console (nerfstudio is optional/heavy)
p = "cpgen/demo_aug/configs/base_config.py"
s = open(p).read()
if "from nerfstudio.utils.rich_utils import CONSOLE" in s:
    s = s.replace("from nerfstudio.utils.rich_utils import CONSOLE",
                  "from rich.console import Console as _RC\nCONSOLE = _RC()")
    open(p, "w").write(s); print("patched base_config (nerf->rich)")
# 2) generate.py: make curobo imports lazy so mink (CPU) path runs without curobo installed
p = "cpgen/demo_aug/generate.py"
s = open(p).read()
old = ('from demo_aug.envs.motion_planners.curobo_mp import (\n'
       '    CuroboMotionPlanner,  # Use lxml instead of xml.etree.ElementTree\n'
       ')\n'
       'from demo_aug.envs.motion_planners.eef_interp_curobo_mp import (\n'
       '    EEFInterpCuroboMotionPlanner,\n'
       ')')
new = ('try:\n'
       '    from demo_aug.envs.motion_planners.curobo_mp import CuroboMotionPlanner\n'
       '    from demo_aug.envs.motion_planners.eef_interp_curobo_mp import EEFInterpCuroboMotionPlanner\n'
       'except Exception as _e:\n'
       '    CuroboMotionPlanner = None\n'
       '    EEFInterpCuroboMotionPlanner = None')
if old in s:
    s = s.replace(old, new); open(p, "w").write(s); print("patched generate.py (lazy curobo)")
else:
    print("WARN: curobo import block not matched exactly", file=sys.stderr); sys.exit(2)
print("PATCH_OK")
