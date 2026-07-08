set -e
cd ~/mimicgen_jihoonkwon/cpgen_stack
[ -d cpgen ] || git clone --depth 1 https://github.com/kevin-thankyou-lin/cpgen.git
[ -d cpgen-envs ] || git clone --depth 1 https://github.com/kevin-thankyou-lin/cpgen-envs.git
[ -d robosuite_scale ] || git clone --depth 1 --branch enable-scale-setting-arena https://github.com/ARISE-Initiative/robosuite.git robosuite_scale
echo "CLONES_DONE"
ls -d */ 2>/dev/null
