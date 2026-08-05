# Dependencies and Offline Use

## Required runtime

- NVIDIA Isaac Sim / Isaac Lab compatible with the target repository
- Python packages used by the target task config
- LeRobot ≥ 0.4 only when the provided collection example is used
- PyYAML for package preflight

## USD dependency

`assets/fr3/fr3_research3.usda` and `fr3_research3_massfix.usda` reference NVIDIA's official Isaac 5.1 FR3 USD:

`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd`

The facelift meshes and composition layer are bundled, but this official base articulation is not duplicated. The target machine must have network access or a resolved local/cache copy. If converting to a local reference, preserve articulation, collision, joint, actuator, and mass properties; change only the asset reference path.

## Bundled visual resources

- FR3 facelift composition and meshes
- lab table wrapper and its source scene
- gray carpet texture
- three indoor HDRIs
- camera measured-v2 YAML and overlay handoff
- lab reference photos and simulator gallery

Before external redistribution, review the licenses for NVIDIA, PolyHaven, robot, and lab-owned image assets.

