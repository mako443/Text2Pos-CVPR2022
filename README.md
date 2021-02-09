TODO:
- Re-create one scene on slurm
- Write Dataset and Config for RefNet, overfit one scene

Pipeline:
- Data preparation separate, not in loader
- Start w/ Semantic3D, switch to ScanNet if any problems (rendering inconsistent, too many objects, too hard, ...)
- Semanict3D imports only in datapreparation

Assumptions:
- Network seeing 150 objects and 10 descriptions (possibly as one sentence) is feasible