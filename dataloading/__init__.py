"""
Temporary fix to use data generated with older pathnames.
The current datapreparation.kitti360pose module was previously named 
datapreparation.kitti360 which throws an error during loading via pickle.
For a clean solution, the data should be fully regenerated.
"""

import datapreparation.kitti360pose
import sys
sys.modules['datapreparation.kitti360'] = datapreparation.kitti360pose
