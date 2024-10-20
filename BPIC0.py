# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 01:28:18 2024

@author: Xuanzi
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:50:28 2024

@author: Xuanzi
"""
import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from sionna.mimo import StreamManagement
from sionna.utils import QAMSource, BinarySource, sim_ber, ebnodb2no, QAMSource, expand_to_rank
from sionna.mapping import Mapper, Constellation
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LinearDetector, KBestDetector, EPDetector, \
    RemoveNulledSubcarriers, MMSEPICDetector
from sionna.channel import GenerateOFDMChannel, OFDMChannel, RayleighBlockFading, gen_single_sector_topology
from sionna.channel.tr38901 import UMa, Antenna, PanelArray
from sionna.fec.ldpc import LDPC5GEncoder
from sionna.fec.ldpc import LDPC5GDecoder

import tensorflow as tf
from tensorflow.keras import Model
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.layers import Layer
from sionna.utils import flatten_dims, split_dim, flatten_last_dims, expand_to_rank
from sionna.ofdm import RemoveNulledSubcarriers
from sionna.mimo import MaximumLikelihoodDetectorWithPrior as MaximumLikelihoodDetectorWithPrior_
from sionna.mimo import MaximumLikelihoodDetector as MaximumLikelihoodDetector_
from sionna.mimo import LinearDetector as LinearDetector_
from sionna.mimo import KBestDetector as KBestDetector_
from sionna.mimo import EPDetector as EPDetector_
from sionna.mimo import MMSEPICDetector as MMSEPICDetector_
from sionna.mapping import Constellation
from detection_0 import OFDMDetector_0,OFDMDetectorWithPrior_0

from BPIC_0 import BPICDetector_0 

class BPICDetector0(OFDMDetectorWithPrior_0):


    def __init__(self,
                 output,
                 resource_grid,
                 stream_management,
                 demapping_method="maxlog",
                 num_iter=1,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=tf.complex64,
                 **kwargs):
        # constants
        # BSO

        BSO_MEAN_INIT_NO     = 0
        BSO_MEAN_INIT_MMSE   = 1
        BSO_MEAN_INIT_MRC    = 2  # in Alva's paper, he calls this matched filter
        BSO_MEAN_INIT_ZF     = 3  # x_hat=inv(H'*H)*H'*y (this requires y.len >= x.len)
        BSO_MEAN_INIT_TYPES  = [BSO_MEAN_INIT_NO, BSO_MEAN_INIT_MMSE, BSO_MEAN_INIT_MRC, BSO_MEAN_INIT_ZF]
        BSO_MEAN_CAL_MRC = 1
        BSO_MEAN_CAL_ZF = 2
        BSO_MEAN_CAL_TYPES = [BSO_MEAN_CAL_MRC, BSO_MEAN_CAL_ZF]
        BSO_VAR_APPRO   = 1  # use approximated variance
        BSO_VAR_ACCUR   = 2  # use accurate variance (will update in the iterations)
        BSO_VAR_TYPES   = [BSO_VAR_APPRO, BSO_VAR_ACCUR]
        BSO_VAR_CAL_MMSE   = 1  # use MMSE to estimate the variance
        BSO_VAR_CAL_MRC    = 2  # use the MRC to estimate the variance
        BSO_VAR_CAL_ZF     = 3  # use ZF to estimate the variance
        BSO_VAR_CAL_TYPES = [BSO_VAR_CAL_MMSE, BSO_VAR_CAL_MRC, BSO_VAR_CAL_ZF]
        # DSC
        # DSC - instantaneous square error
        DSC_ISE_NO      = 0  # use the error directly
        DSC_ISE_MRC     = 1  # in Alva's paper, he calls this matched filter
        DSC_ISE_ZF      = 2
        DSC_ISE_MMSE    = 3
        DSC_ISE_TYPES = [DSC_ISE_NO, DSC_ISE_MRC, DSC_ISE_ZF, DSC_ISE_MMSE]
        # DSC - mean previous source
        DSC_MEAN_PREV_SOUR_BSE = 1  # default in Alva's paper
        DSC_MEAN_PREV_SOUR_DSC = 2
        DSC_MEAN_PREV_SOUR_TYPES = [DSC_MEAN_PREV_SOUR_BSE, DSC_MEAN_PREV_SOUR_DSC]
        # DSC - variance previous source
        DSC_VAR_PREV_SOUR_BSE = 1  # default in Alva's paper
        DSC_VAR_PREV_SOUR_DSC = 2
        DSC_VAR_PREV_SOUR_TYPES = [DSC_VAR_PREV_SOUR_BSE, DSC_VAR_PREV_SOUR_DSC]
        # Detect
        DETECT_SOUR_BSE = 1
        DETECT_SOUR_DSC = 2
        DETECT_SOURS = [DETECT_SOUR_BSE, DETECT_SOUR_DSC]
        eps = tf.keras.backend.epsilon()

        
        # Instantiate the EP detector
        detector = BPICDetector_0(output=output,
                                    demapping_method=demapping_method,
                                    num_iter=num_iter,
                                    constellation_type=constellation_type,
                                    num_bits_per_symbol=num_bits_per_symbol,
                                    constellation=constellation,
                                    hard_out=hard_out,
                                    dtype=dtype,
                                    bso_mean_init=BSO_MEAN_INIT_MMSE,
                                    bso_mean_cal=BSO_MEAN_CAL_MRC,
                                    bso_var=BSO_VAR_ACCUR,
                                    bso_var_cal=BSO_VAR_CAL_MRC,
                                    dsc_ise=DSC_ISE_MRC,
                                    dsc_mean_prev_sour=DSC_MEAN_PREV_SOUR_BSE,
                                    dsc_var_prev_sour=DSC_VAR_PREV_SOUR_BSE,
                                    min_var=eps,
                                    iter_diff_min=eps,
                                    detect_sour=DETECT_SOUR_DSC,
                                    batch_size=None,
                                    **kwargs)

        super().__init__(detector=detector,
                         output=output,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         constellation_type=constellation_type,
                         num_bits_per_symbol=num_bits_per_symbol,
                         constellation=constellation,
                         dtype=dtype,
                         **kwargs)
        
