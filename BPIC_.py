# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 17:42:21 2024

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

import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.utils import expand_to_rank, matrix_sqrt_inv, flatten_last_dims, flatten_dims, split_dim, insert_dims, hard_decisions
from sionna.mapping import Constellation, SymbolLogits2LLRs, LLRs2SymbolLogits, PAM2QAM, Demapper, SymbolDemapper, SymbolInds2Bits, DemapperWithPrior, SymbolLogits2Moments
from sionna.mimo.utils import complex2real_channel, whiten_channel, List2LLR, List2LLRSimple, complex2real_matrix, complex2real_vector, real2complex_vector
from sionna.mimo.equalization import lmmse_equalizer, zf_equalizer, mf_equalizer

from sionna.ofdm import OFDMDetector,OFDMDetectorWithPrior

class BPICDetector_(Layer):

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
    # minimal value
    eps = tf.keras.backend.epsilon()
    # Batch size
    BATCH_SIZE_NO = None

    # variables
    _constellation = None
    constellation_len = 0
    es = 1
    bso_mean_init = None
    bso_mean_cal = None
    bso_var = None
    bso_var_cal = None
    dsc_ise = None
    dsc_mean_prev_sour = None
    dsc_var_prev_sour = None
    min_var = None
    num_iter = None  # maximal iteration
    iter_diff_min = None  # the minimal difference between 2 adjacent iterations
    detect_sour = None
    batch_size = BATCH_SIZE_NO  # the batch size

    def __init__(self,
                 output,
                 demapping_method="maxlog",
                 num_iter=1,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=tf.complex64,
                 bso_mean_init=BSO_MEAN_INIT_MMSE,
                 bso_mean_cal=BSO_MEAN_CAL_MRC,
                 bso_var=BSO_VAR_APPRO,
                 bso_var_cal=BSO_VAR_CAL_MRC,
                 dsc_ise=DSC_ISE_MRC,
                 dsc_mean_prev_sour=DSC_MEAN_PREV_SOUR_BSE,
                 dsc_var_prev_sour=DSC_VAR_PREV_SOUR_BSE,
                 min_var=eps,
                 iter_diff_min=eps,
                 detect_sour=DETECT_SOUR_DSC,
                 batch_size=None,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        
        assert isinstance(num_iter, int), "num_iter must be an integer"
        assert output in ("bit", "symbol"), "Unknown output"
        assert demapping_method in ("app", "maxlog"), "Unknown demapping method"

        assert dtype in [tf.complex64, tf.complex128], \
            "dtype must be tf.complex64 or tf.complex128"

        self._num_iter = num_iter
        self._output = output
        self._epsilon = 1e-4
        self._realdtype = dtype.real_dtype
        self._demapping_method = demapping_method
        self._hard_out = hard_out

        # Create constellation object
        num_bits_per_symbol=4
        self._constellation = Constellation.create_or_check_constellation(
            constellation_type,
            num_bits_per_symbol,
            constellation,
            dtype=dtype)


        # Soft symbol mapping

        self._llr_2_symbol_logits = LLRs2SymbolLogits(
                                        num_bits_per_symbol = num_bits_per_symbol,
                                        dtype=self._realdtype)

        if self._output == "symbol":
            self._llr_2_symbol_logits_output = LLRs2SymbolLogits(
                                    self._constellation.num_bits_per_symbol,
                                    dtype=self._realdtype,
                                    hard_out=hard_out)
            self._symbol_logits_2_llrs = SymbolLogits2LLRs(
                method=demapping_method,
                num_bits_per_symbol=self._constellation.num_bits_per_symbol)
        self._symbol_logits_2_moments = SymbolLogits2Moments(
                                            constellation=self._constellation,
                                            dtype=self._realdtype)


        # soft output demapping
        self._bit_demapper = DemapperWithPrior(
                                            demapping_method=demapping_method,
                                            constellation=self._constellation,
                                            dtype=dtype)




        #    计算星座点的平均功率
        # Calculate average power of constellation points (Es)

        constellation_points = tf.convert_to_tensor(constellation.points, dtype=tf.complex64)
        energies = tf.math.abs(constellation_points) ** 2
        self.es = tf.complex(tf.reduce_mean(energies), tf.constant(0.0, dtype=tf.float32))




        # other configurations
        if bso_mean_init not in BPICDetector_.BSO_MEAN_INIT_TYPES:
            raise Exception("1st iteration method in BSO to calculate the mean is not recognized.")
        else:
            self.bso_mean_init = bso_mean_init
        if bso_mean_cal not in BPICDetector_.BSO_MEAN_CAL_TYPES:
            raise Exception("Other iteration method in BSO to calculate the mean is not recognized.")
        else:
            self.bso_mean_cal = bso_mean_cal
        if bso_var not in BPICDetector_.BSO_VAR_TYPES:
            raise Exception("Not set use whether approximate or accurate variance in BSO.")
        else:
            self.bso_var = bso_var
        if bso_var_cal not in BPICDetector_.BSO_VAR_CAL_TYPES:
            raise Exception("The method in BSO to calculate the variance is not recognized.")
        else:
            self.bso_var_cal = bso_var_cal
        if dsc_ise not in BPICDetector_.DSC_ISE_TYPES:
            raise Exception("How to calculate the instantaneous square error is not recognized.")
        else:
            self.dsc_ise = dsc_ise
        if dsc_mean_prev_sour not in BPICDetector_.DSC_MEAN_PREV_SOUR_TYPES:
            raise Exception("The source of previous mean in DSC is not recognized.")
        else:
            self.dsc_mean_prev_sour = dsc_mean_prev_sour
        if dsc_var_prev_sour not in BPICDetector_.DSC_VAR_PREV_SOUR_TYPES:
            raise Exception("The source of previous variance in DSC is not recognized.")
        else:
            self.dsc_var_prev_sour = dsc_var_prev_sour
        self.min_var = min_var
        self.iter_num = num_iter
        if self.iter_num < 1:
            raise Exception("The iteration number must be positive.")
        self.iter_diff_min = iter_diff_min
        if detect_sour not in BPICDetector_.DETECT_SOURS:
            raise Exception("The source of detection result is not recognized.")
        else:
            self.detect_sour = detect_sour
        if batch_size is not None:
            self.batch_size = batch_size




    def call(self, inputs):

        #y, h, prior, s = inputs
        y, h, prior, No= inputs
        # y is unwhitened receive signal
        #   [..., M]
        # h the channel estimate
        #   [..., M, K]
        # prior is either the soft input LLRs
        #   [..., K, num_bits_per_symbol] or symbol logits [..., K, Q]
        # s the noise covariance matrix
        #   [..., M, M]

        ## Preprocessing
        # Whiten channel
        # y : [..., M]
        # s : [..., M, M]
        print(f"shape of h(after_preprocess): {tf.shape(h)}")
        print(f"shape of y(after_preprocess): {tf.shape(y)}")
        print(f"shape of No(after_preprocess): {tf.shape(No)}")
        y, h = whiten_channel(y, h, No, return_s=False)  # pylint: disable=unbalanced-tuple-unpacking
        print(f"shape of y(whiten): {tf.shape(y)}")
        print("shape of h(whiten):", h.shape)


        # matched filtering of y
        # [..., K, 1]
        Hty = insert_dims(tf.linalg.matvec(h, y, adjoint_a=True),
                            num_dims=1, axis=-1)
        print("shape of Hty(mf):", Hty.shape)


        

        # 获取x_num, 即发射信号的维度
        y_num = tf.shape(h)[-2]
        x_num = tf.shape(h)[-1]

        # Compute a priori LLRs
        if self._output == "symbol":
            llr_a = self._symbol_logits_2_llrs(prior)
        else:
            llr_a = prior
        # llr_a is [..., K, num_bits_per_symbol]
        llr_shape = tf.shape(llr_a)



        print(f"shape of llr_a: {tf.shape(llr_a)}")

        x_logits = self._llr_2_symbol_logits(llr_a)
        #print(f"shape of x_logits: {tf.shape(x_logits)}")
        x_dsc, var_x = self._symbol_logits_2_moments(x_logits)
        
        print(f"shape of x_dsc: {tf.shape(x_dsc)}")
        
        
        
        # Calculating the conjugate transpose
        Ht = tf.linalg.adjoint(h)
        # matrix multiplication
        HtH = tf.matmul(Ht, h)
        print("shape of HtH(mf):", HtH.shape)
        print("shape of Ht(mf):", Ht.shape)
        
        print("shape of No", No.shape)
        print("dtype of es", self.es.dtype)
        print("shape of x_eye",tf.eye(x_num, dtype=tf.complex64).shape)
        print("shape of No/es", (No/self.es).shape)
        
        # Creating unit matrices and calculating non-diagonal parts
        identity_matrix = tf.eye(x_num, dtype=tf.complex64)
        HtH_off = tf.multiply(tf.subtract(tf.add(identity_matrix, 1), tf.multiply(identity_matrix, 2)), HtH)
        # Square the non-diagonal portion
        HtH_off_sqr = tf.square(HtH_off)
        # Compute the inverse matrix
        mrc_mat = tf.linalg.diag(1 / tf.linalg.diag_part(HtH))
    
        # BSO - mean - 1st iteration
        bso_zigma_1 = tf.eye(x_num, dtype=tf.complex64)
        if self.bso_mean_init == BPICDetector_.BSO_MEAN_INIT_MMSE:
            bso_zigma_1 = tf.linalg.inv(HtH + No/self.es * tf.eye(x_num, dtype=tf.complex64))
        elif self.bso_mean_init == BPICDetector_.BSO_MEAN_INIT_MRC:
            bso_zigma_1 = mrc_mat
        elif self.bso_mean_init == BPICDetector_.BSO_MEAN_INIT_ZF:
            bso_zigma_1 = tf.linalg.inv(HtH)

        # BSO - mean - other iterations
        bso_zigma_others = mrc_mat
        if self.bso_mean_cal == BPICDetector_.BSO_MEAN_CAL_ZF:
            bso_zigma_others = tf.linalg.inv(HtH)

        # BSO - variance
        bso_var_mat = tf.expand_dims(tf.math.reciprocal(tf.linalg.diag_part(HtH)), -1)
        if self.bso_var_cal == BPICDetector_.BSO_VAR_CAL_MMSE:
            bso_var_mat = tf.expand_dims(tf.linalg.diag_part(tf.linalg.inv(HtH + No/self.es * tf.eye(x_num, dtype=tf.complex64))), -1)
        elif self.bso_var_cal == BPICDetector_.BSO_VAR_CAL_ZF:
            bso_var_mat = tf.expand_dims(tf.linalg.diag_part(tf.linalg.inv(HtH)), -1)
        bso_var_mat_sqr = tf.square(bso_var_mat)

        # DSC
        dsc_w = tf.eye(x_num, dtype=tf.complex64)
        if self.dsc_ise == BPICDetector_.DSC_ISE_MRC:
            dsc_w = mrc_mat
        elif self.dsc_ise == BPICDetector_.DSC_ISE_ZF:
            dsc_w = tf.linalg.inv(HtH)
        elif self.dsc_ise == BPICDetector_.DSC_ISE_MMSE:
            dsc_w = tf.linalg.inv(HtH + No/self.es * tf.eye(x_num, dtype=tf.complex64))


        # Iterative detection initialization
        #x_dsc = tf.zeros([x_num, 1], dtype=tf.float32)
        v_dsc = tf.zeros([x_num, 1], dtype=tf.float32)
        ise_dsc_prev = tf.zeros([x_num, 1], dtype=tf.float32)
        v_dsc_prev = None
        x_bse_prev = None
        v_bse_prev = None



        # 1st iteration - use MMSE PIC detector
        # BPIC takes in a priori LLRs


        # BSO
        # BSO - mean
        if it == 0:
            x_bso = bso_zigma_1@(Hty - tf.matmul(HtH_off, x_dsc));
        else:
            x_bso = bso_zigma_others@(Hty - tf.matmul(HtH_off, x_dsc));

        if self.bso_var == BPICDetector_.BSO_VAR_APPRO:
            v_bso = No * bso_var_mat
        elif self.bso_var == BPICDetector_.BSO_VAR_ACCUR:
            v_bso = No * bso_var_mat + tf.matmul(tf.matmul(HtH_off_sqr, v_dsc), bso_var_mat_sqr)

        vbso = tf.maximum(v_bso, self.min_var)  # 使用tf.maximum确保variance不低于最小值
        # BSE - 使用高斯分布估计 P(x|y)
        pxy_pdf_exp_power = -1 / (2 * v_bso) * tf.square(tf.abs(tf.tile(tf.expand_dims(x_bso, -1), [1, 1, self.constellation_len]) - tf.tile(self.constellation[None, None, :], [x_num, 1, 1])))
        # BSE - 让每一行最大功率为 0
        pxypdf_exp_norm_power = pxy_pdf_exp_power - tf.expand_dims(tf.reduce_max(pxy_pdf_exp_power, axis=-1), axis=-1)
        pxy_pdf = tf.exp(pxypdf_exp_norm_power)
        # BSE - 计算每一个可能的 x 的系数，使得所有的和为1
        pxy_pdf_coeff = tf.expand_dims(1. / tf.reduce_sum(pxy_pdf, axis=-1), -1)
        pxy_pdf_coeff = tf.tile(pxy_pdf_coeff, [1, self.constellation_len])
        # BSE - PDF 标准化
        pxy_pdf_norm = pxy_pdf_coeff * pxy_pdf
        # BSE - 计算均值和方差
        x_bse = tf.expand_dims(tf.reduce_sum(pxy_pdf_norm * self.constellation, axis=-1), -1)
        x_bse_mat = tf.tile(x_bse, [1, self.constellation_len])
        v_bse = tf.expand_dims(tf.reduce_sum(tf.square(tf.abs(x_bse_mat - self.constellation)) * pxy_pdf_norm, axis=-1), -1)
        v_bse = tf.maximum(v_bse, self.min_var)

        # DSC
        # DSC - error
        ise_dsc = tf.square(dsc_w @ (Hty - HtH @ x_bse))
        ies_dsc_sum = ise_dsc + ise_dsc_prev
        ies_dsc_sum = tf.maximum(ies_dsc_sum, self.min_var)
        # DSC - rho (if we use this rho, we will have a little difference)
        rho_dsc = ise_dsc_prev / ies_dsc_sum
        # DSC - mean
        if it == 0:
            x_dsc = x_bse
        else:
            if self.dsc_mean_prev_sour == BPICDetector.DSC_MEAN_PREV_SOUR_BSE:
                x_dsc = (1 - rho_dsc) * x_bse_prev + rho_dsc * x_bse
            if self.dsc_mean_prev_sour == BPICDetector.DSC_MEAN_PREV_SOUR_DSC:
                x_dsc = (1 - rho_dsc) * x_dsc + rho_dsc * x_bse
        # DSC - variance
        if it == 0:
            v_dsc = v_bse
        else:
            if self.dsc_var_prev_sour == BPICDetector.DSC_VAR_PREV_SOUR_BSE:
                v_dsc = (1 - rho_dsc) * v_bse_prev + rho_dsc * v_bse
            if self.dsc_var_prev_sour == BPICDetector.DSC_VAR_PREV_SOUR_DSC:
                v_dsc = (1 - rho_dsc) * v_dsc + rho_dsc * v_bse

        # 早停
        out = None
        # 更新统计信息
        # 更新统计信息 - BSE
        if self.dsc_mean_prev_sour == BPICDetector.DSC_MEAN_PREV_SOUR_BSE:
            x_bse_prev = x_bse
        if self.dsc_var_prev_sour == BPICDetector.DSC_VAR_PREV_SOUR_BSE:
            v_bse_prev = v_bse
        # 更新统计信息 - DSC
        v_dsc_prev = v_dsc
        # 更新统计信息 - DSC - 即时平方误差
        ise_dsc_prev = ise_dsc

        # 取检测值

        if self.detect_sour == BPICDetector.DETECT_SOUR_BSE:
            out = x_bse
        if self.detect_sour == BPICDetector.DETECT_SOUR_DSC:
            out = x_dsc
        out = tf.squeeze(out, -1)


        # Step 6: LLR demapping (extrinsic LLRs)
        # [..., K, num_bits_per_symbols]
        llr_d = tf.reshape(self._bit_demapper([out, llr_a, No]),
                                llr_shape)



        llr_e = llr_d - llr_a
        if self._output == "symbol":
            # convert back to symbols if requested.
             # output symbol logits computed on extrinsic LLRs
            out = self._llr_2_symbol_logits_output(llr_e)
        else:
            # output extrinsic LLRs
            out = llr_e
            if self._hard_out:
                out = hard_decisions(out)

        return out