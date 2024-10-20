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

import numpy as np
from sionna.utils import expand_to_rank, matrix_inv, matrix_pinv
from sionna.mimo.utils import whiten_channel

import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.utils import expand_to_rank, matrix_sqrt_inv, flatten_last_dims, flatten_dims, split_dim, insert_dims, hard_decisions
from sionna.mapping import Constellation, SymbolLogits2LLRs, LLRs2SymbolLogits, PAM2QAM, Demapper, SymbolDemapper, SymbolInds2Bits, DemapperWithPrior, SymbolLogits2Moments
from sionna.mimo.utils import complex2real_channel, whiten_channel, List2LLR, List2LLRSimple, complex2real_matrix, complex2real_vector, real2complex_vector
from sionna.mimo.equalization import lmmse_equalizer, zf_equalizer, mf_equalizer



class BPICDetector_0(Layer):

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
                 bso_var=BSO_VAR_ACCUR,
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
        
        self._constellation = Constellation.create_or_check_constellation(
            constellation_type,
            num_bits_per_symbol,
            constellation,
            dtype=dtype)
       
        print(num_bits_per_symbol)


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


        self._demapper = Demapper(demapping_method,
                                  constellation=constellation,
                                  hard_out=hard_out,
                                  dtype=dtype)

        #    计算星座点的平均功率
        # Calculate average power of constellation points (Es)
        constellation_points = tf.convert_to_tensor(constellation.points, dtype=tf.complex64)
        energies = tf.math.abs(constellation_points) ** 2
        self.es = tf.complex(tf.reduce_mean(energies), tf.constant(0.0, dtype=tf.float32))




        # other configurations
        if bso_mean_init not in BPICDetector_0.BSO_MEAN_INIT_TYPES:
            raise Exception("1st iteration method in BSO to calculate the mean is not recognized.")
        else:
            self.bso_mean_init = bso_mean_init
        if bso_mean_cal not in BPICDetector_0.BSO_MEAN_CAL_TYPES:
            raise Exception("Other iteration method in BSO to calculate the mean is not recognized.")
        else:
            self.bso_mean_cal = bso_mean_cal
        if bso_var not in BPICDetector_0.BSO_VAR_TYPES:
            raise Exception("Not set use whether approximate or accurate variance in BSO.")
        else:
            self.bso_var = bso_var
        if bso_var_cal not in BPICDetector_0.BSO_VAR_CAL_TYPES:
            raise Exception("The method in BSO to calculate the variance is not recognized.")
        else:
            self.bso_var_cal = bso_var_cal
        if dsc_ise not in BPICDetector_0.DSC_ISE_TYPES:
            raise Exception("How to calculate the instantaneous square error is not recognized.")
        else:
            self.dsc_ise = dsc_ise
        if dsc_mean_prev_sour not in BPICDetector_0.DSC_MEAN_PREV_SOUR_TYPES:
            raise Exception("The source of previous mean in DSC is not recognized.")
        else:
            self.dsc_mean_prev_sour = dsc_mean_prev_sour
        if dsc_var_prev_sour not in BPICDetector_0.DSC_VAR_PREV_SOUR_TYPES:
            raise Exception("The source of previous variance in DSC is not recognized.")
        else:
            self.dsc_var_prev_sour = dsc_var_prev_sour
        self.min_var = min_var
        self.iter_num = num_iter
        if self.iter_num < 1:
            raise Exception("The iteration number must be positive.")
        self.iter_diff_min = iter_diff_min
        if detect_sour not in BPICDetector_0.DETECT_SOURS:
            raise Exception("The source of detection result is not recognized.")
        else:
            self.detect_sour = detect_sour
        if batch_size is not None:
            self.batch_size = batch_size


    def call(self, inputs):

        #y, h, prior, s = inputs
        y, h, no = inputs
       
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
       
        y, h = whiten_channel(y, h, no, return_s=False)  # pylint: disable=unbalanced-tuple-unpacking
        Hty = insert_dims(tf.linalg.matvec(h, y, adjoint_a=True),num_dims=1, axis=-1)
        Ht = tf.linalg.adjoint(h)
        HtH = tf.matmul(Ht, h)
        y = tf.expand_dims(y, axis = -1)
        Hty = tf.matmul(Ht, y)

        # 获取x_num, 即发射信号的维度
        x_num = tf.shape(h)[-1]

        # Compute a priori LLRs

        # llr_a is [..., K, num_bits_per_symbol]

        
        # Compute the inverse matrix
        mrc_mat = tf.linalg.diag(1 / tf.linalg.diag_part(HtH))  #hao 39
        
        # BSO - mean - 1st iteration
        #bso_zigma_1 = tf.eye(x_num, dtype=tf.complex64)
      
        identity_matrix = tf.eye(tf.shape(HtH)[-1], dtype = tf.complex64)
        shape_a = tf.shape(HtH)
        identity_matrix = tf.reshape(identity_matrix, [1] * (len(shape_a) - 2) + [shape_a[-2], shape_a[-1]]) # hao 36
        
        # Creating unit matrices and calculating non-diagonal parts
        HtH_off = tf.multiply(tf.subtract(tf.add(identity_matrix, 1), tf.multiply(identity_matrix, 2)), HtH)  # W * H_2  hao 37  hao 58

        # Square the non-diagonal portion
        HtH_off_sqr = tf.square(HtH_off) # np.square(W * H_2)  hao 58
        #HtH_off_sqr = tf.square(HtH)   # 10.2
        #No = tf.math.real(No[0, 0, 1, 1, 1, 1])
        No = no[0, 0, 1, 1, 1, 1]
        
        # BSO - mean - 1st iteratio
        if self.bso_mean_init == BPICDetector_0.BSO_MEAN_INIT_MMSE:
            bso_zigma_1 = tf.linalg.inv(HtH + No/self.es * identity_matrix)
        elif self.bso_mean_init == BPICDetector_0.BSO_MEAN_INIT_MRC:
            bso_zigma_1 = mrc_mat
        elif self.bso_mean_init == BPICDetector_0.BSO_MEAN_INIT_ZF:
            bso_zigma_1 = tf.linalg.inv(HtH)

        # BSO - variance
        bso_var_mat = tf.expand_dims(tf.math.reciprocal(tf.linalg.diag_part(HtH)), -1)   # 1/diag_part(HtH)
        if self.bso_var_cal == BPICDetector_0.BSO_VAR_CAL_MMSE:
            bso_var_mat = tf.expand_dims(tf.linalg.diag_part(tf.linalg.inv(HtH + No/self.es * identity_matrix), -1))
        elif self.bso_var_cal == BPICDetector_0.BSO_VAR_CAL_ZF:
            bso_var_mat = tf.expand_dims(tf.linalg.diag_part(tf.linalg.inv(HtH)), -1)
        bso_var_mat_sqr = tf.square(bso_var_mat)

        # DSC
        dsc_w = tf.eye(x_num, dtype=tf.complex64)
        if self.dsc_ise == BPICDetector_0.DSC_ISE_MRC:
            dsc_w = mrc_mat
        elif self.dsc_ise == BPICDetector_0.DSC_ISE_ZF:
            dsc_w = tf.linalg.inv(HtH)
        elif self.dsc_ise == BPICDetector_0.DSC_ISE_MMSE:
            dsc_w = tf.linalg.inv(HtH + No/self.es * identity_matrix)

        # Iterative detection initialization
        v_dsc = tf.zeros([1, 1], dtype=tf.float32)
        ise_dsc_prev_real = tf.zeros([1, 1], dtype=tf.float32)
        ise_dsc_prev_imag = tf.zeros([1, 1], dtype=tf.float32)
        ise_dsc_prev = tf.complex(ise_dsc_prev_real, ise_dsc_prev_imag)
        ise_dsc_prev_real = tf.ones([1, 1], dtype=tf.float32)
        ise_dsc_prev_imag = tf.ones([1, 1], dtype=tf.float32)
        x_bse_prev = tf.complex(ise_dsc_prev_real, ise_dsc_prev_imag)
        v_bse_prev = tf.complex(ise_dsc_prev_real, ise_dsc_prev_imag)
        v_dsc_prev = tf.complex(ise_dsc_prev_real, ise_dsc_prev_imag)
        v_dsc_prev = tf.zeros_like(bso_var_mat)

        # 1st iteration - use MMSE PIC detector
        # BSO
        # BSO - mean
        #x_bso = bso_zigma_1@(Hty - tf.matmul(HtH_off, tf.expand_dims(x_dsc, axis = -1)));  #给x_dsc添加一个维度，以进行乘法运算
        x_bso = bso_zigma_1@Hty   # hao 63  # 10.2
        
        
        # Compute G
        i = expand_to_rank(tf.eye(h.shape[-1], dtype=no.dtype), tf.rank(no), 0)
        g = tf.matmul(h, h, adjoint_a=True) + i
        g = tf.matmul(matrix_inv(g), h, adjoint_b=True)

        # Compute Gy
       
        gy = tf.squeeze(tf.matmul(g, y), axis=-1)
    
        # Compute GH
        gh = tf.matmul(g, h)
    
        # Compute diag(GH)
        d = tf.linalg.diag_part(gh)
    
        # Compute x_hat
        x_bso = tf.expand_dims(gy/d,axis=-1)
    
        # Compute residual error variance
        one = tf.cast(1, dtype=d.dtype)
        no_eff = tf.math.real(one/d - one)
        
        
        
        
        if self.bso_var == BPICDetector_0.BSO_VAR_APPRO:
            v_bso = No * bso_var_mat  # eqt (8)
        elif self.bso_var == BPICDetector_0.BSO_VAR_ACCUR:
            #v_bso = No * bso_var_mat
            v_bso = No * bso_var_mat + tf.matmul(HtH_off_sqr, v_dsc_prev) * bso_var_mat_sqr  # hao 59  bso_var_mat_sqr = square(1/diag_part(HtH))
        #real_part = tf.clip_by_value(tf.math.real(v_bso), clip_value_min=tf.math.real(self.min_var), clip_value_max=float('inf'))
        #imag_part = tf.clip_by_value(tf.math.imag(v_bso), clip_value_min=tf.math.imag(self.min_var), clip_value_max=float('inf'))
        # 重新组合实部和虚部回复数
        #v_bso = tf.complex(real_part, imag_part)  # v_bso = var_PIC
        
        # BSE - 使用高斯分布估计 P(x|y)
        #result_tensor = tf.expand_dims(tf.reduce_sum(x_bso - self._constellation.points[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis,:], axis=-1),axis = -1)   #hao 19
        result_tensor = abs(self._constellation.points[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis,:] - x_bso) # 10.4   # hao 19
        # 计算差的平方
        #squared_diff = tf.square(result_tensor)
        squared_diff = tf.cast(tf.square(result_tensor), dtype = tf.complex64)    # hao 20
        # 计算高斯函数的指数部分
        # xinwei 229
        pxy_pdf_exp_power = -1 / (2 * v_bso) * squared_diff   # hao 21
       
        # BSE - 让每一行最大功率为 0  # hao 22
        #pxypdf_exp_norm_power = pxy_pdf_exp_power - tf.expand_dims(tf.reduce_max(pxy_pdf_exp_power, axis=-1), axis=-1)
        # 提取原数组的实部和虚部
        real_part = tf.math.real(pxy_pdf_exp_power)
        imag_part = tf.math.imag(pxy_pdf_exp_power)
        # 计算每一行的最大值，并在相同的维度上扩展这个最大值
        max_real_indices = tf.argmax(real_part, axis=-1)
        max_complex_values = tf.gather(pxy_pdf_exp_power, max_real_indices, batch_dims=len(pxy_pdf_exp_power.shape) - 1)
        pxypdf_exp_norm_power = pxy_pdf_exp_power - tf.expand_dims(max_complex_values,axis=-1)
        
        pxy_pdf = tf.exp(pxypdf_exp_norm_power)   # hao 23
        
        # BSE - 计算每一个可能的 x 的系数，使得所有的和为1
        pxy_pdf_coeff = tf.expand_dims(1. / tf.reduce_sum(pxy_pdf, axis=-1), -1)  # hao 24   (np.expand_dims(np.sum(p_y_x, axis=2),2))
        #pxy_pdf_coeff = tf.tile(pxy_pdf_coeff, [1, self.constellation_len])
        # BSE - PDF 标准化
        pxy_pdf_norm = pxy_pdf_coeff * pxy_pdf    # hao 24  calculate_pyx
        
        # BSE - 计算均值和方差
        x_bse = tf.expand_dims(tf.reduce_sum(pxy_pdf_norm * self._constellation.points, axis=-1), -1)   # hao 9  hao 10

        #x_bse_mat = tf.tile(x_bse, [1, self.constellation_len])
        v_bse_sqr = tf.expand_dims(tf.square(abs(self._constellation.points[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis,:] - x_bse)),axis = -1)  # 10.4   # hao 11
        v_bse_sqr_c = tf.reduce_sum(tf.squeeze(tf.cast(v_bse_sqr, dtype = tf.complex64),axis=-1)*pxy_pdf_norm,axis=-1)
        v_bse = tf.expand_dims(v_bse_sqr_c,axis=-1)   # 10.2
        
        #real_part = tf.clip_by_value(tf.math.real(v_bse), clip_value_min=tf.math.real(self.min_var), clip_value_max=float('inf'))
        #imag_part = tf.clip_by_value(tf.math.imag(v_bse), clip_value_min=tf.math.imag(self.min_var), clip_value_max=float('inf'))
        # 重新组合实部和虚部回复数
        #v_bse = tf.complex(real_part, imag_part)   # hao 82
       
    
        # DSC
        # DSC - error
        ise_dsc = tf.square(dsc_w @ (Hty - HtH @ x_bse))

        # DSC - mean
        #if it == 0:
        x_dsc = x_bse
        
        # DSC - variance
        #if it == 0:
        v_dsc = v_bse
        

        # 更新统计信息
        # 更新统计信息 - BSE
        if self.dsc_mean_prev_sour == BPICDetector_0.DSC_MEAN_PREV_SOUR_BSE:
            x_bse_prev = x_bse
        if self.dsc_var_prev_sour == BPICDetector_0.DSC_VAR_PREV_SOUR_BSE:
            v_bse_prev = v_bse
        # 更新统计信息 - DSC
        v_dsc_prev = v_dsc
        # 更新统计信息 - DSC - 即时平方误差
        ise_dsc_prev = ise_dsc

        # 取检测值
        if self.detect_sour == BPICDetector_0.DETECT_SOUR_BSE:
            out = x_bse
        if self.detect_sour == BPICDetector_0.DETECT_SOUR_DSC:
            out = x_dsc
        out = tf.squeeze(out, -1)

        # Step 6: LLR demapping (extrinsic LLRs)
        # [..., K, num_bits_per_symbols]
        no = tf.math.real(No)
        z = self._demapper([out, no])

        num_streams = tf.shape(inputs[1])[-1]
        if self._output == 'bit':
            num_bits_per_symbol = self._constellation.num_bits_per_symbol
            z = split_dim(z, [num_streams, num_bits_per_symbol], tf.rank(z)-1)

      

        return z, v_dsc_prev, x_bse_prev, v_bse_prev, ise_dsc_prev, x_dsc