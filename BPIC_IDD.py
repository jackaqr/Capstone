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

from BPIC import BPICDetector 



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



SIMPLE_SIM = True   # reduced simulation time for simple simulation if set to True
if SIMPLE_SIM:
    batch_size = int(1e1)  # number of OFDM frames to be analyzed per batch
    batch_size = int(2)  # number of OFDM frames to be analyzed per batch
    num_iter = 5  # number of Monte Carlo Iterations (total number of Monte Carlo trials is num_iter*batch_size)
    num_steps = 6
    tf.config.run_functions_eagerly(True)   # run eagerly for better debugging
else:
    batch_size = int(64)  # number of OFDM frames to be analyzed per batch
    num_iter = 128  # number of Monte Carlo Iterations (total number of Monte Carlo trials is num_iter*batch_size)
    num_steps = 11


ebno_db_min_perf_csi = -10  # min EbNo value in dB for perfect csi benchmarks
ebno_db_max_perf_csi = 0
ebno_db_min_cest = -10
ebno_db_max_cest = 10


SIMPLE_DATA = False
if SIMPLE_DATA:
    NUM_OFDM_SYMBOLS = 4
    #FFT_SIZE = 12*4 # 4 PRBs
    FFT_SIZE = 12 # 4 PRBs
    SUBCARRIER_SPACING = 30e3 # Hz
    CARRIER_FREQUENCY = 3.5e9 # Hz
    SPEED = 3. # m/s
    num_bits_per_symbol = 4 # 16 QAM
    #n_ue = 4 # 4 UEs
    n_ue = 1 # 4 UEs
    NUM_RX_ANT = 4 # 16 BS antennas
    num_pilot_symbols = 2
else:
    NUM_OFDM_SYMBOLS = 14
    FFT_SIZE = 12*4 # 4 PRBs
    SUBCARRIER_SPACING = 30e3 # Hz
    CARRIER_FREQUENCY = 3.5e9 # Hz
    SPEED = 3. # m/s
    num_bits_per_symbol = 4 # 16 QAM
    n_ue = 4 # 4 UEs
    NUM_RX_ANT = 16 # 16 BS antennas
    num_pilot_symbols = 2


# The user terminals (UTs) are equipped with a single antenna
# with vertial polarization.
UT_ANTENNA = Antenna(polarization='single',
                     polarization_type='V',
                     antenna_pattern='omni', # Omnidirectional antenna pattern
                     carrier_frequency=CARRIER_FREQUENCY)

# The base station is equipped with an antenna
# array of 8 cross-polarized antennas,
# resulting in a total of 16 antenna elements.
BS_ARRAY = PanelArray(num_rows_per_panel=2,
                      num_cols_per_panel=4,
                      polarization='dual',
                      polarization_type='cross',
                      antenna_pattern='38.901', # 3GPP 38.901 antenna pattern
                      carrier_frequency=CARRIER_FREQUENCY)

# 3GPP UMa channel model is considered
channel_model_uma = UMa(carrier_frequency=CARRIER_FREQUENCY,
                    o2i_model='low',
                    ut_array=UT_ANTENNA,
                    bs_array=BS_ARRAY,
                    direction='uplink',
                    enable_shadow_fading=False,
                    enable_pathloss=False)

channel_model_rayleigh = RayleighBlockFading(num_rx=1, num_rx_ant=NUM_RX_ANT, num_tx=n_ue, num_tx_ant=1)

constellation = Constellation("qam", num_bits_per_symbol=num_bits_per_symbol)

rx_tx_association = np.ones([1, n_ue])
sm = StreamManagement(rx_tx_association, 1)

# Parameterize the OFDM channel
rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS, pilot_ofdm_symbol_indices = [2, 3],
                  fft_size=FFT_SIZE, num_tx=n_ue,
                  pilot_pattern = "kronecker",
                  subcarrier_spacing=SUBCARRIER_SPACING)

rg.show()
plt.show()

# Parameterize the LDPC code
R = 0.5  # rate 1/2
N = int(FFT_SIZE * (NUM_OFDM_SYMBOLS - 2) * num_bits_per_symbol)
# N = int((FFT_SIZE) * (NUM_OFDM_SYMBOLS - 2) * num_bits_per_symbol)
# code length; - 12 because of 11 guard carriers and 1 DC carrier, - 2 becaues of 2 pilot symbols
K = int(N * R)  # number of information bits per codeword





class IddModel(Model):
    def __init__(self, num_idd_iter=3, num_bp_iter_per_idd_iter=12, detector='lmmse', _siso_detector = '_', cest_type="LS", interp="lin", perfect_csi_rayleigh=False):
        super().__init__()
        ######################################
        ## Transmitter
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(K, N, num_bits_per_symbol=num_bits_per_symbol)
        self._mapper = Mapper(constellation=constellation)
        self._rg_mapper = ResourceGridMapper(rg)

        # Channel
        if perfect_csi_rayleigh:
            self._channel_model = channel_model_rayleigh
        else:
            self._channel_model = channel_model_uma

        self._channel = OFDMChannel(channel_model=self._channel_model,
                                    resource_grid=rg,
                                    add_awgn=True, normalize_channel=True, return_channel=True)

        # Receiver
        self._cest_type = cest_type
        self._interp = interp

        # Channel estimation
        self._perfect_csi_rayleigh = perfect_csi_rayleigh
        if self._perfect_csi_rayleigh:
            self._removeNulledSc = RemoveNulledSubcarriers(rg)
        elif cest_type == "LS":
            self._ls_est = LSChannelEstimator(rg, interpolation_type=interp)
        else:
            raise NotImplementedError('Not implemented:' + cest_type)

        # Detection
        if detector == "lmmse":
            self._detector = LinearDetector("lmmse", 'bit', "maxlog", rg, sm, constellation_type="qam",
                                            num_bits_per_symbol=num_bits_per_symbol, hard_out=False)
        elif detector == "k-best":
            k = 64
            self._detector = KBestDetector('bit', n_ue, k, rg, sm, constellation_type="qam",
                                           num_bits_per_symbol=num_bits_per_symbol, hard_out=False)
        elif detector == "ep":
            l = 10
            self._detector = EPDetector('bit', rg, sm, num_bits_per_symbol, l=l, hard_out=False)

        # first IDD detector is LMMSE as MMSE-PIC with zero-prior bils down to soft-output LMMSE
        self._num_idd_iter = num_idd_iter
        
        if _siso_detector == 'MMSEPIC':
            self._siso_detector = MMSEPICDetector(output="bit", resource_grid=rg, stream_management=sm,
                                                  demapping_method='maxlog', constellation=constellation, num_iter=1,
                                                  hard_out=False)
        elif _siso_detector == 'BPIC':
            self._siso_detector = BPICDetector(output="bit", resource_grid=rg, stream_management=sm,
                                                  demapping_method='maxlog', constellation=constellation, num_iter=1,
                                                  hard_out=False)
            
        self._siso_decoder = LDPC5GDecoder(self._encoder, return_infobits=False,
                                           num_iter=num_bp_iter_per_idd_iter, stateful=True, hard_out=False, cn_type='minsum')
        self._decoder = LDPC5GDecoder(self._encoder, return_infobits=True, stateful=True, hard_out=True, num_iter=num_bp_iter_per_idd_iter, cn_type='minsum')
        # last decoder must also be statefull


    def new_topology(self, batch_size):
        """Set new topology"""
        if isinstance(self._channel_model, UMa):
            # sensible values according to 3GPP standard, no mobility by default
            topology = gen_single_sector_topology(batch_size,
                                                  n_ue, max_ut_velocity=SPEED,
                                                  scenario="uma")
            self._channel_model.set_topology(*topology)

    @tf.function  # We don't use jit_compile=True to ensure better numerical stability
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)

        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db=ebno_db, num_bits_per_symbol=num_bits_per_symbol,
                       coderate=R)  # normalize in OFDM freq. domain
        b = self._binary_source([batch_size, n_ue, 1, K])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel([x_rg, no_])
        print(no)
        #print("b:",b)
        #print("c:",c)
        #print("x:",x)
        #print("x_rg:",x_rg)
        #print("x_rg+1", tf.expand_dims(x_rg,axis = -1))
        #print("x_rg+2", tf.expand_dims(tf.expand_dims(x_rg,axis = -1),axis = -1))
        #print("x_rg/2",x_rg/2)
        #print("x_rg/c2", x_rg / tf.complex(2.0, 0.0))
        """
        print(f"1 shape of y: {tf.shape(y)}")
        print("batch size, num_tx_ant, num_rx_ant，num_ofdm_sym, num_eff_subcarriers")
        """
        #print(y)
        
        ######################################
        ## Receiver
        if self._perfect_csi_rayleigh:
            h_hat = self._removeNulledSc(h)
            chan_est_var = tf.zeros(tf.shape(h_hat),
                                    dtype=tf.float32)  # No channel estimation error when perfect CSI knowledge is assumed
        else:
            h_hat, chan_est_var = self._ls_est([y, no])

        llr_ch = self._detector((y, h_hat, chan_est_var, no))  # soft-output LMMSE detection
        #print(f"shape of llr_ch: {tf.shape(llr_ch)}")
        msg_vn = None

        if self._num_idd_iter >= 2:
            # perform first iteration outside the while_loop to initialize msg_vn
            [llr_dec, msg_vn] = self._siso_decoder((llr_ch, msg_vn))
            # forward a posteriori information from decoder
            #print(f"shape of llr_dec: {tf.shape(llr_dec)}")
            print(f"2 shape of h: {tf.shape(h_hat)}")
            
            #llr_ach = self._siso_detector((y, h_hat, llr_dec, chan_est_var, no))
            # forward extrinsic information
            

            def idd_iter(llr_ch, msg_vn, it):
                [llr_dec, msg_vn] = self._siso_decoder([llr_ch, msg_vn])
                # forward a posteriori information from decoder
                print("3 shape of llr_dec", llr_dec.shape)
                llr_ch = self._siso_detector((y, h_hat, llr_dec, chan_est_var, no))
                # forward extrinsic information from detector

                it += 1
                return llr_ch, msg_vn

            def idd_stop(llr_ch, msg_vn, it):
                return tf.less(it, self._num_idd_iter - 1)

            it = tf.constant(1)     # we already performed initial detection and one full iteration
            llr_ch, msg_vn, it = tf.while_loop(idd_stop, idd_iter, (llr_ch, msg_vn, it), parallel_iterations=1,
                                               maximum_iterations=self._num_idd_iter - 1)
        else:
            # non-idd
            pass

        [b_hat, _] = self._decoder((llr_ch, msg_vn))    # final hard-output decoding (only returning information bits)
        return b, b_hat
    
    
# Range of SNR (dB)
snr_range_cest = np.linspace(ebno_db_min_cest, ebno_db_max_cest, num_steps)
snr_range_perf_csi = np.linspace(ebno_db_min_perf_csi, ebno_db_max_perf_csi, num_steps)

def run_idd_sim(snr_range, perfect_csi_rayleigh):
    idd2 = IddModel(_siso_detector = 'BPIC', num_idd_iter=3, perfect_csi_rayleigh=perfect_csi_rayleigh)
    idd2_m = IddModel(_siso_detector = 'MMSEPIC',num_idd_iter=3, perfect_csi_rayleigh=perfect_csi_rayleigh)



    ber_idd2, bler_idd2 = sim_ber(idd2,
                                  snr_range,
                                  batch_size=batch_size,
                                  max_mc_iter=num_iter,
                                  num_target_block_errors=int(batch_size * num_iter * 0.1))
    ber_idd2_m, bler_idd2_m = sim_ber(idd2_m,
                                  snr_range,
                                  batch_size=batch_size,
                                  max_mc_iter=num_iter,
                                  num_target_block_errors=int(batch_size * num_iter * 0.1))


    return bler_idd2,  bler_idd2_m


BLER = {}

# Perfect CSI
bler_idd2, bler_idd2_m = run_idd_sim(snr_range_perf_csi, perfect_csi_rayleigh=True)


BLER['Perf. CSI / IDD2'] = bler_idd2
BLER['Perf. CSI / IDD2_m'] = bler_idd2_m

# Estimated CSI
bler_idd2, bler_idd2_m = run_idd_sim(snr_range_cest, perfect_csi_rayleigh=False)


BLER['Ch. Est. / IDD2'] = bler_idd2
BLER['Ch. Est. / IDD2_m'] = bler_idd2_m


fig, ax = plt.subplots(1,2, figsize=(16,7))
fig.suptitle(f"{n_ue}x{NUM_RX_ANT} MU-MIMO UL | {2**num_bits_per_symbol}-QAM")

## Perfect CSI Rayleigh
ax[0].set_title("Perfect CSI iid. Rayleigh")

ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / IDD2'], 'd:', label=r'IDD $I=2$', c='C1')
ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / IDD2_m'], 'd:', label=r'IDD_m $I=2$', c='C2')

ax[0].set_xlabel(r"$E_b/N0$")
ax[0].set_ylabel("BLER")
ax[0].set_ylim((1e-4, 1.0))
ax[0].legend()
ax[0].grid(True)

## Estimated CSI Rayleigh
ax[1].set_title("Estimated CSI 3GPP UMa")

ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / IDD2'], 'd:', label=r'IDD $I=2$', c='C1')
ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / IDD2_m'], 'd:', label=r'IDD_m $I=2$', c='C2')

ax[1].set_xlabel(r"$E_b/N0$")
ax[1].set_ylabel("BLER")
ax[1].set_ylim((1e-3, 1.0))
ax[1].legend()
ax[1].grid(True)

plt.show()