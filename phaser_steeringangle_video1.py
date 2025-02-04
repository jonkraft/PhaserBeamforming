#  Must use Python 3
#  Copyright (C) 2025 Analog Devices, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#     - Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#     - Neither the name of Analog Devices, Inc. nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#     - The use of this software may or may not infringe the patent rights
#       of one or more patent holders.  This license does not release you
#       from the requirement that you obtain separate licenses from these
#       patent holders to use this software.
#     - Use of the software either in source or binary form, must be run
#       on or directly connected to an Analog Devices Inc. component.
#
# THIS SOFTWARE IS PROVIDED BY ANALOG DEVICES "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
#
# IN NO EVENT SHALL ANALOG DEVICES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, INTELLECTUAL PROPERTY
# RIGHTS, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

''' Simple Phased Array Beamforming Example Using Phaser and Python
    Follow the "Quick Start Guide" here first:  https://wiki.analog.com/phaser 
    This script uses the HB100 as the 10 GHz signal source, so OUT1 and OUT2 are disabled on Phaser
    This script can be found at https://github.com/jonkraft/PhaserBeamforming
    Jon Kraft, Jan 18 2025'''

# %%
# Imports
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')
from phaser_functions import load_hb100_cal
import adi
print(adi.__version__)

'''Key Parameters'''
sample_rate = 3e6 
rx_lo = 2.2e9
rx_gain = 0   # must be between -3 and 70
rpi_ip = "ip:phaser.local"  # default IP address of Phaser's Raspberry Pi
#sdr_ip = "ip:192.168.2.1"   # default Pluto IP address, connect pluto into USB of computer
sdr_ip = "ip:phaser.local:50901"  # using IIO context port forwarding, connect Pluto to Rasp Pi


# %% User parameters
#steer_angles = [0]
#steer_angles = [-30, 0, 30] # desired steering angles, in degrees
steer_angles = np.arange(-90, 90, 10)


# %% Create and Configure the Phaser Object
my_sdr = adi.ad9361(uri=sdr_ip)
my_phaser = adi.CN0566(uri=rpi_ip, sdr=my_sdr)

# Initialize both ADAR1000s, set gains to max, and all phases to 0
my_phaser.configure(device_mode="rx")
my_phaser.load_channel_cal()
my_phaser.load_gain_cal()
my_phaser.load_phase_cal()
try:
    my_phaser.SignalFreq = load_hb100_cal()
    print("Found signal freq file.  Freq is ", int(my_phaser.SignalFreq/1e6), " MHz")
except:
    my_phaser.SignalFreq = 10.525e9
    print("No signal freq file found, setting to 10.525 GHz")
SignalFreq = my_phaser.SignalFreq

for i in range(0, 8):
    my_phaser.set_chan_phase(i, 0)
gain_list = [127] * 8
for i in range(0, len(gain_list)):
    my_phaser.set_chan_gain(i, gain_list[i], apply_cal=True)

# Configure SDR Rx
my_sdr.sample_rate = int(sample_rate)
my_sdr.rx_buffer_size = int(1024)  # Number of samples per buffer
my_sdr._rxadc.set_kernel_buffers_count(1)  # Setting to 1 means there will be no stale buffers to flush
my_sdr.rx_lo = int(rx_lo)
my_sdr.rx_enabled_channels = [0, 1]   # enable Rx1 and Rx2
my_sdr.gain_control_mode_chan0 = 'manual'  # manual or slow_attack
my_sdr.gain_control_mode_chan1 = 'manual'  # manual or slow_attack
my_sdr.rx_hardwaregain_chan0 = int(rx_gain)   # must be between -3 and 70
my_sdr.rx_hardwaregain_chan1 = int(rx_gain)   # must be between -3 and 70

# Configure and disable SDR Tx
my_sdr.tx_enabled_channels = [0, 1]
my_sdr.tx_cyclic_buffer = False
my_sdr.tx_hardwaregain_chan0 = int(-88)   # must be between 0 and -88
my_sdr.tx_hardwaregain_chan1 = int(-88)   # must be between 0 and -88

# Set GPIOs on the Raspberry Pi
my_phaser._gpios.gpio_tx_sw = 0   # 0 = TX_OUT_2, 1 = TX_OUT_1
my_phaser._gpios.gpio_vctrl_1 = 1 # 1=Use onboard PLL/LO source  (0=disable PLL and VCO, and set switch to use external LO input)
my_phaser._gpios.gpio_vctrl_2 = 0 # 1=Send LO to transmit circuitry  (0=disable Tx path, and send LO to LO_OUT)

# Configure the ADF4159 Ramping PLL
vco_freq = int(my_phaser.SignalFreq + my_sdr.rx_lo)
my_phaser.frequency = int(vco_freq / 4)
my_phaser.ramp_mode = "disabled"  # ramp_mode can be:  "disabled", "continuous_sawtooth", "continuous_triangular", "single_sawtooth_burst", "single_ramp_burst"
my_phaser.enable = 0  # 0 = PLL enable.  Write this last to update all the registers

# %% Define Common Functions
def dbfs(raw_data):
    # function to convert IQ samples to FFT plot, scaled in dBFS
    NumSamples = len(raw_data)
    win = np.hamming(NumSamples)
    y = raw_data * win
    s_fft = np.fft.fft(y) / np.sum(win)
    s_shift = np.fft.fftshift(s_fft)
    s_dbfs = 20*np.log10(np.abs(s_shift)/(2**11))     # Pluto is a signed 12 bit ADC, so use 2^11 to convert to dBFS
    return s_dbfs

# %% Steer the beam and grab a buffer of data.  Then calc FFT using dbfs()
N = int(my_sdr.rx_buffer_size)
freqs = np.fft.fftshift(np.fft.fftfreq(N, 1 / sample_rate))
freqs /= 1e6  # Scale Hz -> MHz
fig, ax = plt.subplots()
line, = ax.plot([], [])
line.set_xdata(freqs)
ax.set_xlim(-sample_rate/(2*1e6), sample_rate/(2*1e6))
ax.set_ylim(-80, 10)
plt.title("FFT plot")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Amplitude (dBFS)")

for i in range(len(steer_angles)):
    print(int(steer_angles[i]))
    steer_rad = np.radians(steer_angles[i])
    PhDelta = 2*np.pi*my_phaser.element_spacing*np.sin(steer_rad) / (my_phaser.c/SignalFreq) # convert the steering angle into the required phase delta
    PhDelta = np.degrees(PhDelta)  # convert the phase difference to degrees
    for element in range(0, len(my_phaser.elements)):
        my_phaser.set_chan_phase(element, element*PhDelta, apply_cal=True)    
    
    data = my_sdr.rx()
    data_sum = data[0]+data[1]
    sum_dbfs = dbfs(data_sum)
    
    line.set_ydata(sum_dbfs)
    for text in ax.texts:
        text.remove()
    phases = np.arange(len(my_phaser.elements))*PhDelta%360
    phases = phases.astype(int)
    phases = np.array2string(phases, separator=',')[1:-1]
    ax.text(0.1, 0.9, f'Steering Angle = {steer_angles[i]}', transform=ax.transAxes, fontsize=12, color='red')
    ax.text(0.1, 0.8, f'Phases = {phases}', transform=ax.transAxes, fontsize=12, color='red')
    plt.draw()
    plt.show()
    plt.pause(2)
