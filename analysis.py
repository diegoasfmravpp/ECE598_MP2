import numpy as np				# Data manipulation and math
import scipy.io.wavfile as wav			# Importing .wav files
import matplotlib.pyplot as plt			# Plotting 2D plots
from scipy.fft import fft, fftfreq			# Taking 1-sided FFT
from scipy.signal import filtfilt, butter		# Smoothing data with filtfilt zero-phase filter
import sys

def compute_fft(signal, sample_rate):
    N = len(signal)
    freq = fftfreq(N, d=1/sample_rate)[:N//2]  # One-sided frequencies
    fft_values = fft(signal)[:N//2]  		# Compute FFT and take one-sided spectrum
    return freq, fft_values

def smooth_impedance(magnitude, phase, cutoff=0.1, order=2):
    # magnitude = np.abs(impedance)
    # phase = np.angle(impedance, deg=True)
        
    # Design Butterworth low-pass filter
    b, a = butter(order, cutoff, btype='low', analog=False)
        
    # Apply zero-phase filtering
    smoothed_magnitude = filtfilt(b, a, magnitude)
    smoothed_phase = filtfilt(b, a, phase)
            
    # Reform complex impedance with magnitude and complex exponential
    # return smoothed_magnitude * np.exp(1j * np.radians(smoothed_phase)) 
    return smoothed_magnitude, smoothed_phase

def plot_signals(time, voltage_signal, current_signal):
    print("Plotting signals...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    axes[0].plot(time, voltage_signal, label="Voltage", color='b')
    axes[0].set_ylabel("Voltage (V)")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(time, current_signal, label="Current", color='r')
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Current (A)")
    axes[1].legend()
    axes[1].grid()
        
    plt.tight_layout()
    plt.show()

def plot_bode(freq, magnitude, phase):

    smoothed_magnitude, smoothed_phase = smooth_impedance(magnitude, phase)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Magnitude plot
    axes[0].semilogx(freq, magnitude, label='raw')  # Linear magnitude
    axes[0].semilogx(freq, smoothed_magnitude, label='smoothed')  # Linear magnitude
    axes[0].legend()
    axes[0].set_ylabel("Magnitude (Ohms)")
    axes[0].set_ylim(7, 25)  
    axes[0].set_xlim(10, 500)  # Limits x-axis from 10 Hz to 500Hz
    axes[0].grid(True, which="both", linestyle="--")
    axes[0].set_title("Bode Plot of Impedance")

    # Phase plot
    axes[1].semilogx(freq, phase, label='raw')  # Phase in degrees
    axes[1].semilogx(freq, smoothed_phase, label='smoothed')  # Phase in degrees
    axes[1].legend()
    axes[1].set_ylim(-90, 90)  
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (degrees)")
    axes[1].grid(True, which="both", linestyle="--")

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    filename = None

    if len(sys.argv) < 2:
        print("no wav file passed. Exiting...")
        sys.exit(1)
    else:
        filename = sys.argv[1]

    sample_rate, data = wav.read(filename)

    voltage_signal_int = data[:, 0].astype(np.int16)  # First channel as voltage
    current_signal_int = data[:, 1].astype(np.int16)  # Second channel as current

    LSB_Volts = 11.0 / 32768  # Least Significant Bit size
    voltage_signal_volts = voltage_signal_int * LSB_Volts  # Convert to Volts
            
    LSB_Amps = 3.0 / 32768  # Least Significant Bit size
    current_signal_amps = current_signal_int * LSB_Amps  # Convert to Amps

    # Compute ffts
    freq_voltage, fft_voltage = compute_fft(voltage_signal_volts, sample_rate)
    freq_current, fft_current = compute_fft(current_signal_amps, sample_rate)

    # Compute impedance
    impedance = fft_voltage / fft_current

    # Find impedance magnitude and phase
    magnitude = np.abs(impedance)		# Returns complex magnitude
    phase = np.angle(impedance, deg=True)	# Returns complex phase, in degrees

    time = np.linspace(0, len(voltage_signal_volts) / sample_rate, num=len(voltage_signal_volts))

    # plots
    # plot_signals(time, voltage_signal_volts, current_signal_amps)
    plot_bode(freq_voltage, magnitude, phase)
