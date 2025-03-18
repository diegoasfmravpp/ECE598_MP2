import numpy as np				# Data manipulation and math
import scipy.io.wavfile as wav			# Importing .wav files
import matplotlib.pyplot as plt			# Plotting 2D plots
from scipy.fft import fft, fftfreq			# Taking 1-sided FFT
from scipy.signal import filtfilt, butter		# Smoothing data with filtfilt zero-phase filter
import sys
import json
import os

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

def process_raw_data(sample_rate, data):
    voltage_signal_int = data[:, 0].astype(np.int16)  # First channel as voltage
    current_signal_int = data[:, 1].astype(np.int16)  # Second channel as current

    LSB_Volts = 11.0 / 32768  # Least Significant Bit size
    voltage_signal_volts = voltage_signal_int * LSB_Volts  # Convert to Volts
            
    LSB_Amps = 3.0 / 32768  # Least Significant Bit size
    current_signal_amps = current_signal_int * LSB_Amps  # Convert to Amps

    # Compute ffts
    freq, fft_voltage = compute_fft(voltage_signal_volts, sample_rate)
    _, fft_current = compute_fft(current_signal_amps, sample_rate)

    # Compute impedance
    impedance = fft_voltage / fft_current

    # Find impedance magnitude and phase
    magnitude = np.abs(impedance)		# Returns complex magnitude
    phase = np.angle(impedance, deg=True)	# Returns complex phase, in degrees

    return freq, magnitude, phase

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

def plot_bode(high_freq, freq, magnitude, phase, smoothed_magnitude, smoothed_phase, Zmax=None, fs=None, Re=None, f1=None, f2=None):

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Magnitude plot
    axes[0].semilogx(freq, magnitude, label='raw')  # Linear magnitude
    axes[0].semilogx(freq, smoothed_magnitude, label='smoothed')  # Linear magnitude

    if Zmax is not None:    
        axes[0].axhline(y=Zmax, color='b', linestyle='dotted', linewidth=2, label="Zmax: " + str(round(Zmax,2)))
        axes[0].axvline(x=fs, color='g', linestyle='dotted', linewidth=2, label="fs: " + str(round(fs,2)))
       
    if Re is not None:
        axes[0].axhline(y=Re, color='r', linestyle='dotted', linewidth=2, label="Re: " + str(round(Re,2)))
        axes[0].axvline(x=f1, color='y', linestyle='dotted', linewidth=2, label="f1: " + str(round(f1,2)))
        axes[0].axvline(x=f2, color='y', linestyle='dotted', linewidth=2, label="f2: " + str(round(f2,2)))
     
    axes[0].legend()
    axes[0].set_ylabel("Magnitude (Ohms)")
    axes[0].set_ylim(7, 25)  
    axes[0].set_xlim(10, high_freq)  # Limits x-axis from 10 Hz to 500Hz
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

def q_factors(freq, smooth_impedance, Re=None):
    last_fs = np.argmin(np.abs(freq - 500))
    # print("last_fs: " + str(last_fs))
    Z_magnitude = np.abs(smooth_impedance[:last_fs])
    fs_index = np.argmax(Z_magnitude)  	# Resonance frequency index
    fs = freq[fs_index]  # Resonance frequency (Hz)

    Zmax = Z_magnitude[fs_index]  		# Maximum impedance at resonance

    # avg impedance 10-40 Hz from narrow BW exp = 7.51
    if Re is None:
        indices = (freq[:last_fs] >= 10) & (freq[:last_fs] <= 40)
        Re = np.mean(Z_magnitude[indices])

    # print("fs_index: " + str(fs_index))
    print("fs: " + str(fs))
    print("zmax: " + str(Zmax))
    print("Re: " + str(Re))

    Z1_2 = np.sqrt(Re * Zmax) 	 # Side impedance magnitude levels
    
    # Find f1 and f2 where impedance equals Z1_2
    f1_index = np.where(Z_magnitude[:fs_index] >= Z1_2)[0][0]
    f2_index = np.where(Z_magnitude[fs_index:] <= Z1_2)[0][0] + fs_index
    f1, f2 = freq[f1_index], freq[f2_index]

    Qms = fs/(f2-f1)*Zmax/Re 
    Qes = Qms * (Zmax/Re - 1) 
    Qts =(Qms * Qes) / (Qms + Qes) 

    print("Qms: " + str(Qms))
    print("Qes: " + str(Qes))
    print("Qts: " + str(Qts))
    return Qms, Qes, Qts, Zmax, fs, Re, f1, f2

def save_values(filename, Zmax, fs, Re=None, f1=None, f2=None, Qms=None, Qes=None, Qts=None):

    # JSON file path
    json_file = "saved_values.json"

    # Load existing data if the file exists
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            try:
                data = json.load(f)  # Read existing JSON
            except json.JSONDecodeError:
                data = []  # If the file is empty or corrupted, start fresh
    else:
        data = []  # If file doesn't exist, start with an empty list

    # New calculated values (example) Qms, Qes, Qts, Zmax, fs, Re, f1, f2
    if Re is None:
        new_entry = {"filename": filename, "Zmax": Zmax, "fs": fs}
    else:
        new_entry = {"filename": filename, "Zmax": Zmax, "fs": fs, "Re": Re, "f1": f1, "f2": f2, "Qms": Qms, "Qes": Qes, "Qts": Qts}
    

    # Convert list to dictionary for easier updating
    data_dict = {entry["filename"]: entry for entry in data}

    # Update or insert new entry
    data_dict[new_entry["filename"]] = new_entry

    # Convert back to list format
    updated_data = list(data_dict.values())

    # Write back to JSON
    with open(json_file, "w") as f:
        json.dump(updated_data, f, indent=4)

    print(f"Saved entry: {new_entry}")

def other_factors():

    # JSON file path
    json_file = "saved_values.json"
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            try:
                data = json.load(f)  # Read existing JSON
            except json.JSONDecodeError:
                return "Not found"

    # Use the data
    if isinstance(data, list):  # If the JSON is a list
        for entry in data:
            filename = entry.get('filename')
            if filename == "narrow_bandwidth.wav":
                fs = entry.get('fs')
                Zmax = entry.get('Zmax')
                Re = entry.get('Re')
                Qms = entry.get('Qms')
            else:
                fo = entry.get('fs')
    else:
        print(data)  # Print error message if any

    Mms = 0.00015/((fs/fo)**2-1)

    inner = 2*np.pi*fs

    Cms = 1/((inner)**2*Mms)
    Rms = (inner*Mms) / Qms
    BL = np.sqrt((Zmax-Re)*Rms)

    print("Mms: " + str(Mms))
    print("Cms: " + str(Cms))
    print("Rms: " + str(Rms))
    print("BL: " + str(BL))
    
    return Mms, Cms, Rms, BL

def z_model(freq, BL, Re, Rms, Cms, Mms):
    freq = np.where(freq == 0, 1e-10, freq)  # Replace 0 with small number

    # Definitions of Impedances from T/S Parameters
    BL2 = (BL*BL)
    Zre = Re
    Zr = BL2 / (Rms)
    
    Zc = (BL2 * Cms) * (1j * 2 * np.pi * freq)    
    Zm = 1/((Mms / (BL2))* 1j * 2 * np.pi * freq)

    Z_parallel = 1.0 / (1/Zr + 1/Zc + 1/Zm) 	# Parallel connection of R, L, C
    Z_actuator = Zre + Z_parallel			# Combining RLC in parallel with Zre in series

    Z_total_magnitude = np.abs(Z_actuator) 	# Calculates the model’s total magnitude
    Z_total_phase = np.angle(Z_actuator, deg=True)	# Calculates the model’s total phase angle

    return Z_total_magnitude, Z_total_phase

def plot_model(high_freq, freq, magnitude, phase, modeled_impedance, modeled_phase):

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Magnitude plot
    axes[0].semilogx(freq, magnitude, label='Measured impedance')  # Linear magnitude
    axes[0].semilogx(freq, modeled_impedance, linestyle="--", label='Modeled impedance')  # Linear magnitude
    axes[0].legend()
    axes[0].set_ylabel("Magnitude (Ohms)")
    axes[0].set_ylim(7, 25)  
    axes[0].set_xlim(10, high_freq)  # Limits x-axis from 10 Hz to 500Hz
    axes[0].grid(True, which="both", linestyle="--")
    axes[0].set_title("Bode Plot of Impedance")

    # Phase plot
    axes[1].semilogx(freq, phase, label='Measured phase')  # Phase in degrees
    axes[1].semilogx(freq, modeled_phase, linestyle="--", label='Modeled phase')  # Phase in degrees
    axes[1].legend()
    axes[1].set_ylim(-90, 90)  
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (degrees)")
    axes[1].grid(True, which="both", linestyle="--")

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    filename = None

    if len(sys.argv) < 3:
        print("no wav file or highest frequency passed. Exiting...")
        sys.exit(1)
    else:
        filename = sys.argv[1]
        high_freq = float(sys.argv[2])

    sample_rate, data = wav.read(filename)

    freq, magnitude, phase = process_raw_data(sample_rate, data)

    smoothed_magnitude, smoothed_phase = smooth_impedance(magnitude, phase)

    # Q factors and plots
    if filename == "narrow_bandwidth.wav":
        Qms, Qes, Qts, Zmax, fs, Re, f1, f2 = q_factors(freq, smoothed_magnitude)
        plot_bode(high_freq, freq, magnitude, phase, smoothed_magnitude, smoothed_phase, Zmax, fs, Re, f1, f2)
        save_values(filename, Zmax, fs, Re, f1, f2, Qms, Qes, Qts)
        Mms, Cms, Rms, BL = other_factors()
        Z_total_magnitude, Z_total_phase = z_model(freq, BL, Re, Rms, Cms, Mms)
        plot_model(high_freq, freq, magnitude, phase, Z_total_magnitude, Z_total_phase)
    elif filename == "added_mass.wav"  or "light_touch.wav":
        Qms, Qes, Qts, Zmax, fs, Re, f1, f2 = q_factors(freq, smoothed_magnitude, 7.51)
        plot_bode(high_freq, freq, magnitude, phase, smoothed_magnitude, smoothed_phase, Zmax, fs)
        save_values(filename, Zmax, fs)
    else:
        plot_bode(high_freq, freq, magnitude, phase, smoothed_magnitude, smoothed_phase)

    # time = np.linspace(0, len(voltage_signal_volts) / sample_rate, num=len(voltage_signal_volts))
    # plot_signals(time, voltage_signal_volts, current_signal_amps)

    
