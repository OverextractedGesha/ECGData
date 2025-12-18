import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def manual_mean(data):
    total = 0.0
    count = 0
    for x in data:
        total += x
        count += 1
    if count == 0: return 0.0
    return total / count

def manual_square_signal(data):
    return data ** 2

def manual_moving_average_filter(data, window_size):
    n = len(data)
    window_size = int(window_size)
    if window_size < 1: return data
    y = np.zeros(n)
    kernel = np.ones(window_size) / window_size
    y = np.convolve(data, kernel, mode='same')
    return y

def manual_threshold_and_count(data, threshold):
    n = len(data)
    binary_signal = np.zeros(n)
    beats_count = 0
    
    for i in range(n):
        if data[i] > threshold:
            binary_signal[i] = 1.0
        else:
            binary_signal[i] = 0.0
            
    for i in range(1, n):
        prev = binary_signal[i-1]
        curr = binary_signal[i]
        if prev == 0 and curr == 1:
            beats_count += 1
            
    return binary_signal, beats_count

def sinc_func(x):
    if x == 0: return 1.0
    return np.sin(np.pi * x) / (np.pi * x)

def design_fir_bandpass(N, fs, f_low, f_high):
    alpha = (N - 1) / 2.0
    
    fc_low = f_low / fs
    fc_high = f_high / fs
    
    h = np.zeros(N)
    
    for n in range(N):
        m = n - alpha
        
        term1 = 2 * fc_high * sinc_func(2 * fc_high * m)
        term2 = 2 * fc_low * sinc_func(2 * fc_low * m)
        
        h[n] = term1 - term2
        
    for n in range(N):
        w = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
        h[n] = h[n] * w
        
    return h

def apply_fir_filter(data, coeffs):
    return np.convolve(data, coeffs, mode='same')

def calculate_dft_of_coefficients(h, fs, num_points=500):
    N = len(h)
    padded_h = np.zeros(num_points)
    padded_h[0:N] = h
    
    x_real = np.zeros(num_points)
    x_imaj = np.zeros(num_points)
    
    for k in range(num_points):
        for n in range(N): 
            angle = 2 * np.pi * k * n / num_points
            x_real[k] += padded_h[n] * np.cos(angle)
            x_imaj[k] -= padded_h[n] * np.sin(angle)
            
    magnitude = np.sqrt(x_real**2 + x_imaj**2)
    freqs = np.arange(num_points) * fs / num_points
    
    half_point = num_points // 2
    return freqs[:half_point], magnitude[:half_point]

@st.cache_data
def load_data(file_path_or_buffer):
    try:
        df = pd.read_csv(file_path_or_buffer, header=None)
        df.columns = ['Index', 'Value']
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

st.title("ECG Analysis: FIR Implementation")

st.sidebar.header("1. Data Load")
uploaded_file = st.sidebar.file_uploader("Choose a csv file", type="csv")
file_to_load = "DataHiMe.csv" if os.path.exists("DataHiMe.csv") else None
if uploaded_file is not None: file_to_load = uploaded_file

if file_to_load is not None:
    df_raw = load_data(file_to_load)
    
    if df_raw is not None:
        try:
            time_diffs = np.diff(df_raw['Index'])
            fs_est = 1.0 / np.median(time_diffs)
        except:
            fs_est = 100.0
        st.sidebar.write(f"**Fs:** {fs_est:.1f} Hz")
        
        st.sidebar.markdown("---")
        st.sidebar.header("2. FIR Filter Design")
        st.sidebar.info("Homework Logic: Order = Filter Length")
        
        fir_order = st.sidebar.slider("Filter Order (Length N)", 5, 101, 31, step=2)
        
        low_cut = st.sidebar.number_input("Low Cutoff (Hz)", 0.1, 50.0, 2.0, step=0.5)
        high_cut = st.sidebar.number_input("High Cutoff (Hz)", 1.0, 100.0, 30.0, step=1.0)
        
        coeffs = design_fir_bandpass(fir_order, fs_est, low_cut, high_cut)
        
        st.sidebar.markdown("---")
        st.sidebar.write("**FIR Coefficients (h[n]):**")
        st.sidebar.code(np.round(coeffs, 4))

        df_processed = df_raw.copy()
        filtered_values = apply_fir_filter(df_processed['Value'].values, coeffs)
        df_processed['Value'] = filtered_values

        st.subheader("Data Preview (Filtered)")
        min_time = float(df_processed['Index'].min())
        max_time = float(df_processed['Index'].max())
        zoom_range = st.slider("Zoom Time", min_time, max_time, (min_time, min(max_time, min_time+5.0)))
        
        fig_full, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_raw['Index'], df_raw['Value'], color='lightgray', alpha=0.5, label='Raw')
        ax.plot(df_processed['Index'], df_processed['Value'], color='blue', label=f'FIR Bandpass (N={fir_order})')
        ax.set_xlim(zoom_range)
        ax.legend()
        ax.set_title("Time Domain Result")
        st.pyplot(fig_full)

        st.markdown("---")
        st.subheader("Filter Frequency Response")
        st.write("Matches Homework Step (c): Magnitude Frequency Response up to Fs/2")
        
        freqs, magnitude = calculate_dft_of_coefficients(coeffs, fs_est, num_points=1000)
        
        col_a, col_b = st.columns([1, 2])
        
        with col_a:
            st.write("**Difference Equation:**")
            st.latex(r"y[n] = \sum_{k=0}^{N-1} h[k] x[n-k]")
            st.write(f"This filter uses **{fir_order}** taps (coefficients).")
            st.dataframe(pd.DataFrame({'h[n]': coeffs}), height=300)
            
        with col_b:
            fig_freq, (ax_imp, ax_mag) = plt.subplots(2, 1, figsize=(8, 8))
            
            ax_imp.stem(range(len(coeffs)), coeffs)
            ax_imp.set_title("Impulse Response h[n] (Filter Coefficients)")
            ax_imp.set_xlabel("Sample Index (n)")
            ax_imp.grid(True, alpha=0.3)
            
            ax_mag.plot(freqs, magnitude, color='purple', linewidth=2)
            ax_mag.set_title("Magnitude Frequency Response |H(w)|")
            ax_mag.set_xlabel("Frequency (Hz)")
            ax_mag.set_ylabel("Gain")
            
            ax_mag.axvline(low_cut, color='red', linestyle='--', alpha=0.5, label='Low Cut')
            ax_mag.axvline(high_cut, color='red', linestyle='--', alpha=0.5, label='High Cut')
            
            ax_mag.grid(True, alpha=0.3)
            ax_mag.legend()
            
            st.pyplot(fig_freq)

        st.markdown("---")
        st.subheader("Beat Detection (Using FIR Output)")
        
        sq_signal = manual_square_signal(filtered_values)
        
        window_ms = st.slider("MAV Window (ms)", 10, 300, 150)
        win_samples = int(window_ms * fs_est / 1000)
        mav_signal = manual_moving_average_filter(sq_signal, win_samples)
        
        c1, c2 = st.columns(2)
        calc_start = c1.number_input("Start (s)", min_time, max_time, zoom_range[0])
        calc_end = c2.number_input("End (s)", min_time, max_time, zoom_range[1])
        
        mask = (df_processed['Index'] >= calc_start) & (df_processed['Index'] <= calc_end)
        seg_time = df_processed['Index'][mask]
        seg_mav = mav_signal[mask]
        
        if len(seg_mav) > 0:
            max_amp = np.max(seg_mav)
            thresh_perc = st.slider("Threshold %", 10, 90, 40)
            threshold = max_amp * thresh_perc / 100.0
            
            binary, count = manual_threshold_and_count(seg_mav, threshold)
            
            bpm = 0
            dur = calc_end - calc_start
            if dur > 0: bpm = (count / dur) * 60
            
            st.metric("Heart Rate", f"{bpm:.1f} BPM", f"{count} Beats")
            
            fig_res, ax_res = plt.subplots(figsize=(10, 4))
            ax_res.plot(seg_time, seg_mav, 'orange', label='MAV')
            ax_res.axhline(threshold, color='k', linestyle='--', label='Threshold')
            ax_res.plot(seg_time, binary * max_amp, 'r', alpha=0.3, label='Pulse')
            ax_res.set_title("Detection Result")
            ax_res.legend()
            st.pyplot(fig_res);
            
