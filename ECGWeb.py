import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --- HELPER: Manual IIR Filter Functions ---
def manual_iir_lowpass(data, cutoff, fs):
    if cutoff <= 0: return data
    dt = 1.0 / fs
    rc = 1.0 / (2 * np.pi * cutoff)
    alpha = dt / (rc + dt)
    n = len(data)
    y = np.zeros(n)
    y[0] = data[0]
    for i in range(1, n):
        y[i] = alpha * data[i] + (1 - alpha) * y[i-1]
    return y

def manual_iir_highpass(data, cutoff, fs):
    if cutoff <= 0: return data
    dt = 1.0 / fs
    rc = 1.0 / (2 * np.pi * cutoff)
    alpha = rc / (rc + dt)
    n = len(data)
    y = np.zeros(n)
    y[0] = 0 
    for i in range(1, n):
        y[i] = alpha * (y[i-1] + data[i] - data[i-1])
    return y

# --- HELPER: Frequency Domain Filter (Brick-wall) ---
def fft_brickwall_filter(data_segment, fs, low_cut, high_cut):
    n = len(data_segment)
    if n == 0: return data_segment
    
    # 1. FFT
    fft_coeffs = np.fft.fft(data_segment)
    frequencies = np.fft.fftfreq(n, d=1/fs)
    
    # 2. Create Mask
    mask = (np.abs(frequencies) >= low_cut) & (np.abs(frequencies) <= high_cut)
    
    # 3. Apply mask and Inverse FFT
    fft_filtered = fft_coeffs * mask
    filtered_signal = np.fft.ifft(fft_filtered)
    
    return np.real(filtered_signal)

@st.cache_data
def load_data(file_path_or_buffer):
    try:
        df = pd.read_csv(file_path_or_buffer, header=None)
        df.columns = ['Index', 'Value']
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

@st.cache_data
def create_full_plot(df, x_range=None, raw_df=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if raw_df is not None:
        ax.plot(raw_df['Index'], raw_df['Value'], label='Original Raw', color='lightgray', alpha=0.6, linewidth=1)
        ax.plot(df['Index'], df['Value'], label='Filtered Signal', color='#1f77b4', linewidth=1.2)
    else:
        ax.plot(df['Index'], df['Value'], label='Signal', color='#1f77b4', linewidth=1)
        
    ax.set_title('ECG Signal Preview')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_xlabel('Time')
    ax.grid(True, alpha=0.3)
    if x_range: ax.set_xlim(x_range)
    ax.legend()
    return fig

@st.cache_data
def calculate_dft(df_segment, fs):
    N_dft = 200
    original_signal = df_segment['Value'].values
    N_orig = len(original_signal)
    if N_orig < 2: return np.array([]), np.array([]), 0

    x_detrended = original_signal - np.mean(original_signal)
    x_padded = np.zeros(N_dft)
    points_to_copy = min(N_orig, N_dft)
    x_padded[0:points_to_copy] = x_detrended[0:points_to_copy]
    
    x_real = np.zeros(N_dft)
    x_imaj = np.zeros(N_dft)
    for k in range (N_dft):
        for n in range (N_dft):
            x_real[k] += x_padded[n]*np.cos(2*np.pi*k*n/N_dft)
            x_imaj[k] -= x_padded[n]*np.sin(2*np.pi*k*n/N_dft)
    
    half_N = round(N_dft/2)
    MagDFT = np.zeros(half_N)
    for k in range (half_N):
        MagDFT[k] = np.sqrt(np.square(x_real[k]) + np.square(x_imaj[k]))
    
    xf_positive = np.arange(0, half_N) * fs / N_dft
    yf_positive_magnitude = MagDFT * 2.0 / N_orig
    if half_N > 0: yf_positive_magnitude[0] = MagDFT[0] / N_orig 
    return xf_positive, yf_positive_magnitude, fs

# --- MAIN APP ---
st.title("ECG Analysis: Bandpass Pre-Filtering")

# 1. Data Load
st.sidebar.header("1. Data Load")
uploaded_file = st.sidebar.file_uploader("Choose a csv file", type="csv")

file_to_load = None
if uploaded_file is not None:
    file_to_load = uploaded_file
elif os.path.exists("DataHiMe.csv"):
    file_to_load = "DataHiMe.csv"
    st.sidebar.info("Using default: DataHiMe.csv")

if file_to_load is not None:
    df_raw = load_data(file_to_load)
    
    if df_raw is not None:
        # Calculate FS
        try:
            time_diffs = np.diff(df_raw['Index'])
            fs_est = 1.0 / np.median(time_diffs)
        except:
            fs_est = 100.0
        st.sidebar.write(f"**Fs:** {fs_est:.1f} Hz")
        
        # --- PRE-FILTERING SECTION ---
        st.sidebar.markdown("---")
        st.sidebar.header("2. Pre-Filtering (Global)")
        st.sidebar.caption("Optimal: Enable BOTH to create a Bandpass Filter.")
        
        df_processed = df_raw.copy()
        is_filtered = False

        # 1. High Pass Checkbox
        use_hpf = st.sidebar.checkbox("Enable High Pass (Remove Drift)", value=True)
        if use_hpf:
            cutoff_h = st.sidebar.slider("HPF Cutoff (Hz)", 0.1, 5.0, 0.5, step=0.1)
            df_processed['Value'] = manual_iir_highpass(df_processed['Value'].values, cutoff_h, fs_est)
            is_filtered = True
            
        # 2. Low Pass Checkbox
        use_lpf = st.sidebar.checkbox("Enable Low Pass (Remove Noise)", value=True)
        if use_lpf:
            max_cutoff = float(fs_est / 2.0) - 1.0
            cutoff_l = st.sidebar.slider("LPF Cutoff (Hz)", 10.0, max_cutoff, 40.0, step=1.0)
            df_processed['Value'] = manual_iir_lowpass(df_processed['Value'].values, cutoff_l, fs_est)
            is_filtered = True

        raw_for_plot = df_raw if is_filtered else None
            
        # --- MAIN PREVIEW ---
        st.subheader("1. Data Preview")
        min_time = float(df_processed['Index'].min())
        max_time = float(df_processed['Index'].max())
        
        col1, col2 = st.columns([3, 1])
        with col1:
             zoom_range = st.slider("Zoom (Time Range)", min_time, max_time, (min_time, max_time))
        
        fig_full = create_full_plot(df_processed, zoom_range, raw_df=raw_for_plot) 
        st.pyplot(fig_full)

        # --- CYCLE SELECTION ---
        st.subheader("2. Select a Single ECG Cycle")
        default_duration = (max_time - min_time) * 0.1
        if default_duration == 0: default_duration = 1.0
        default_start = min_time
        default_end = min(min_time + default_duration, max_time)

        with st.form(key='ecg_cycle_form'):
            c1, c2 = st.columns(2)
            with c1: start_input = st.number_input('Start Time', value=default_start, step=0.01, format="%.3f")
            with c2: end_input = st.number_input('End Time', value=default_end, step=0.01, format="%.3f")
            submit_button = st.form_submit_button(label='Analyze Cycle')

        if submit_button:
            if start_input >= end_input:
                st.error("Error: Start < End")
            else:
                st.session_state.cycle_selected = True
                st.session_state.start_index = start_input
                st.session_state.end_index = end_input

        if st.session_state.get('cycle_selected', False):
            start_index = st.session_state.start_index
            end_index = st.session_state.end_index
            df_cycle = df_processed[(df_processed['Index'] >= start_index) & (df_processed['Index'] <= end_index)].copy()

            st.subheader("3. Identify Complexes")
            dur = end_index - start_index
            p_range = st.slider("P Wave", start_index, end_index, (start_index, start_index + dur*0.15), step=0.01)
            qrs_range = st.slider("QRS Complex", start_index, end_index, (start_index + dur*0.2, start_index + dur*0.4), step=0.01)
            t_range = st.slider("T Wave", start_index, end_index, (start_index + dur*0.5, start_index + dur*0.8), step=0.01)

            st.session_state.p_data = df_cycle[(df_cycle['Index'] >= p_range[0]) & (df_cycle['Index'] <= p_range[1])]
            st.session_state.qrs_data = df_cycle[(df_cycle['Index'] >= qrs_range[0]) & (df_cycle['Index'] <= qrs_range[1])]
            st.session_state.t_data = df_cycle[(df_cycle['Index'] >= t_range[0]) & (df_cycle['Index'] <= t_range[1])]

            # Plot Highlight
            fig_hl, ax_hl = plt.subplots(figsize=(10, 4))
            ax_hl.plot(df_cycle['Index'], df_cycle['Value'], 'k', alpha=0.8)
            ax_hl.axvspan(p_range[0], p_range[1], color='blue', alpha=0.2, label='P')
            ax_hl.axvspan(qrs_range[0], qrs_range[1], color='red', alpha=0.2, label='QRS')
            ax_hl.axvspan(t_range[0], t_range[1], color='green', alpha=0.2, label='T')
            ax_hl.legend()
            st.pyplot(fig_hl)

            st.markdown("---")
            st.subheader("4. Segment Analysis")
            
            # --- FILTER INPUTS ---
            st.write("**Frequency Domain Filter Settings**")
            c_freq1, c_freq2 = st.columns(2)
            with c_freq1: low_dft = st.number_input("DFT Low Cut (Hz)", 0.5, step=0.5)
            with c_freq2: high_dft = st.number_input("DFT High Cut (Hz)", 40.0, step=1.0)
            
            # --- 1. CALCULATE AND PLOT DFT (NOW FIRST) ---
            st.write("#### A. DFT Spectrum (Frequency Domain)")
            xf_p, yf_p, _ = calculate_dft(st.session_state.p_data, fs_est)
            xf_qrs, yf_qrs, _ = calculate_dft(st.session_state.qrs_data, fs_est)
            xf_t, yf_t, _ = calculate_dft(st.session_state.t_data, fs_est)
            
            fig_dft, ax_dft = plt.subplots(figsize=(10, 5))
            if len(xf_p)>0: ax_dft.plot(xf_p, yf_p, label='P', color='blue')
            if len(xf_qrs)>0: ax_dft.plot(xf_qrs, yf_qrs, label='QRS', color='red')
            if len(xf_t)>0: ax_dft.plot(xf_t, yf_t, label='T', color='green')
            
            # Add cutoff lines
            ax_dft.axvline(low_dft, c='k', ls='--', alpha=0.5, label='Cutoff')
            ax_dft.axvline(high_dft, c='k', ls='--', alpha=0.5)
            ax_dft.set_xlabel("Frequency (Hz)")
            ax_dft.set_ylabel("Magnitude")
            ax_dft.legend()
            st.pyplot(fig_dft)

            # --- 2. CALCULATE AND PLOT RECONSTRUCTION (NOW SECOND) ---
            st.write("#### B. Reconstructed Signal (Time Domain)")
            
            # Process Filter
            p_raw = st.session_state.p_data['Value'].values
            p_filt = fft_brickwall_filter(p_raw, fs_est, low_dft, high_dft)
            qrs_raw = st.session_state.qrs_data['Value'].values
            qrs_filt = fft_brickwall_filter(qrs_raw, fs_est, low_dft, high_dft)
            t_raw = st.session_state.t_data['Value'].values
            t_filt = fft_brickwall_filter(t_raw, fs_est, low_dft, high_dft)

            # Recon Plot
            fig_rec, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            ax1.set_title("P Wave")
            ax1.plot(st.session_state.p_data['Index'], p_raw, color='lightgray', label='Input')
            ax1.plot(st.session_state.p_data['Index'], p_filt, color='blue', label='Filtered')
            ax1.legend()
            
            ax2.set_title("QRS Complex")
            ax2.plot(st.session_state.qrs_data['Index'], qrs_raw, color='lightgray', label='Input')
            ax2.plot(st.session_state.qrs_data['Index'], qrs_filt, color='red', label='Filtered')
            ax2.legend()
            
            ax3.set_title("T Wave")
            ax3.plot(st.session_state.t_data['Index'], t_raw, color='lightgray', label='Input')
            ax3.plot(st.session_state.t_data['Index'], t_filt, color='green', label='Filtered')
            ax3.legend()
            
            st.pyplot(fig_rec)
