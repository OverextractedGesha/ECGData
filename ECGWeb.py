import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --- HELPER: Manual IIR Filter Functions ---

def manual_iir_lowpass(data, cutoff, fs):
    """
    First-Order IIR Low Pass Filter.
    Formula: y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    """
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
    """
    First-Order IIR High Pass Filter.
    Formula: y[i] = alpha * (y[i-1] + x[i] - x[i-1])
    """
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
    
    # Apply mask and Inverse FFT
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
        ax.plot(df['Index'], df['Value'], label='Pre-Filtered Signal', color='#1f77b4', linewidth=1.2)
    else:
        ax.plot(df['Index'], df['Value'], label='Signal', color='#1f77b4', linewidth=1)
        
    ax.set_title('ECG Signal Preview')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_xlabel('Time')
    ax.grid(True, alpha=0.3)
    
    if x_range:
        ax.set_xlim(x_range)
        
    ax.legend()
    return fig

# --- MANUAL DFT FOR PLOTTING ---
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
st.title("ECG Analysis: Pre-Filtering & Segment DFT")

# --- SIDEBAR CONFIGURATION ---
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
        
        # --- CALCULATE SAMPLING RATE ---
        try:
            time_diffs = np.diff(df_raw['Index'])
            fs_est = 1.0 / np.median(time_diffs)
        except:
            fs_est = 100.0
        st.sidebar.write(f"**Fs:** {fs_est:.1f} Hz")
        
        # --- PRE-FILTERING SECTION ---
        st.sidebar.markdown("---")
        st.sidebar.header("2. Pre-Filtering (Global)")
        
        filter_type = st.sidebar.selectbox(
            "Select Pre-Filter Method",
            ["None", "High Pass Filter (HPF)", "Low Pass Filter (LPF)"]
        )
        
        df_processed = df_raw.copy()
        raw_for_plot = None 
        
        # --- FILTER LOGIC ---
        if filter_type == "High Pass Filter (HPF)":
            st.sidebar.caption("Best for: Removing drifting baselines.")
            cutoff = st.sidebar.slider("HPF Cutoff (Hz)", 0.1, 5.0, 0.5, step=0.1)
            df_processed['Value'] = manual_iir_highpass(df_raw['Value'].values, cutoff, fs_est)
            raw_for_plot = df_raw
            st.sidebar.success(f"HPF Applied (> {cutoff} Hz)")
            
        elif filter_type == "Low Pass Filter (LPF)":
            st.sidebar.caption("Best for: Removing fuzzy/sharp noise.")
            # Ensure max cutoff doesn't exceed Nyquist
            max_cutoff = float(fs_est / 2.0) - 1.0
            cutoff = st.sidebar.slider("LPF Cutoff (Hz)", 10.0, max_cutoff, 40.0, step=1.0)
            df_processed['Value'] = manual_iir_lowpass(df_raw['Value'].values, cutoff, fs_est)
            raw_for_plot = df_raw
            st.sidebar.success(f"LPF Applied (< {cutoff} Hz)")
            
        # --- MAIN PREVIEW ---
        st.subheader("1. Data Preview")
        st.caption(f"Visualizing data after {filter_type}")
        
        min_time = float(df_processed['Index'].min())
        max_time = float(df_processed['Index'].max())
        
        col1, col2 = st.columns([3, 1])
        with col1:
             zoom_range = st.slider("Zoom (Time Range)", min_time, max_time, (min_time, max_time))
        
        fig_full = create_full_plot(df_processed, zoom_range, raw_df=raw_for_plot) 
        st.pyplot(fig_full)

        st.subheader("2. Select a Single ECG Cycle")
        
        default_duration = (max_time - min_time) * 0.1
        if default_duration == 0: default_duration = 1.0
        default_start = min_time
        default_end = min(min_time + default_duration, max_time)
        step_size = 0.01

        with st.form(key='ecg_cycle_form'):
            c1, c2 = st.columns(2)
            with c1:
                start_index_input = st.number_input('Start Time', value=default_start, min_value=min_time, max_value=max_time, step=step_size, format="%.3f")
            with c2:
                end_index_input = st.number_input('End Time', value=default_end, min_value=min_time, max_value=max_time, step=step_size, format="%.3f")
            submit_button = st.form_submit_button(label='Analyze Cycle')

        if submit_button:
            start_index = float(start_index_input)
            end_index = float(end_index_input)

            if start_index >= end_index:
                st.error("Error: Start Index must be less than End Index.")
                st.session_state.cycle_selected = False
            else:
                df_cycle_check = df_processed[(df_processed['Index'] >= start_index) & (df_processed['Index'] <= end_index)]
                if len(df_cycle_check) < 3: 
                    st.error("Error: Not enough data points selected.")
                    st.session_state.cycle_selected = False
                else:
                    st.session_state.cycle_selected = True
                    st.session_state.start_index = start_index
                    st.session_state.end_index = end_index

        if st.session_state.get('cycle_selected', False):
            
            start_index = st.session_state.start_index
            end_index = st.session_state.end_index
            
            # Extract cycle from the PRE-FILTERED data
            df_cycle = df_processed[(df_processed['Index'] >= start_index) & (df_processed['Index'] <= end_index)].copy()

            st.subheader("3. Identify P, QRS, and T Complexes")
            cycle_duration = end_index - start_index
            
            p_wave_range = st.slider("Select P Wave Range", min_value=start_index, max_value=end_index, value=(start_index, start_index + (cycle_duration * 0.15)), step=step_size/2)
            qrs_complex_range = st.slider("Select QRS Complex Range", min_value=start_index, max_value=end_index, value=(start_index + (cycle_duration * 0.2), start_index + (cycle_duration * 0.4)), step=step_size/2)
            t_wave_range = st.slider("Select T Wave Range", min_value=start_index, max_value=end_index, value=(start_index + (cycle_duration * 0.5), start_index + (cycle_duration * 0.8)), step=step_size/2)

            # Store segments
            st.session_state.p_wave_data = df_cycle[(df_cycle['Index'] >= p_wave_range[0]) & (df_cycle['Index'] <= p_wave_range[1])]
            st.session_state.qrs_data = df_cycle[(df_cycle['Index'] >= qrs_complex_range[0]) & (df_cycle['Index'] <= qrs_complex_range[1])]
            st.session_state.t_wave_data = df_cycle[(df_cycle['Index'] >= t_wave_range[0]) & (df_cycle['Index'] <= t_wave_range[1])]

            # --- PLOT HIGHLIGHTED REGIONS ---
            fig_highlight, ax_highlight = plt.subplots(figsize=(10, 4))
            ax_highlight.plot(df_cycle['Index'], df_cycle['Value'], color='black', alpha=0.8, label='Cycle Data')
            ax_highlight.axvspan(p_wave_range[0], p_wave_range[1], color='blue', alpha=0.2, label='P Wave')
            ax_highlight.axvspan(qrs_complex_range[0], qrs_complex_range[1], color='red', alpha=0.2, label='QRS')
            ax_highlight.axvspan(t_wave_range[0], t_wave_range[1], color='green', alpha=0.2, label='T Wave')
            ax_highlight.legend()
            st.pyplot(fig_highlight)

            st.markdown("---")
            st.subheader("4. Segment-Based Frequency Domain Filtering")
            
            c_freq1, c_freq2 = st.columns(2)
            with c_freq1:
                low_cut_filter = st.number_input("DFT Low Cutoff (Hz)", value=0.5, step=0.5, min_value=0.0)
            with c_freq2:
                high_cut_filter = st.number_input("DFT High Cutoff (Hz)", value=40.0, step=1.0, max_value=fs_est/2)

            # --- PROCESS SEGMENTS ---
            p_raw = st.session_state.p_wave_data['Value'].values
            p_filtered = fft_brickwall_filter(p_raw, fs_est, low_cut_filter, high_cut_filter)
            
            qrs_raw = st.session_state.qrs_data['Value'].values
            qrs_filtered = fft_brickwall_filter(qrs_raw, fs_est, low_cut_filter, high_cut_filter)

            t_raw = st.session_state.t_wave_data['Value'].values
            t_filtered = fft_brickwall_filter(t_raw, fs_est, low_cut_filter, high_cut_filter)

            # --- VISUALIZE RECONSTRUCTION ---
            st.write("### Filtered Segments Reconstruction")
            fig_recon, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            ax1.set_title("P Wave")
            if len(p_raw) > 0:
                ax1.plot(st.session_state.p_wave_data['Index'], p_raw, color='lightgray', label='Input')
                ax1.plot(st.session_state.p_wave_data['Index'], p_filtered, color='blue', label='DFT Filtered')
            
            ax2.set_title("QRS Complex")
            if len(qrs_raw) > 0:
                ax2.plot(st.session_state.qrs_data['Index'], qrs_raw, color='lightgray', label='Input')
                ax2.plot(st.session_state.qrs_data['Index'], qrs_filtered, color='red', label='DFT Filtered')

            ax3.set_title("T Wave")
            if len(t_raw) > 0:
                ax3.plot(st.session_state.t_wave_data['Index'], t_raw, color='lightgray', label='Input')
                ax3.plot(st.session_state.t_wave_data['Index'], t_filtered, color='green', label='DFT Filtered')
            
            st.pyplot(fig_recon)

            # --- SHOW DFT SPECTRUM ---
            st.subheader("DFT Spectrum of Segments (Magnitude)")
            
            xf_p, yf_p, fs_p = calculate_dft(st.session_state.p_wave_data, fs_est)
            xf_qrs, yf_qrs, fs_qrs = calculate_dft(st.session_state.qrs_data, fs_est)
            xf_t, yf_t, fs_t = calculate_dft(st.session_state.t_wave_data, fs_est)

            fig_dft, ax_dft = plt.subplots(figsize=(10, 5))
            if len(xf_p) > 0: ax_dft.plot(xf_p, yf_p, label='P Wave', color='blue', alpha=0.7)
            if len(xf_qrs) > 0: ax_dft.plot(xf_qrs, yf_qrs, label='QRS Complex', color='red', alpha=0.7)
            if len(xf_t) > 0: ax_dft.plot(xf_t, yf_t, label='T Wave', color='green', alpha=0.7)
            
            ax_dft.axvline(low_cut_filter, color='k', linestyle='--', alpha=0.5, label='Cutoff')
            ax_dft.axvline(high_cut_filter, color='k', linestyle='--', alpha=0.5)

            ax_dft.set_xlabel("Frequency (Hz)")
            ax_dft.set_ylabel("Magnitude")
            ax_dft.legend()
            st.pyplot(fig_dft)
