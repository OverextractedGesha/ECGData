import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --- HELPER: Frequency Domain Filter (Brick-wall) ---
def fft_brickwall_filter(data_segment, fs, low_cut, high_cut):
    """
    Applies a Brick-wall filter in the Frequency Domain.
    1. Compute FFT of the segment.
    2. Zero out coefficients outside [low_cut, high_cut].
    3. Compute Inverse FFT to recover time-domain signal.
    """
    n = len(data_segment)
    if n == 0: return data_segment
    
    # 1. FFT
    fft_coeffs = np.fft.fft(data_segment)
    frequencies = np.fft.fftfreq(n, d=1/fs)
    
    # 2. Create Mask (Keep frequencies within range)
    # We must keep both positive and negative frequencies for real signal reconstruction
    mask = (np.abs(frequencies) >= low_cut) & (np.abs(frequencies) <= high_cut)
    
    # Apply mask
    fft_filtered = fft_coeffs * mask
    
    # 3. Inverse FFT
    filtered_signal = np.fft.ifft(fft_filtered)
    
    # Return real part (imaginary part should be near zero due to symmetry)
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
def create_full_plot(df, x_range=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['Index'], df['Value'], label='Raw Signal', color='#1f77b4', linewidth=1)
        
    ax.set_title('Raw ECG Signal')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_xlabel('Time')
    ax.grid(True, alpha=0.3)
    
    if x_range:
        ax.set_xlim(x_range)
        
    ax.legend()
    return fig

# --- YOUR ORIGINAL MANUAL DFT FOR PLOTTING ---
@st.cache_data
def calculate_dft(df_segment, fs):
    # 1. Define the target DFT size
    N_dft = 200

    # 2. Get the original signal properties
    original_signal = df_segment['Value'].values
    N_orig = len(original_signal)

    if N_orig < 2:
        return np.array([]), np.array([]), 0

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
    if half_N > 0:
        yf_positive_magnitude[0] = MagDFT[0] / N_orig 

    return xf_positive, yf_positive_magnitude, fs

# --- MAIN APP ---
st.title("ECG Analysis: Segment-Based DFT Filtering")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Choose a csv file", type="csv")

file_to_load = None
if uploaded_file is not None:
    file_to_load = uploaded_file
elif os.path.exists("DataHiMe.csv"):
    file_to_load = "DataHiMe.csv"
    st.sidebar.info("Using default: DataHiMe.csv")

if file_to_load is not None:
    df = load_data(file_to_load)
    
    if df is not None:
        
        # --- CALCULATE SAMPLING RATE ---
        try:
            time_diffs = np.diff(df['Index'])
            fs_est = 1.0 / np.median(time_diffs)
        except:
            fs_est = 100.0
        
        st.sidebar.write(f"**Detected Sampling Rate:** {fs_est:.1f} Hz")
        
        # --- NO GLOBAL FILTERING ---
        # We now work with Raw data and filter specific segments later
        df_processed = df.copy()

        # --- MAIN PREVIEW ---
        st.subheader("1. Raw ECG Data Preview")
        
        min_time = float(df_processed['Index'].min())
        max_time = float(df_processed['Index'].max())
        
        col1, col2 = st.columns([3, 1])
        with col1:
             zoom_range = st.slider(
                "Zoom (Time Range)",
                min_value=min_time,
                max_value=max_time,
                value=(min_time, max_time)
            )
        
        fig_full = create_full_plot(df_processed, zoom_range) 
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
            ax_highlight.plot(df_cycle['Index'], df_cycle['Value'], color='black', alpha=0.6, label='Raw Cycle')
            ax_highlight.axvspan(p_wave_range[0], p_wave_range[1], color='blue', alpha=0.2, label='P Wave')
            ax_highlight.axvspan(qrs_complex_range[0], qrs_complex_range[1], color='red', alpha=0.2, label='QRS')
            ax_highlight.axvspan(t_wave_range[0], t_wave_range[1], color='green', alpha=0.2, label='T Wave')
            ax_highlight.legend()
            st.pyplot(fig_highlight)

            st.markdown("---")
            st.subheader("4. Segment-Based Frequency Domain Filtering")
            st.info("Adjust the Bandpass settings below. The logic computes the FFT of the segment, zeroes out frequencies outside the range (Brick-wall), and reconstructs the signal.")

            # Controls for Filtering
            c_freq1, c_freq2 = st.columns(2)
            with c_freq1:
                low_cut_filter = st.number_input("Low Cutoff (Hz)", value=0.5, step=0.5, min_value=0.0)
            with c_freq2:
                high_cut_filter = st.number_input("High Cutoff (Hz)", value=40.0, step=1.0, max_value=fs_est/2)

            # --- PROCESS SEGMENTS ---
            # 1. P Wave
            p_raw = st.session_state.p_wave_data['Value'].values
            p_filtered = fft_brickwall_filter(p_raw, fs_est, low_cut_filter, high_cut_filter)
            
            # 2. QRS Wave
            qrs_raw = st.session_state.qrs_data['Value'].values
            qrs_filtered = fft_brickwall_filter(qrs_raw, fs_est, low_cut_filter, high_cut_filter)

            # 3. T Wave
            t_raw = st.session_state.t_wave_data['Value'].values
            t_filtered = fft_brickwall_filter(t_raw, fs_est, low_cut_filter, high_cut_filter)

            # --- VISUALIZE RECONSTRUCTION ---
            st.write("### Filtered Segments Reconstruction")
            
            fig_recon, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # P Wave Plot
            ax1.set_title("P Wave")
            if len(p_raw) > 0:
                ax1.plot(st.session_state.p_wave_data['Index'], p_raw, color='lightgray', label='Raw')
                ax1.plot(st.session_state.p_wave_data['Index'], p_filtered, color='blue', label='Filtered (DFT)')
                ax1.legend()
            
            # QRS Plot
            ax2.set_title("QRS Complex")
            if len(qrs_raw) > 0:
                ax2.plot(st.session_state.qrs_data['Index'], qrs_raw, color='lightgray', label='Raw')
                ax2.plot(st.session_state.qrs_data['Index'], qrs_filtered, color='red', label='Filtered (DFT)')
                ax2.legend()

            # T Wave Plot
            ax3.set_title("T Wave")
            if len(t_raw) > 0:
                ax3.plot(st.session_state.t_wave_data['Index'], t_raw, color='lightgray', label='Raw')
                ax3.plot(st.session_state.t_wave_data['Index'], t_filtered, color='green', label='Filtered (DFT)')
                ax3.legend()
            
            st.pyplot(fig_recon)

            # --- SHOW DFT SPECTRUM (Using your original manual calculation function) ---
            st.subheader("DFT Spectrum of Segments (Magnitude)")
            
            xf_p, yf_p, fs_p = calculate_dft(st.session_state.p_wave_data, fs_est)
            xf_qrs, yf_qrs, fs_qrs = calculate_dft(st.session_state.qrs_data, fs_est)
            xf_t, yf_t, fs_t = calculate_dft(st.session_state.t_wave_data, fs_est)

            fig_dft, ax_dft = plt.subplots(figsize=(10, 5))
            if len(xf_p) > 0: ax_dft.plot(xf_p, yf_p, label='P Wave', color='blue', alpha=0.7)
            if len(xf_qrs) > 0: ax_dft.plot(xf_qrs, yf_qrs, label='QRS Complex', color='red', alpha=0.7)
            if len(xf_t) > 0: ax_dft.plot(xf_t, yf_t, label='T Wave', color='green', alpha=0.7)
            
            # Draw cutoff lines to show what was kept
            ax_dft.axvline(low_cut_filter, color='k', linestyle='--', alpha=0.5, label='Cutoff')
            ax_dft.axvline(high_cut_filter, color='k', linestyle='--', alpha=0.5)

            ax_dft.set_xlabel("Frequency (Hz)")
            ax_dft.set_ylabel("Magnitude")
            ax_dft.legend()
            st.pyplot(fig_dft)
