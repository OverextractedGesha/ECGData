import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# -----------------------------------------------------------------------------
# --- 1. MANUAL FILTER FUNCTIONS (TIME DOMAIN - NO LIBRARIES) ---
# -----------------------------------------------------------------------------

def manual_iir_lowpass(data, cutoff, fs):
    """
    Manual implementation of a 1st-order Low Pass Filter using a difference equation.
    Formula equivalent to an RC circuit simulation.
    """
    if cutoff <= 0: return data
    n = len(data)
    y = np.zeros(n)
    
    # Calculate Alpha (smoothing factor) based on RC time constant
    dt = 1.0 / fs
    rc = 1.0 / (2 * np.pi * cutoff)
    alpha = dt / (rc + dt)
    
    # Initial condition
    y[0] = data[0]
    
    # Manual Loop (The "Algorithm")
    for i in range(1, n):
        y[i] = alpha * data[i] + (1 - alpha) * y[i-1]
        
    return y

def manual_iir_highpass(data, cutoff, fs):
    """
    Manual implementation of a 1st-order High Pass Filter.
    """
    if cutoff <= 0: return data
    n = len(data)
    y = np.zeros(n)
    
    # Calculate Alpha
    dt = 1.0 / fs
    rc = 1.0 / (2 * np.pi * cutoff)
    alpha = rc / (rc + dt)
    
    y[0] = 0 
    
    # Manual Loop
    for i in range(1, n):
        y[i] = alpha * (y[i-1] + data[i] - data[i-1])
        
    return y

def manual_bandpass_filter(data, fs, low_cut, high_cut):
    """
    Manual Bandpass = Series of Low Pass + High Pass.
    No FFT, no scipy. Just math loops.
    """
    # 1. Apply Low Pass (Remove High Frequencies, e.g., > 15Hz)
    # Note: We pass 'high_cut' as the limit for the Low Pass Filter
    step1_signal = manual_iir_lowpass(data, high_cut, fs)
    
    # 2. Apply High Pass (Remove Low Frequencies, e.g., < 5Hz)
    # Note: We pass 'low_cut' as the limit for the High Pass Filter
    final_signal = manual_iir_highpass(step1_signal, low_cut, fs)
    
    return final_signal

# --- HELPER: Moving Average Filter ---
def moving_average_filter(data, window_size):
    window_size = int(window_size)
    if window_size < 1: return data
    
    # Manual Convolution for Moving Average
    # (Although np.convolve is standard, here is the manual logic concept)
    window = np.ones(window_size) / float(window_size)
    return np.convolve(data, window, 'same')

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
        ax.plot(df['Index'], df['Value'], label='Pre-Filtered', color='#1f77b4', linewidth=1.2)
    else:
        ax.plot(df['Index'], df['Value'], label='Processed Signal', color='#004cc9', linewidth=1)
        
    ax.set_title('ECG Signal Analysis')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    if x_range: ax.set_xlim(x_range)
    ax.legend()
    return fig

# Helper for Visualization Only (Does not affect filtering process)
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

# -----------------------------------------------------------------------------
# --- MAIN APP ---
# -----------------------------------------------------------------------------
st.title("ECG Analysis: Manual Calculations")
st.write("Implementation: Time-Domain Difference Equations (IIR) instead of FFT.")

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
        
        # --- PRE-FILTERING (OPTIONAL CLEANUP) ---
        st.sidebar.markdown("---")
        st.sidebar.header("2. Pre-Filtering (Optional)")
        
        df_processed = df_raw.copy()
        
        # Global drift removal (Using the same manual function)
        use_hpf = st.sidebar.checkbox("Enable Initial Drift Removal", value=True)
        if use_hpf:
            df_processed['Value'] = manual_iir_highpass(df_processed['Value'].values, 0.5, fs_est)
            
        # --- MAIN PREVIEW ---
        st.subheader("1. Data Preview")
        min_time = float(df_processed['Index'].min())
        max_time = float(df_processed['Index'].max())
        
        col1, col2 = st.columns([3, 1])
        with col1:
             zoom_range = st.slider("Zoom (Time Range)", min_time, max_time, (min_time, max_time))
        
        fig_full = create_full_plot(df_processed, zoom_range, raw_df=df_raw if use_hpf else None) 
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
            st.write("Define Bandpass Frequencies (Pan-Tompkins typically uses 5-15 Hz):")
            c_freq1, c_freq2 = st.columns(2)
            with c_freq1: 
                low_dft = st.number_input("Bandpass Low Cut (Hz)", min_value=0.1, value=5.0, step=0.5)
            with c_freq2: 
                high_dft = st.number_input("Bandpass High Cut (Hz)", min_value=1.0, value=15.0, step=1.0)
            
            # --- PROCESS SEGMENT FILTERS (MANUAL) ---
            p_raw = st.session_state.p_data['Value'].values
            p_filt = manual_bandpass_filter(p_raw, fs_est, low_dft, high_dft)
            
            qrs_raw = st.session_state.qrs_data['Value'].values
            qrs_filt = manual_bandpass_filter(qrs_raw, fs_est, low_dft, high_dft)
            
            t_raw = st.session_state.t_data['Value'].values
            t_filt = manual_bandpass_filter(t_raw, fs_est, low_dft, high_dft)

            # --- PLOT B: SEGMENT RECONSTRUCTION ---
            st.write("#### Result of Manual Bandpass on Segments")
            fig_rec, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            ax1.set_title("P Wave")
            ax1.plot(st.session_state.p_data['Index'], p_raw, color='lightgray', label='Input')
            ax1.plot(st.session_state.p_data['Index'], p_filt, color='blue', label='Filtered')
            ax1.legend()
            
            ax2.set_title("QRS Complex")
            ax2.plot(st.session_state.qrs_data['Index'], qrs_raw, color='lightgray', label='Input')
            ax2.plot(st.session_state.qrs_data['Index'], qrs_filt, color='red', label='Filtered')
            
            ax3.set_title("T Wave")
            ax3.plot(st.session_state.t_data['Index'], t_raw, color='lightgray', label='Input')
            ax3.plot(st.session_state.t_data['Index'], t_filt, color='green', label='Filtered')
            st.pyplot(fig_rec)
            
            # --- DFT PLOT (VISUALIZATION ONLY) ---
            st.write("#### Spectrum Check (Visualization of the Manual Filter)")
            # Compute DFT of the *Filtered* QRS to see if it worked
            xf_raw, yf_raw, _ = calculate_dft(st.session_state.qrs_data, fs_est) # Raw Spectrum
            
            # Make a dummy dataframe for the filtered data to reuse calculate_dft
            df_qrs_filt = st.session_state.qrs_data.copy()
            df_qrs_filt['Value'] = qrs_filt
            xf_filt, yf_filt, _ = calculate_dft(df_qrs_filt, fs_est) # Filtered Spectrum

            fig_dft, ax_dft = plt.subplots(figsize=(10, 4))
            ax_dft.plot(xf_raw, yf_raw, color='lightgray', label='Original Spectrum', linestyle='--')
            ax_dft.plot(xf_filt, yf_filt, color='red', label='Filtered Spectrum (Manual IIR)')
            ax_dft.axvline(low_dft, c='k', ls=':', alpha=0.5)
            ax_dft.axvline(high_dft, c='k', ls=':', alpha=0.5)
            ax_dft.set_title("QRS Spectrum: Before vs After Manual Filter")
            ax_dft.legend()
            st.pyplot(fig_dft)

            st.markdown("---")
            st.subheader("5. Final Output: Manual Bandpass")
            
            # --- APPLY MANUAL FILTER TO GLOBAL DATA ---
            global_signal = df_processed['Value'].values
            
            # This calls the loop-based function defined at the top
            global_filtered = manual_bandpass_filter(global_signal, fs_est, low_dft, high_dft)
            
            df_global_filtered = df_processed.copy()
            df_global_filtered['Value'] = global_filtered
            
            # --- FINAL PLOT CONTROLS ---
            final_min_t = float(df_global_filtered['Index'].min())
            final_max_t = float(df_global_filtered['Index'].max())
            
            final_zoom_range = st.slider(
                "Final Result Zoom", 
                min_value=final_min_t, 
                max_value=final_max_t, 
                value=(final_min_t, final_max_t) 
            )
            
            fig_global_check = create_full_plot(df_global_filtered, x_range=final_zoom_range, raw_df=None)
            st.pyplot(fig_global_check)

            # --- SECTION 6: SQUARING ---
            st.markdown("---")
            st.subheader("6. Squaring Process")
            st.write("**Function**: $y[n] = (x[n])^2$.")

            global_squared = global_filtered ** 2

            fig_sq, ax_sq = plt.subplots(figsize=(10, 6))
            ax_sq.plot(df_global_filtered['Index'], global_squared, color='#800080', label='Squared Signal', linewidth=1.2)
            ax_sq.set_title('Squaring Result')
            ax_sq.set_ylabel('Amplitude ($mV^2$)') 
            ax_sq.set_xlabel('Time (s)')
            ax_sq.grid(True, alpha=0.3)
            ax_sq.set_xlim(final_zoom_range)
            ax_sq.legend()
            st.pyplot(fig_sq)

            # --- SECTION 7: MAV ---
            st.markdown("---")
            st.subheader("7. Moving Window Integration (MAV)")

            # 1. Slider
            window_ms = st.slider("Window Width (ms)", 10, 400, 150, step=10)
            
            # 2. Conversion
            window_samples = int(window_ms * fs_est / 1000.0)
            if window_samples < 1: window_samples = 1
            st.write(f"Window size: {window_ms} ms (~{window_samples} samples)")

            # 3. Apply MAV
            global_mav = moving_average_filter(global_squared, window_samples)

            # 4. Plot MAV
            fig_mav, ax_mav = plt.subplots(figsize=(10, 6))
            ax_mav.plot(df_global_filtered['Index'], global_mav, color='orange', label='MAV Output', linewidth=1.5)
            
            ax_mav.set_title('Moving Window Integration Result')
            ax_mav.set_ylabel('Amplitude (Integrated)')
            ax_mav.set_xlabel('Time (s)')
            ax_mav.grid(True, alpha=0.3)
            ax_mav.set_xlim(final_zoom_range) 
            ax_mav.legend()
            st.pyplot(fig_mav)
