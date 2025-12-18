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
    n = len(data)
    y = np.zeros(n)
    for i in range(n):
        y[i] = data[i] * data[i]
    return y

def manual_moving_average_filter(data, window_size):
    n = len(data)
    window_size = int(window_size)
    if window_size < 1: return data
    
    y = np.zeros(n)
    offset = window_size // 2 
    
    for i in range(n):
        current_sum = 0.0
        
        start_idx = i - offset
        
        for j in range(window_size):
            data_idx = start_idx + j
            
            if 0 <= data_idx < n:
                current_sum += data[data_idx]
        
        y[i] = current_sum / window_size
        
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

def manual_iir_lowpass(data, cutoff, fs):
    if cutoff <= 0: return data
    n = len(data)
    y = np.zeros(n)
    
    dt = 1.0 / fs
    rc = 1.0 / (2 * np.pi * cutoff)
    alpha = dt / (rc + dt)
    
    y[0] = data[0]
    
    for i in range(1, n):
        y[i] = alpha * data[i] + (1 - alpha) * y[i-1]
        
    return y

def manual_iir_highpass(data, cutoff, fs):
    if cutoff <= 0: return data
    n = len(data)
    y = np.zeros(n)
    
    dt = 1.0 / fs
    rc = 1.0 / (2 * np.pi * cutoff)
    alpha = rc / (rc + dt)
    
    y[0] = 0 
    
    for i in range(1, n):
        y[i] = alpha * (y[i-1] + data[i] - data[i-1])
        
    return y

def manual_bandpass_filter(data, fs, low_cut, high_cut):
    step1_signal = manual_iir_lowpass(data, high_cut, fs)
    final_signal = manual_iir_highpass(step1_signal, low_cut, fs)
    return final_signal

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
        ax.plot(df['Index'], df['Value'], label='Processed Signal', color='#004cc9', linewidth=1)
        
    ax.set_title('ECG Signal Analysis')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_xlabel('Time (s)')
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

    mean_val = manual_mean(original_signal)
    x_detrended = original_signal - mean_val
    
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

st.title("ECG Analysis")

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
        try:
            time_diffs = np.diff(df_raw['Index'])
            fs_est = 1.0 / np.median(time_diffs)
        except:
            fs_est = 100.0
        st.sidebar.write(f"**Fs:** {fs_est:.1f} Hz")
        
        st.sidebar.markdown("---")
        st.sidebar.header("2. Pre-Filtering (Global)")
        
        df_processed = df_raw.copy()
        is_filtered = False

        use_hpf = st.sidebar.checkbox("Enable HPF (Remove Drift)", value=True)
        if use_hpf:
            cutoff_h = st.sidebar.slider("HPF Cutoff (Hz)", 0.1, 5.0, 0.5, step=0.1)
            df_processed['Value'] = manual_iir_highpass(df_processed['Value'].values, cutoff_h, fs_est)
            is_filtered = True
            
        use_lpf = st.sidebar.checkbox("Enable LPF (Remove High Freq)", value=True)
        if use_lpf:
            max_cutoff = float(fs_est / 2.0) - 1.0
            cutoff_l = st.sidebar.slider("LPF Cutoff (Hz)", 10.0, max_cutoff, 100.0, step=1.0)
            df_processed['Value'] = manual_iir_lowpass(df_processed['Value'].values, cutoff_l, fs_est)
            is_filtered = True

        raw_for_plot = df_raw if is_filtered else None
            
        st.subheader("Data Preview")
        min_time = float(df_processed['Index'].min())
        max_time = float(df_processed['Index'].max())
        
        col1, col2 = st.columns([3, 1])
        with col1:
              zoom_range = st.slider("Zoom (Time Range)", min_time, max_time, (min_time, max_time))
        
        fig_full = create_full_plot(df_processed, zoom_range, raw_df=raw_for_plot) 
        st.pyplot(fig_full)

        st.subheader("Select a Single ECG Cycle")
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

            st.subheader("Identify Complexes")
            dur = end_index - start_index
            p_range = st.slider("P Wave", start_index, end_index, (start_index, start_index + dur*0.15), step=0.01)
            qrs_range = st.slider("QRS Complex", start_index, end_index, (start_index + dur*0.2, start_index + dur*0.4), step=0.01)
            t_range = st.slider("T Wave", start_index, end_index, (start_index + dur*0.5, start_index + dur*0.8), step=0.01)

            st.session_state.p_data = df_cycle[(df_cycle['Index'] >= p_range[0]) & (df_cycle['Index'] <= p_range[1])]
            st.session_state.qrs_data = df_cycle[(df_cycle['Index'] >= qrs_range[0]) & (df_cycle['Index'] <= qrs_range[1])]
            st.session_state.t_data = df_cycle[(df_cycle['Index'] >= t_range[0]) & (df_cycle['Index'] <= t_range[1])]

            fig_hl, ax_hl = plt.subplots(figsize=(10, 4))
            ax_hl.plot(df_cycle['Index'], df_cycle['Value'], 'k', alpha=0.8)
            ax_hl.axvspan(p_range[0], p_range[1], color='blue', alpha=0.2, label='P')
            ax_hl.axvspan(qrs_range[0], qrs_range[1], color='red', alpha=0.2, label='QRS')
            ax_hl.axvspan(t_range[0], t_range[1], color='green', alpha=0.2, label='T')
            ax_hl.legend()
            st.pyplot(fig_hl)

            st.markdown("---")
            st.subheader("Segment Analysis")
            
            st.write("Define Manual Bandpass Frequencies:")
            c_freq1, c_freq2 = st.columns(2)
            with c_freq1: 
                low_dft = st.number_input("Bandpass Low Cut (Hz)", min_value=0.1, value=5.0, step=0.5)
            with c_freq2: 
                high_dft = st.number_input("Bandpass High Cut (Hz)", min_value=1.0, value=15.0, step=1.0)
            
            xf_p, yf_p, _ = calculate_dft(st.session_state.p_data, fs_est)
            xf_qrs, yf_qrs, _ = calculate_dft(st.session_state.qrs_data, fs_est)
            xf_t, yf_t, _ = calculate_dft(st.session_state.t_data, fs_est)

            st.write("DFT Spectrum")
            fig_dft, ax_dft = plt.subplots(figsize=(10, 5))
            if len(xf_p)>0: ax_dft.plot(xf_p, yf_p, label='P', color='blue')
            if len(xf_qrs)>0: ax_dft.plot(xf_qrs, yf_qrs, label='QRS', color='red')
            if len(xf_t)>0: ax_dft.plot(xf_t, yf_t, label='T', color='green')
            
            ax_dft.axvline(low_dft, c='k', ls='--', alpha=0.5, label='Manual Bandpass Limits')
            ax_dft.axvline(high_dft, c='k', ls='--', alpha=0.5)
            
            ax_dft.set_xlabel("Frequency (Hz)")
            ax_dft.set_ylabel("Magnitude")
            ax_dft.legend()
            st.pyplot(fig_dft)

            p_raw = st.session_state.p_data['Value'].values
            p_filt = manual_bandpass_filter(p_raw, fs_est, low_dft, high_dft)
            qrs_raw = st.session_state.qrs_data['Value'].values
            qrs_filt = manual_bandpass_filter(qrs_raw, fs_est, low_dft, high_dft)
            t_raw = st.session_state.t_data['Value'].values
            t_filt = manual_bandpass_filter(t_raw, fs_est, low_dft, high_dft)

            st.write("Individual Segment Reconstruction")
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
            
            st.markdown("---")
            st.subheader("Manual Bandpass")
            
            global_signal = df_processed['Value'].values
            global_filtered = manual_bandpass_filter(global_signal, fs_est, low_dft, high_dft)
            
            df_global_filtered = df_processed.copy()
            df_global_filtered['Value'] = global_filtered
            
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

            # --- NEW FREQUENCY RESPONSE PLOT USING DFT FUNCTION ---
            st.write(f"**BPF Frequency Response Check ({low_dft}Hz - {high_dft}Hz)**")
            
            # 1. Create Impulse
            # We generate 200 samples to match the calculate_dft function's hardcoded N_dft
            N_imp = 200
            impulse = np.zeros(N_imp)
            impulse[0] = 1.0 
            
            # 2. Run impulse through the filter
            imp_response = manual_bandpass_filter(impulse, fs_est, low_dft, high_dft)
            
            # 3. Create a temporary DataFrame because calculate_dft expects one
            df_imp = pd.DataFrame({'Value': imp_response, 'Index': np.arange(N_imp)/fs_est})
            
            # 4. Use the manual calculate_dft function
            xf_response, yf_response, _ = calculate_dft(df_imp, fs_est)
            
            # 5. Plot
            fig_freq, ax_freq = plt.subplots(figsize=(10, 4))
            ax_freq.plot(xf_response, yf_response, color='purple', linewidth=1.5, label='DFT of Impulse Response')
            ax_freq.set_title(f"Simulated Frequency Response")
            ax_freq.set_xlabel("Frequency (Hz)")
            ax_freq.set_ylabel("Gain (Magnitude)")
            ax_freq.axvline(low_dft, color='k', linestyle='--', alpha=0.5, label='Low Cutoff')
            ax_freq.axvline(high_dft, color='k', linestyle='--', alpha=0.5, label='High Cutoff')
            ax_freq.set_xlim(0, max(high_dft * 2, 50)) 
            ax_freq.grid(True, alpha=0.3)
            ax_freq.legend()
            st.pyplot(fig_freq)

            st.markdown("---")
            st.subheader("Squaring & MAV")

            global_squared = manual_square_signal(global_filtered)

            window_ms = st.slider("MAV Window Width (ms)", 10, 400, 150, step=10)
            window_samples = int(window_ms * fs_est / 1000.0)
            if window_samples < 1: window_samples = 1
            st.write(f"Window size: {window_samples} samples")
            
            global_mav = manual_moving_average_filter(global_squared, window_samples)

            fig_compare, ax_comb = plt.subplots(figsize=(10, 6))
            
            ax_comb.plot(df_global_filtered['Index'], global_squared, color='#800080', label='Squared Signal', alpha=0.3, linewidth=1)
            ax_comb.plot(df_global_filtered['Index'], global_mav, color='orange', label='MAV Output', alpha=1.0, linewidth=2)
            
            ax_comb.set_title('Squaring (Low Opacity) vs MAV (Solid)')
            ax_comb.set_ylabel('Amplitude')
            ax_comb.set_xlabel('Time (s)')
            ax_comb.grid(True, alpha=0.3)
            ax_comb.legend(loc="upper right")
            ax_comb.set_xlim(final_zoom_range)
            
            st.pyplot(fig_compare)

            st.markdown("---")
            st.subheader("Thresholding & BPM Calculation (Segment)")

            st.write("Select the time range to analyze:")
            min_t = float(df_global_filtered['Index'].min())
            max_t = float(df_global_filtered['Index'].max())

            c1, c2 = st.columns(2)
            with c1:
                calc_start = st.number_input("Calculation Start (s)", min_value=min_t, max_value=max_t, value=min_t, step=0.1)
            with c2:
                calc_end = st.number_input("Calculation End (s)", min_value=min_t, max_value=max_t, value=max_t, step=0.1)

            mask = (df_global_filtered['Index'] >= calc_start) & (df_global_filtered['Index'] <= calc_end)
            
            segment_mav = global_mav[mask]
            segment_time = df_global_filtered['Index'][mask]

            if len(segment_mav) > 0:
                max_mav_segment = np.max(segment_mav)
            else:
                max_mav_segment = 0.0

            st.write(f"**Max MAV Amplitude (Segment):** {max_mav_segment:.4f}")

            threshold_perc = st.slider("Set Threshold Level (% of Segment Max)", 0, 100, 40, step=1)
            threshold_val = max_mav_segment * (threshold_perc / 100.0)

            binary_segment, beats_detected = manual_threshold_and_count(segment_mav, threshold_val)
            
            duration_seconds = calc_end - calc_start
            bpm = 0.0
            if duration_seconds > 0:
                bpm = (beats_detected / duration_seconds) * 60.0
            
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Beats Found (Segment)", beats_detected)
            col_res2.metric("Heart Rate (BPM)", f"{bpm:.1f}")

            fig_th, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            ax_top.plot(segment_time, segment_mav, color='orange', label='MAV Segment')
            ax_top.axhline(threshold_val, color='black', linestyle='--', label=f'Threshold')
            ax_top.set_title(f"Analysis Segment ({calc_start}s to {calc_end}s)")
            ax_top.set_ylabel("Amplitude")
            ax_top.legend(loc='upper right')
            ax_top.grid(True, alpha=0.3)

            ax_bot.plot(segment_time, binary_segment, color='red', label='Detected Pulse', drawstyle='steps-pre')
            ax_bot.fill_between(segment_time, binary_segment, step='pre', color='red', alpha=0.3)
            ax_bot.set_ylabel("Logic State (0/1)")
            ax_bot.set_xlabel("Time (s)")
            ax_bot.set_ylim(-0.1, 1.2)
            ax_bot.grid(True, alpha=0.3)
            
            st.pyplot(fig_th)
