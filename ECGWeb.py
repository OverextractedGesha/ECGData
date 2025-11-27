import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --- FILTERING FUNCTIONS ---
def apply_mav(data, window_size):
    # Simple Moving Average using convolution
    window = np.ones(window_size) / window_size
    # mode='same' returns output of length max(M, N) - boundary effects are visible at edges
    y = np.convolve(data, window, mode='same')
    return y

@st.cache_data
def load_data(file_path_or_buffer):
    try:
        df = pd.read_csv(file_path_or_buffer, header=None)
        df.columns = ['Index', 'Value']
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

# --- MODIFIED: Added x_range parameter for zooming ---
@st.cache_data
def create_full_plot(df, x_range=None, raw_df=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Raw data faintly if filtered data exists
    if raw_df is not None:
        ax.plot(raw_df['Index'], raw_df['Value'], label='Raw Signal', color='lightgray', alpha=0.7, linewidth=1)
        ax.plot(df['Index'], df['Value'], label='Filtered Signal', color='#1f77b4', linewidth=1.2)
    else:
        ax.plot(df['Index'], df['Value'], label='ECG Signal')
        
    ax.set_title('ECG Signal')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_xlabel('Time')
    ax.grid(True, alpha=0.3)
    
    # Apply zoom if a range is provided
    if x_range:
        ax.set_xlim(x_range)
        
    ax.legend()
    return fig

@st.cache_data
def calculate_dft(df_segment, fs):
    
    # 1. Define the target DFT size
    N_dft = 200

    # 2. Get the original signal properties
    original_signal = df_segment['Value'].values
    N_orig = len(original_signal)

    # 3. Check if the original signal is valid
    if N_orig < 2:
        return np.array([]), np.array([]), 0

    # 4. Detrend the original signal
    x_detrended = original_signal - np.mean(original_signal)
    
    # 5. Create the padded array (200 zeros)
    x_padded = np.zeros(N_dft)
    
    # 6. Copy original data into padded array
    points_to_copy = min(N_orig, N_dft)
    x_padded[0:points_to_copy] = x_detrended[0:points_to_copy]
    
    # 7. Perform DFT on the full 200 points
    x_real = np.zeros(N_dft)
    x_imaj = np.zeros(N_dft)
    
    for k in range (N_dft):
        for n in range (N_dft):
            x_real[k] += x_padded[n]*np.cos(2*np.pi*k*n/N_dft)
            x_imaj[k] -= x_padded[n]*np.sin(2*np.pi*k*n/N_dft)
    
    # 8. Calculate Magnitude
    half_N = round(N_dft/2)
    if half_N == 0:
        return np.array([]), np.array([]), 0

    MagDFT = np.zeros(half_N)

    for k in range (half_N):
        MagDFT[k] = np.sqrt(np.square(x_real[k]) + np.square(x_imaj[k]))
    
    # 9. Create Frequency Axis
    xf_positive = np.arange(0, half_N) * fs / N_dft
    
    # 10. Normalize Amplitude (Divide by N_orig to keep correct scale)
    yf_positive_magnitude = MagDFT * 2.0 / N_orig
    if half_N > 0:
        yf_positive_magnitude[0] = MagDFT[0] / N_orig 

    return xf_positive, yf_positive_magnitude, fs


st.title("ECG Data DFT Analysis")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Choose a csv file", type="csv")

# Logic to determine which file to load
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
        # Robustly calculate fs from the time index
        try:
            time_diffs = np.diff(df['Index'])
            fs_est = 1.0 / np.median(time_diffs)
        except:
            fs_est = 100.0 # Fallback
        
        st.sidebar.write(f"**Detected Sampling Rate:** {fs_est:.1f} Hz")

        # --- FILTERING CONTROLS ---
        st.sidebar.subheader("Noise Filtering")
        
        # Explicit MAV Filter Controls
        apply_mav_filter = st.sidebar.checkbox("Apply Moving Average (MAV)", value=True)
        
        df_processed = df.copy() # Working copy
        raw_df_for_plot = None   # For visualization comparison

        if apply_mav_filter:
            window_size = st.sidebar.slider(
                "Window Size (Points)", 
                min_value=3, 
                max_value=50, 
                value=5, 
                help="Higher values smooth more but may reduce peak heights (QRS)."
            )
            
            try:
                filtered_signal = apply_mav(df['Value'], window_size)
                df_processed['Value'] = filtered_signal
                raw_df_for_plot = df # Save original for comparison plot
                st.sidebar.success(f"MAV (N={window_size}) Active")
            except Exception as e:
                st.sidebar.error(f"Filter Error: {e}")

        # --- MAIN PREVIEW ---
        st.subheader("ECG Data Preview")
        
        # --- ADDED ZOOM FEATURE ---
        min_time = float(df_processed['Index'].min())
        max_time = float(df_processed['Index'].max())
        
        # Create a layout for the zoom controls
        col1, col2 = st.columns([3, 1])
        with col1:
             zoom_range = st.slider(
                "Zoom (Time Range)",
                min_value=min_time,
                max_value=max_time,
                value=(min_time, max_time)
            )
        
        # Pass the zoom range to the plot function
        # We pass df_processed (which might be filtered) and optional raw_df for comparison
        fig_full = create_full_plot(df_processed, zoom_range, raw_df=raw_df_for_plot) 
        st.pyplot(fig_full)

        st.subheader("1. Select a Single ECG Cycle")
        
        # --- DYNAMIC RANGE FIX ---
        min_val = min_time 
        max_val = max_time 
        
        default_duration = (max_val - min_val) * 0.1
        if default_duration == 0: default_duration = 1.0
        
        default_start = min_val
        default_end = min(min_val + default_duration, max_val)
        
        step_size = 0.01 if (max_val - min_val) < 100 else 1.0

        with st.form(key='ecg_cycle_form'):
            start_index_input = st.number_input('Start Time', value=default_start, min_value=min_val, max_value=max_val, step=step_size, format="%.3f")
            end_index_input = st.number_input('End Time', value=default_end, min_value=min_val, max_value=max_val, step=step_size, format="%.3f")
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
                    st.error("Error: Not enough data points selected. Please choose a wider range.")
                    st.session_state.cycle_selected = False
                else:
                    st.session_state.cycle_selected = True
                    st.session_state.start_index = start_index
                    st.session_state.end_index = end_index

        if st.session_state.get('cycle_selected', False):
            
            start_index = st.session_state.start_index
            end_index = st.session_state.end_index
            
            # Use processed (potentially filtered) data here
            df_cycle = df_processed[(df_processed['Index'] >= start_index) & (df_processed['Index'] <= end_index)].copy()

            st.subheader("2. Identify P, QRS, and T Complexes")
            st.write("Use the sliders to highlight the different parts of the ECG cycle.")
            cycle_duration = end_index - start_index
            
            p_wave_range = st.slider(
                "Select P Wave Range",
                min_value=start_index,
                max_value=end_index,
                value=(start_index, start_index + (cycle_duration * 0.15)),
                step=step_size/2
            )
            
            qrs_complex_range = st.slider(
                "Select QRS Complex Range",
                min_value=start_index,
                max_value=end_index,
                value=(start_index + (cycle_duration * 0.2), start_index + (cycle_duration * 0.4)),
                step=step_size/2
            )
            
            t_wave_range = st.slider(
                "Select T Wave Range",
                min_value=start_index,
                max_value=end_index,
                value=(start_index + (cycle_duration * 0.5), start_index + (cycle_duration * 0.8)),
                step=step_size/2
            )

            st.subheader("Highlighted ECG Cycle")
            fig_highlight, ax_highlight = plt.subplots(figsize=(10, 6))
            
            ax_highlight.plot(df_cycle['Index'], df_cycle['Value'], label='Single ECG Cycle', color='black', zorder=10)

            ax_highlight.axvspan(p_wave_range[0], p_wave_range[1], color='blue', alpha=0.3, 
                                 label=f'P Wave ({p_wave_range[1]-p_wave_range[0]:.3f} s)')
            
            ax_highlight.axvspan(qrs_complex_range[0], qrs_complex_range[1], color='red', alpha=0.3, 
                                 label=f'QRS ({qrs_complex_range[1]-qrs_complex_range[0]:.3f} s)')
            
            ax_highlight.axvspan(t_wave_range[0], t_wave_range[1], color='green', alpha=0.3, 
                                 label=f'T Wave ({t_wave_range[1]-t_wave_range[0]:.3f} s)')
            
            ax_highlight.set_title('Highlighted ECG Complexes')
            ax_highlight.set_ylabel('Amplitude (mV)')
            ax_highlight.set_xlabel('Time')
            ax_highlight.set_xlim(start_index, end_index) 
            ax_highlight.grid(True)
            ax_highlight.legend()

            st.pyplot(fig_highlight)

            st.session_state.p_wave_data = df_cycle[(df_cycle['Index'] >= p_wave_range[0]) & (df_cycle['Index'] <= p_wave_range[1])]
            st.session_state.qrs_data = df_cycle[(df_cycle['Index'] >= qrs_complex_range[0]) & (df_cycle['Index'] <= qrs_complex_range[1])]
            st.session_state.t_wave_data = df_cycle[(df_cycle['Index'] >= t_wave_range[0]) & (df_cycle['Index'] <= t_wave_range[1])]

            st.write("---")
            st.write(f"**P Wave Duration:** {p_wave_range[1]-p_wave_range[0]:.3f} s")
            st.write(f"**QRS Complex Duration:** {qrs_complex_range[1]-qrs_complex_range[0]:.3f} s")
            st.write(f"**T Wave Duration:** {t_wave_range[1]-t_wave_range[0]:.3f} s")

            st.subheader("3. DFT Analysis of Segments")
            
            # Pass estimated fs to DFT function
            xf_p, yf_p, fs_p = calculate_dft(st.session_state.p_wave_data, fs_est)
            xf_qrs, yf_qrs, fs_qrs = calculate_dft(st.session_state.qrs_data, fs_est)
            xf_t, yf_t, fs_t = calculate_dft(st.session_state.t_wave_data, fs_est)

            fig_dft, ax_dft = plt.subplots(figsize=(10, 6))
            
            plot_empty = True

            if fs_p > 0 and len(xf_p) > 0:
                ax_dft.plot(xf_p, yf_p, label=f'P Wave (fs={fs_p:.1f} Hz)', color='blue', alpha=0.7)
                plot_empty = False
            
            if fs_qrs > 0 and len(xf_qrs) > 0:
                ax_dft.plot(xf_qrs, yf_qrs, label=f'QRS Complex (fs={fs_qrs:.1f} Hz)', color='red', alpha=0.7)
                plot_empty = False

            if fs_t > 0 and len(xf_t) > 0:
                ax_dft.plot(xf_t, yf_t, label=f'T Wave (fs={fs_t:.1f} Hz)', color='green', alpha=0.7)
                plot_empty = False

            if plot_empty:
                st.write("Not enough data selected in any segment to plot DFT.")
            else:
                ax_dft.set_title("Combined DFT Analysis of ECG Segments (Padded to 200 points)")
                ax_dft.set_xlabel("Frequency (Hz)")
                ax_dft.set_ylabel("Amplitude")
                
                ax_dft.grid(True, which="both", ls="--")
                ax_dft.legend()
                st.pyplot(fig_dft, use_container_width=True)
