import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, header=None)
        df.columns = ['Index', 'Value']
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

@st.cache_data
def create_full_plot(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Index'], df['Value'], label='ECG Signal')
    ax.set_title('ECG Signal from Uploaded File')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_xlabel('Time (ms)')
    ax.grid(True)
    ax.legend()
    return fig


@st.cache_data
def calculate_dft(df_segment):
    
    N_dft = 200

    original_signal = df_segment['Value'].values
    time_ms = df_segment['Index'].values
    N_orig = len(original_signal)

    if N_orig < 2:
        return np.array([]), np.array([]), 0

    fs = 100.0

    
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
    if half_N == 0:
        return np.array([]), np.array([]), 0

    MagDFT = np.zeros(half_N)

    for k in range (half_N):
        MagDFT[k] = np.sqrt(np.square(x_real[k]) + np.square(x_imaj[k]))
    
    xf_positive = np.arange(0, half_N) * fs / N_dft
    
    
    yf_positive_magnitude = MagDFT * 2.0 / N_orig
    if half_N > 0:
        yf_positive_magnitude[0] = MagDFT[0] / N_orig 

    return xf_positive, yf_positive_magnitude, fs


st.title("ECG Data DFT Analysis")
st.subheader("File Upload")
uploaded_file = st.file_uploader("Choose a csv file", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.subheader("ECG Data Preview")
        fig_full = create_full_plot(df) 
        st.pyplot(fig_full)

        st.subheader("1. Select a Single ECG Cycle")
        
        min_val = int(df['Index'].min())
        max_val = int(df['Index'].max())

        form = st.form(key='ecg_cycle_form')
        start_index_input = form.number_input('Start Index (ms)', value=min_val, min_value=min_val, max_value=max_val, step=1)
        end_index_input = form.number_input('End Index (ms)', value=min_val + 500, min_value=min_val, max_value=max_val, step=1)
        submit_button = form.form_submit_button(label='Analyze Cycle')

        if submit_button:
            start_index = int(start_index_input)
            end_index = int(end_index_input)

            if start_index >= end_index:
                st.error("Error: Start Index must be less than End Index.")
                st.session_state.cycle_selected = False
            else:
                df_cycle_check = df[(df['Index'] >= start_index) & (df['Index'] <= end_index)]
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
            
            df_cycle = df[(df['Index'] >= start_index) & (df['Index'] <= end_index)].copy()

            st.subheader("2. Identify P, QRS, and T Complexes")
            st.write("Use the sliders to highlight the different parts of the ECG cycle.")
            cycle_duration = end_index - start_index
            
            p_wave_range = st.slider(
                "Select P Wave Range",
                min_value=start_index,
                max_value=end_index,
                value=(start_index, start_index + int(cycle_duration * 0.15)) 
            )
            
            qrs_complex_range = st.slider(
                "Select QRS Complex Range",
                min_value=start_index,
                max_value=end_index,
                value=(start_index + int(cycle_duration * 0.2), start_index + int(cycle_duration * 0.4))
            )
            
            t_wave_range = st.slider(
                "Select T Wave Range",
                min_value=start_index,
                max_value=end_index,
                value=(start_index + int(cycle_duration * 0.5), start_index + int(cycle_duration * 0.8))
            )

            st.subheader("Highlighted ECG Cycle")
            fig_highlight, ax_highlight = plt.subplots(figsize=(10, 6))
            
            ax_highlight.plot(df_cycle['Index'], df_cycle['Value'], label='Single ECG Cycle', color='black', zorder=10)

            ax_highlight.axvspan(p_wave_range[0], p_wave_range[1], color='blue', alpha=0.3, 
                                 label=f'P Wave ({p_wave_range[1]-p_wave_range[0]} ms)')
            
            ax_highlight.axvspan(qrs_complex_range[0], qrs_complex_range[1], color='red', alpha=0.3, 
                                 label=f'QRS ({qrs_complex_range[1]-qrs_complex_range[0]} ms)')
            
            ax_highlight.axvspan(t_wave_range[0], t_wave_range[1], color='green', alpha=0.3, 
                                 label=f'T Wave ({t_wave_range[1]-t_wave_range[0]} ms)')
            
            ax_highlight.set_title('Highlighted ECG Complexes')
            ax_highlight.set_ylabel('Amplitude (mV)')
            ax_highlight.set_xlabel('Time (ms)')
            ax_highlight.set_xlim(start_index, end_index) 
            ax_highlight.grid(True)
            ax_highlight.legend()

            st.pyplot(fig_highlight)

            st.session_state.p_wave_data = df_cycle[(df_cycle['Index'] >= p_wave_range[0]) & (df_cycle['Index'] <= p_wave_range[1])]
            st.session_state.qrs_data = df_cycle[(df_cycle['Index'] >= qrs_complex_range[0]) & (df_cycle['Index'] <= qrs_complex_range[1])]
            st.session_state.t_wave_data = df_cycle[(df_cycle['Index'] >= t_wave_range[0]) & (df_cycle['Index'] <= t_wave_range[1])]

            st.write("---")
            st.write(f"**P Wave Duration:** {p_wave_range[1]-p_wave_range[0]} ms")
            st.write(f"**QRS Complex Duration:** {qrs_complex_range[1]-qrs_complex_range[0]} ms")
            st.write(f"**T Wave Duration:** {t_wave_range[1]-t_wave_range[0]} ms")

            st.subheader("3. DFT Analysis of Segments")
            
            
            xf_p, yf_p, fs_p = calculate_dft(st.session_state.p_wave_data)
            xf_qrs, yf_qrs, fs_qrs = calculate_dft(st.session_state.qrs_data)
            xf_t, yf_t, fs_t = calculate_dft(st.session_state.t_wave_data)

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
