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
            segment_prefiltered = df_processed['Value'][mask]

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
            ax_top.set_ylim(bottom=0)

            ax_bg = ax_bot.twinx()
            ax_bg.plot(segment_time, segment_prefiltered, color='gray', alpha=0.35, linewidth=1, label='Pre-filtered ECG')
            ax_bg.set_ylabel("ECG Amplitude (mV)", color='gray')
            ax_bg.tick_params(axis='y', labelcolor='gray')
            
            ax_bot.plot(segment_time, binary_segment, color='red', label='Detected Pulse', drawstyle='steps-pre', zorder=10)
            ax_bot.fill_between(segment_time, binary_segment, step='pre', color='red', alpha=0.2, zorder=10)
            
            ax_bot.set_ylabel("Logic State (0/1)", color='red')
            ax_bot.set_xlabel("Time (s)")
            ax_bot.set_ylim(-0.1, 1.2)
            ax_bot.tick_params(axis='y', labelcolor='red')
            ax_bot.grid(True, alpha=0.3)

            ax_bot.set_zorder(ax_bg.get_zorder() + 1)
            ax_bot.patch.set_visible(False)

            lines_1, labels_1 = ax_bot.get_legend_handles_labels()
            lines_2, labels_2 = ax_bg.get_legend_handles_labels()
            ax_bot.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
            
            st.pyplot(fig_th)
