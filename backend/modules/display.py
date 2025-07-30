import ipywidgets as widgets
from IPython.display import display, Audio, clear_output, HTML
import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import random
import time
import soundfile as sf

from modules import config as cfg
from modules import utilities as u
def annotate(audio_db, black_and_white=False, review_mode="random", bin_target=50):
    """
    Interactive annotation function for audio clips.
    Filters clips based on annotation status and score range.
    Enables selection of clips, audio playback, and annotation.
    
    Args:
        audio_db: An instance of Audio_DB class containing the clips to annotate
        black_and_white: If True, display spectrogram in black and white instead of color
        review_mode: Mode for selecting clips - "random", "sorted" (by score, highest first), or "cde review"
        bin_target: For "cde review" mode, how many clips should be annotated per bin (default: 50)
    """
    # Validate review mode
    if review_mode not in ["random", "sorted", "cde review"]:
        print(f"Invalid review mode: {review_mode}. Using 'random' instead.")
        review_mode = "random"
    
    # Create state variables
    annotation_active = False
    current_index = None
    filtered_df = None
    sorted_history = []  # Keep track of viewed clip indices in sorted mode
    sorted_position = -1  # Position in sorted_history
    viewed_clips = []  # Keep track of clips we've viewed in order (for back button in random mode)
    current_position = -1  # Position in viewed_clips (for back/next navigation in random mode)
    
    # CDE Review mode specific variables
    cde_bins = None
    current_bin = None
    bin_progress = {}
    bin_counts = {}
    
    # Create widgets for filtering
    score_range_slider = widgets.FloatRangeSlider(
        value=[0.0, 1.0],
        min=audio_db.score_min,
        max=audio_db.score_max,
        step=0.01,
        description='Score Range:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='500px')
    )
    
    # Button to apply filters
    filter_button = widgets.Button(
        description='Start Annotating',
        button_style='primary',
        tooltip='Filter clips and start annotation process',
        icon='play'
    )
    
    # Stop annotation button
    stop_button = widgets.Button(
        description='Stop',
        button_style='danger',
        tooltip='Exit annotation mode',
        icon='stop',
        layout=widgets.Layout(width='80px', height='40px')
    )
    
    # Next button
    next_button = widgets.Button(
        description='Next',
        button_style='info',
        tooltip='Skip to next clip without annotating',
        icon='forward',
        layout=widgets.Layout(width='80px', height='40px')
    )
    
    # Back button
    back_button = widgets.Button(
        description='Back',
        button_style='info',
        tooltip='Go back to previous clip',
        icon='backward',
        layout=widgets.Layout(width='80px', height='40px')
    )
    
    # Bin selection dropdown for CDE review mode
    bin_dropdown = widgets.Dropdown(
        options=[],
        description='Bin:',
        disabled=True,
        layout=widgets.Layout(width='200px')
    )
    
    # Output widget to display filter results
    filter_output = widgets.Output()
    
    # Output widget for displaying the spectrogram
    spectrogram_output = widgets.Output()
    
    # Output widget for displaying audio player and clip info
    audio_info_output = widgets.Output()
    
    # Output widget for annotation status
    status_output = widgets.Output()
    
    # Output widget for CDE bin progress
    cde_progress_output = widgets.Output()
    
    # Create annotation buttons
    not_present_button = widgets.Button(
        description='Not Present (0)',
        button_style='danger',
        tooltip='Target sound is not present in the clip',
        layout=widgets.Layout(width='150px', height='50px')
    )
    
    present_button = widgets.Button(
        description='Present (1)',
        button_style='success',
        tooltip='Target sound is present in the clip',
        layout=widgets.Layout(width='150px', height='50px')
    )
    
    uncertain_button = widgets.Button(
        description='Uncertain (3)',
        button_style='warning',
        tooltip='Uncertain if target sound is present',
        layout=widgets.Layout(width='150px', height='50px')
    )
    
    # Create a HBox to layout the annotation buttons
    annotation_buttons = widgets.HBox([
        not_present_button, 
        present_button, 
        uncertain_button
    ], layout=widgets.Layout(
        display='flex',
        flex_flow='row',
        justify_content='flex-end',
        width='100%',
        margin='5px 0px'
    ))
    
    # Function to create mel spectrogram with buffer and markers
    def create_mel_spectrogram(audio_path, clip_start, clip_end, sr=None):
        # Add buffer of up to 1 second on each side (but don't go below 0 for start)
        buffer_samples = 32000  # Buffer in samples
        buffer_s = buffer_samples / cfg.TARGET_SR

        # Load the full audio file
        f = sf.SoundFile(audio_path)
        # Calculate file duration in seconds
        file_duration = f.frames / f.samplerate

        # Calculate buffered indices (ensuring we don't go out of bounds)
        buffered_start = max(0, clip_start - buffer_s)
        buffered_end = min(file_duration, clip_end + buffer_s)
        
        # Extract the buffered audio segment
        y_buffered = u.load_audio(audio_path, (buffered_start, buffered_end, f.samplerate))

        # Create figure 
        plt.figure(figsize=(15, 6))

        # Generate mel spectrogram
        nyquist = cfg.MODEL_SR // 2
        fmax = min(cfg.MAX_FREQ, nyquist)  # Use max model spec freq or Nyquist, whichever is smaller

        # Create the mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y_buffered, 
            sr=cfg.MODEL_SR,  # Always use MODEL_SR for spectrogram
            n_mels=256,
            fmax=fmax,
            hop_length=128
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Display mel spectrogram
        cmap = 'gray_r' if black_and_white else 'viridis'
        img = librosa.display.specshow(
            S_dB, 
            sr=cfg.MODEL_SR,
            x_axis='time', 
            y_axis='mel', 
            fmax=fmax, 
            x_coords=np.linspace(buffered_start, buffered_end, S.shape[1]),
            cmap=cmap
        )

        # Add colorbar
        if black_and_white:
            plt.colorbar(img, format='%+2.0f dB', shrink=0.7)
        else:
            plt.colorbar(img, format='%+2.0f dB')
            
        plt.xlabel("Time (seconds)")

        # Add vertical lines for clip boundaries
        plt.axvline(x=clip_start, color='r', linestyle='-', linewidth=2, alpha=0.7)
        plt.axvline(x=clip_end, color='r', linestyle='-', linewidth=2, alpha=0.7)

        # Add text labels
        plt.text(clip_start, 0, f"{clip_start:.1f}s", color='r', fontweight='bold', 
                    verticalalignment='bottom', horizontalalignment='center')
        plt.text(clip_end, 0, f"{clip_end:.1f}s", color='r', fontweight='bold', 
                    verticalalignment='bottom', horizontalalignment='center')

        # Set title
        plt.title(f'Clip: {clip_start:.1f}s - {clip_end:.1f}s (Duration: {clip_end-clip_start:.1f}s)')

        # Use tight layout
        plt.tight_layout(pad=0.5)

        # Extract the exact clip audio for playback (without buffer)
        # Convert from time positions to buffer-relative sample positions
        clip_start_rel_sample = int((clip_start - buffered_start) * cfg.TARGET_SR)
        clip_end_rel_sample = int((clip_end - buffered_start) * cfg.TARGET_SR)

        # Ensure indices are within bounds
        clip_start_rel_sample = max(0, min(clip_start_rel_sample, len(y_buffered) - 1))
        clip_end_rel_sample = max(clip_start_rel_sample + 1, min(clip_end_rel_sample, len(y_buffered)))

        clip_audio = y_buffered[clip_start_rel_sample:clip_end_rel_sample]

        return clip_audio, cfg.TARGET_SR, y_buffered, buffered_start, buffered_end

    # Function to identify CDE bins in the audio database
    def identify_cde_bins():
        nonlocal cde_bins, bin_progress, bin_counts
        
        if review_mode != "cde review":
            return
        
        # Identify bins based on directory structure
        # We expect 4 subfolders in the audio folder
        bin_folders = set()
        
        # Extract parent folder from each file path
        for row in audio_db.df.iter_rows():
            file_path = row[1]  # file_path column
            
            # Get the parent folder name (which should be the bin)
            parent_folder = os.path.basename(os.path.dirname(file_path))
            bin_folders.add(parent_folder)
        
        # Create a dictionary to store file paths by bin
        cde_bins = {}
        bin_progress = {}
        bin_counts = {}
        
        # Count how many clips in each bin are already annotated and how many remain
        for bin_name in bin_folders:
            # Filter for clips in this bin that haven't been annotated yet
            bin_clips = audio_db.df.filter(
                (pl.col("file_path").str.contains(f"/{bin_name}/")) & 
                (pl.col("annotation") == 4)  # Not yet annotated
            )
            
            # Count clips that have already been annotated in this bin with definitive labels (0 or 1)
            # We don't count uncertain (3) annotations toward our bin target
            annotated_clips = audio_db.df.filter(
                (pl.col("file_path").str.contains(f"/{bin_name}/")) & 
                ((pl.col("annotation") == 0) | (pl.col("annotation") == 1))  # Only count definitive annotations
            )
            
            # Get all clips in this bin for other metrics
            all_clips_in_bin = audio_db.df.filter(
                (pl.col("file_path").str.contains(f"/{bin_name}/"))
            )
            
            cde_bins[bin_name] = bin_clips
            bin_progress[bin_name] = len(annotated_clips)  # Only count 0 or 1 annotations
            bin_counts[bin_name] = len(all_clips_in_bin)  # Total clips in this bin
        
        # Update the bin dropdown with the found bins
        bin_options = [(f"{b} ({bin_progress[b]}/{min(bin_target, bin_counts[b])} done)", b) for b in sorted(cde_bins.keys())]
        bin_dropdown.options = bin_options
        
        if bin_options:
            bin_dropdown.value = bin_options[0][1]  # Select the first bin by default
            bin_dropdown.disabled = False
        
        return cde_bins
    
    # Function to select a bin for CDE review
    def select_cde_bin(change):
        nonlocal current_bin, filtered_df
        
        if change.new and review_mode == "cde review":
            current_bin = change.new
            filtered_df = cde_bins[current_bin]
            
            with filter_output:
                clear_output(wait=True)
                print(f"Selected bin: {current_bin}")
                print(f"Clips remaining to annotate: {len(filtered_df)}")
                print(f"Progress: {bin_progress[current_bin]}/{min(bin_target, bin_counts[current_bin])} clips")
            
            # Update progress display
            update_cde_progress()
    
    # Update the CDE bin progress display
    def update_cde_progress():
        if review_mode != "cde review":
            return
            
        with cde_progress_output:
            clear_output(wait=True)
            
            print("CDE Review Progress:")
            for bin_name in sorted(cde_bins.keys()):
                progress = bin_progress[bin_name]
                target = min(bin_target, bin_counts[bin_name])
                status = "✓" if progress >= target else " "
                
                # Also get counts for uncertain annotations (for information purposes)
                uncertain_clips = audio_db.df.filter(
                    (pl.col("file_path").str.contains(f"/{bin_name}/")) & 
                    (pl.col("annotation") == 3)  # Uncertain annotations
                )
                uncertain_count = len(uncertain_clips)
                
                # Display the progress, including uncertain count for information
                print(f"[{status}] {bin_name}: {progress}/{target} definitive labels ({progress/target*100:.1f}%)")
                if uncertain_count > 0:
                    print(f"    + {uncertain_count} uncertain annotations (not counted toward target)")
    
    # Connect bin dropdown to the selection function
    bin_dropdown.observe(select_cde_bin, names='value')
    
    # Function to refresh filtered data
    def refresh_filtered_data():
        nonlocal filtered_df
        
        if review_mode == "cde review":
            # For CDE review mode, filtered_df will be the clips from the current bin
            if current_bin and current_bin in cde_bins:
                # Re-filter to get only clips that still need annotation in this bin
                bin_clips = audio_db.df.filter(
                    (pl.col("file_path").str.contains(f"/{current_bin}/")) & 
                    (pl.col("annotation") == 4)  # Not yet annotated
                )
                filtered_df = bin_clips
                
                # Update bin progress display - only count definitive annotations (0 or 1)
                # We don't count uncertain (3) annotations toward our bin target
                annotated_clips = audio_db.df.filter(
                    (pl.col("file_path").str.contains(f"/{current_bin}/")) & 
                    ((pl.col("annotation") == 0) | (pl.col("annotation") == 1))  # Only count definitive annotations
                )
                bin_progress[current_bin] = len(annotated_clips)
                
                # Update dropdown options to reflect progress
                bin_options = [(f"{b} ({bin_progress[b]}/{min(bin_target, bin_counts[b])} done)", b) for b in sorted(cde_bins.keys())]
                bin_dropdown.options = bin_options
                
                # Update progress display
                update_cde_progress()
        else:
            # For other modes, use the score range slider
            min_score, max_score = score_range_slider.value
            
            # Re-filter to get only clips that still need annotation
            filtered_df = audio_db.df.filter(
                (pl.col("annotation") == 4) & 
                (pl.col("score") >= min_score) & 
                (pl.col("score") <= max_score)
            )
            
            # If in sorted mode, sort the filtered DataFrame by score (descending)
            if review_mode == "sorted":
                filtered_df = filtered_df.sort("score", descending=True)
        
        return filtered_df
    
    # Function to find a clip's unique identifier
    def get_clip_identifier(clip):
        # Create a unique identifier based on file path and time range
        return f"{clip[1]}_{clip[3]}_{clip[4]}"  # file_path_start_end
    
    # Function to check if a clip is available in the filtered data
    def is_clip_available(clip_id):
        if filtered_df is None or len(filtered_df) == 0:
            return False, None
        
        for idx, row in enumerate(filtered_df.iter_rows()):
            current_id = get_clip_identifier(row)
            if current_id == clip_id:
                return True, idx
                
        return False, None
    
    # Function to get the next clip based on review mode
    def get_next_clip():
        nonlocal sorted_position, current_bin, filtered_df  # Added filtered_df to nonlocal
        
        if filtered_df is None or len(filtered_df) == 0:
            # Check if we need to switch bins in CDE review mode
            if review_mode == "cde review":
                # Find the next bin that needs more annotations
                for bin_name in sorted(cde_bins.keys()):
                    if bin_progress[bin_name] < min(bin_target, bin_counts[bin_name]):
                        # Switch to this bin if it's not the current one
                        if bin_name != current_bin:
                            current_bin = bin_name
                            bin_dropdown.value = bin_name  # This will trigger select_cde_bin
                            
                            # Check if the new bin has clips
                            if len(cde_bins[bin_name]) > 0:
                                filtered_df = cde_bins[bin_name]
                                random_idx = random.randint(0, len(filtered_df) - 1)
                                return filtered_df.row(random_idx), random_idx
            
            # If we get here, either it's not CDE review mode or we've exhausted all bins
            return None, None
        
        if review_mode == "random" or review_mode == "cde review":
            # Select a random index from the filtered data
            random_idx = random.randint(0, len(filtered_df) - 1)
            return filtered_df.row(random_idx), random_idx
        else:  # sorted mode
            if sorted_position < len(filtered_df) - 1:
                sorted_position += 1
                return filtered_df.row(sorted_position), sorted_position
            else:
                return None, None  # No more clips in order
    
    # Function to get a specific clip by index
    def get_clip_by_index(idx):
        nonlocal filtered_df  # Added filtered_df to nonlocal
        
        if filtered_df is None or idx is None or idx >= len(filtered_df) or idx < 0:
            return None, None
        return filtered_df.row(idx), idx
    
    # Function to display audio and spectrogram
    def display_clip(clip_data):
        # Extract clip details
        file_name = clip_data[0]  # file_name
        file_path = clip_data[1]  # file_path
        clip_start = clip_data[3]  # clip_start
        clip_end = clip_data[4]   # clip_end
        sampling_rate = clip_data[5]  # sampling_rate
        score = clip_data[6]  # score
        
        # Display spectrogram in its output area
        with spectrogram_output:
            clear_output(wait=True)
            
            try:
                # Create spectrogram with buffer and markers, and get audio data
                clip_audio, sr, buffered_audio, buffered_start, buffered_end = create_mel_spectrogram(
                    file_path, clip_start, clip_end, sr=sampling_rate
                )
                plt.show()
            except Exception as e:
                print(f"Error generating spectrogram: {e}")
                print(f"File path: {file_path}")
        
        # Display audio player and clip info in their output area
        with audio_info_output:
            clear_output(wait=True)
            
            try:
                # Create audio widget for playback with autoplay
                display(HTML("<h4 style='margin-top:0;margin-bottom:5px;'>Audio:</h4>"))
                display(Audio(data=clip_audio, rate=sr, autoplay=True))
                
                # Display clip details with reduced margins
                position_info = ""
                if review_mode == "sorted":
                    position_info = f"<p style='margin:2px 0;'><b>Position:</b> {sorted_position + 1} of {len(filtered_df)}</p>"
                elif review_mode == "cde review":
                    position_info = f"<p style='margin:2px 0;'><b>Bin:</b> {current_bin}</p>"
                    position_info += f"<p style='margin:2px 0;'><b>Progress:</b> {bin_progress[current_bin]}/{min(bin_target, bin_counts[current_bin])}</p>"
                
                # Extract bin info from the file path for display in CDE mode
                bin_info = ""
                if review_mode == "cde review":
                    bin_name = os.path.basename(os.path.dirname(file_path))
                    bin_info = f"<p style='margin:2px 0;'><b>Bin:</b> {bin_name}</p>"
                
                clip_info = f"""
                <div style="margin: 5px 0;">
                    <h4 style='margin-top:5px;margin-bottom:5px;'>Info:</h4>
                    <p style='margin:2px 0;'><b>File:</b> {os.path.basename(file_path)}</p>
                    <p style='margin:2px 0;'><b>Time:</b> {clip_start:.1f}s - {clip_end:.1f}s</p>
                    <p style='margin:2px 0;'><b>Duration:</b> {clip_end - clip_start:.1f}s</p>
                    <p style='margin:2px 0;'><b>Rate:</b> {sampling_rate} Hz</p>
                    <p style='margin:2px 0;'><b>Score:</b> {score:.3f}</p>
                    {bin_info}
                    {position_info}
                </div>
                """
                display(HTML(clip_info))
                
            except Exception as e:
                print(f"Error with audio playback: {e}")
                
                # Try alternative approach
                try:
                    print("\nAttempting alternative loading method...")
                    full_audio, sr = librosa.load(file_path, sr=sampling_rate)
                    start_idx = int(clip_start * sr)
                    end_idx = int(clip_end * sr)
                    
                    # Ensure indices are within bounds
                    start_idx = max(0, min(start_idx, len(full_audio) - 1))
                    end_idx = max(start_idx + 1, min(end_idx, len(full_audio)))
                    
                    clip_audio = full_audio[start_idx:end_idx]
                    
                    # Create audio widget for playback
                    display(Audio(data=clip_audio, rate=sr, autoplay=True))
                    print("Audio loaded using alternative method.")
                except Exception as e2:
                    print(f"Alternative method also failed: {e2}")
                    print("Please check the file path and format.")
    
    # Function to update annotation in the dataframe
    def update_annotation(annotation_value):
        nonlocal current_index, filtered_df, annotation_active, current_bin
        
        # For CDE review mode, we need special handling to ensure uncertain annotations
        # don't count toward the bin_target
        
        if current_index is None or filtered_df is None:
            return
        
        # Get the current clip's file_name for identification
        current_clip = filtered_df.row(current_index)
        file_name = current_clip[0]  # file_name
        file_path = current_clip[1]  # file_path
        clip_start = current_clip[3]  # clip_start
        clip_end = current_clip[4]  # clip_end
        
        # Update the annotation in the Audio_DB dataframe
        # We need to find the matching row and update its annotation
        mask = (audio_db.df["file_name"] == file_name) & \
               (audio_db.df["file_path"] == file_path) & \
               (audio_db.df["clip_start"] == clip_start) & \
               (audio_db.df["clip_end"] == clip_end)
        
        # Create a new column that will have the updated annotation value where the mask is True
        update_col = pl.when(mask).then(annotation_value).otherwise(pl.col("annotation"))
        
        # Update the dataframe
        audio_db.df = audio_db.df.with_columns(update_col.alias("annotation"))
        
        # Important: Re-filter the data to exclude the just-annotated clip
        refresh_filtered_data()
        
        with status_output:
            clear_output(wait=True)
            print(f"✓ Clip annotated as: {annotation_value_to_text(annotation_value)}")
            
            if review_mode == "cde review":
                # Add information about the type of annotation (important for uncertain case)
                if annotation_value == 3:  # Uncertain
                    print("Note: Uncertain annotations don't count toward bin target")
                
                # Check if we've reached the target for the current bin
                current_target = min(bin_target, bin_counts[current_bin])
                
                # Count uncertain annotations for this bin (for information purposes)
                uncertain_clips = audio_db.df.filter(
                    (pl.col("file_path").str.contains(f"/{current_bin}/")) & 
                    (pl.col("annotation") == 3)  # Uncertain annotations
                )
                uncertain_count = len(uncertain_clips)
                
                if bin_progress[current_bin] >= current_target:
                    print(f"✓ Target reached for bin {current_bin}: {bin_progress[current_bin]}/{current_target} definitive labels")
                    if uncertain_count > 0:
                        print(f"  + {uncertain_count} uncertain annotations (not counted toward target)")
                    
                    # Check if we're done with all bins
                    all_done = True
                    for bin_name in cde_bins:
                        if bin_progress[bin_name] < min(bin_target, bin_counts[bin_name]):
                            all_done = False
                            break
                            
                    if all_done:
                        print("✓ All bins have reached their targets!")
                    else:
                        # Find the next bin that needs more annotations
                        for bin_name in sorted(cde_bins.keys()):
                            if bin_progress[bin_name] < min(bin_target, bin_counts[bin_name]):
                                print(f"Switching to bin {bin_name} which needs more annotations.")
                                current_bin = bin_name
                                bin_dropdown.value = bin_name  # This will trigger select_cde_bin
                                break
                else:
                    print(f"Remaining clips in bin {current_bin}: {len(filtered_df)}")
                    print(f"Progress: {bin_progress[current_bin]}/{current_target} definitive labels")
                    if uncertain_count > 0:
                        print(f"  + {uncertain_count} uncertain annotations (not counted toward target)")
            else:
                print(f"Remaining clips: {len(filtered_df)}")
                if review_mode == "sorted":
                    print(f"Mode: Sorted (by score, highest first)")
                else:
                    print(f"Mode: Random selection")
        
        # For CDE review mode, update the progress display
        if review_mode == "cde review":
            update_cde_progress()
        
        # If we still have clips to annotate, load the next clip
        if (review_mode != "cde review" and len(filtered_df) > 0 and annotation_active) or \
           (review_mode == "cde review" and annotation_active):
            # For CDE review, if the current bin is done, we should try to switch to the next bin
            if review_mode == "cde review":
                current_target = min(bin_target, bin_counts[current_bin])
                if bin_progress[current_bin] >= current_target:
                    # Find the next bin that needs more annotations
                    next_bin_found = False
                    for bin_name in sorted(cde_bins.keys()):
                        if bin_progress[bin_name] < min(bin_target, bin_counts[bin_name]):
                            print(f"Switching to bin {bin_name} which needs more annotations.")
                            current_bin = bin_name
                            bin_dropdown.value = bin_name  # This will trigger select_cde_bin
                            next_bin_found = True
                            break
                    
                    # If no next bin is found, we're done with all bins
                    if not next_bin_found:
                        annotation_active = False
                        with status_output:
                            clear_output(wait=True)
                            print("✓ All bins have reached their targets!")
                            print("Click 'Start Annotating' to begin a new annotation session.")
                        
                        # Clear the displays when done
                        with spectrogram_output:
                            clear_output(wait=True)
                        with audio_info_output:
                            clear_output(wait=True)
                        return
            
            load_next_clip()
        else:
            # If there are no more clips in any bin for CDE review mode
            if review_mode == "cde review":
                all_done = True
                for bin_name in cde_bins:
                    if bin_progress[bin_name] < min(bin_target, bin_counts[bin_name]) and len(cde_bins[bin_name]) > 0:
                        all_done = False
                        break
                        
                if all_done:
                    annotation_active = False
                    with status_output:
                        clear_output(wait=True)
                        print("✓ All bins have reached their targets or are out of clips!")
                        print("Click 'Start Annotating' to begin a new annotation session.")
                    
                    # Clear the displays when done
                    with spectrogram_output:
                        clear_output(wait=True)
                    with audio_info_output:
                        clear_output(wait=True)
            else:
                annotation_active = False
                with status_output:
                    clear_output(wait=True)
                    print("✓ All clips have been annotated in the selected range!")
                    print("Click 'Start Annotating' to select a new range or continue with additional clips.")
                
                # Clear the displays when done
                with spectrogram_output:
                    clear_output(wait=True)
                with audio_info_output:
                    clear_output(wait=True)
    
    # Helper function to convert annotation value to text
    def annotation_value_to_text(value):
        if value == 0:
            return "Not Present"
        elif value == 1:
            return "Present"
        elif value == 3:
            return "Uncertain"
        else:
            return f"Unknown ({value})"
    
    # Function to handle going to the next clip
    def next_clip(b):
        nonlocal current_index, filtered_df, annotation_active, current_position, sorted_position, cde_bins
        
        if filtered_df is None or not annotation_active:
            return
        
        with status_output:
            clear_output(wait=True)
            
            if review_mode == "sorted":
                if sorted_position < len(filtered_df) - 1:
                    sorted_position += 1
                    current_index = sorted_position
                    clip_data, _ = get_clip_by_index(current_index)
                    if clip_data is not None:
                        display_clip(clip_data)
                        print(f"▶️ Next clip ({sorted_position + 1} of {len(filtered_df)})")
                    else:
                        print("⚠️ Could not load next clip.")
                else:
                    print("⚠️ No more clips available.")
            else:  # random or cde review mode
                print("▶️ Next clip.")
                load_next_clip()
            
            if review_mode == "cde review":
                current_target = min(bin_target, bin_counts[current_bin])
                
                # Count uncertain annotations for this bin (for information purposes)
                uncertain_clips = audio_db.df.filter(
                    (pl.col("file_path").str.contains(f"/{current_bin}/")) & 
                    (pl.col("annotation") == 3)  # Uncertain annotations
                )
                uncertain_count = len(uncertain_clips)
                
                print(f"Bin {current_bin}: {bin_progress[current_bin]}/{current_target} definitive labels")
                if uncertain_count > 0:
                    print(f"  + {uncertain_count} uncertain annotations (not counted toward target)")
            else:
                print(f"Remaining clips: {len(filtered_df)}")
                if review_mode == "sorted":
                    print(f"Mode: Sorted (by score, highest first)")
                else:
                    print(f"Mode: Random selection")
    
    # Function to handle going back to the previous clip
    def back_clip(b):
        nonlocal current_index, filtered_df, annotation_active, current_position, sorted_position, viewed_clips, current_bin, cde_bins
        
        if filtered_df is None or not annotation_active:
            return
        
        with status_output:
            clear_output(wait=True)
            
            if review_mode == "sorted":
                if sorted_position > 0:
                    sorted_position -= 1
                    current_index = sorted_position
                    clip_data, _ = get_clip_by_index(current_index)
                    if clip_data is not None:
                        display_clip(clip_data)
                        print(f"◀️ Previous clip ({sorted_position + 1} of {len(filtered_df)})")
                    else:
                        print("⚠️ Could not load previous clip.")
                else:
                    print("⚠️ At the beginning of the list, cannot go back further.")
            else:  # random or cde review mode
                # Both random and CDE review modes use the viewed_clips history
                if current_position <= 0:
                    print("⚠️ No previous clips available.")
                    return
                
                # Go back to the previous clip
                current_position -= 1
                previous_clip_info = viewed_clips[current_position]
                
                # For CDE review mode, we need to check if the previous clip is in a different bin
                previous_bin = None
                if review_mode == "cde review":
                    previous_bin = os.path.basename(os.path.dirname(previous_clip_info['file_path']))
                    
                    # If we're in a different bin than the previous clip, switch to that bin first
                    if previous_bin != current_bin:
                        print(f"Switching to previous bin: {previous_bin}")
                        current_bin = previous_bin
                        bin_dropdown.value = previous_bin  # This will trigger select_cde_bin and update filtered_df
                        
                        # Give time for the bin switch to complete
                        time.sleep(0.1)
                        
                        # Refresh the filtered data for the new bin
                        filtered_df = refresh_filtered_data()
                
                # Find this clip in the current filtered data
                clip_found = False
                for idx, row in enumerate(filtered_df.iter_rows()):
                    file_path = row[1]
                    clip_start = row[3]
                    clip_end = row[4]
                    
                    if (file_path == previous_clip_info['file_path'] and 
                        clip_start == previous_clip_info['clip_start'] and 
                        clip_end == previous_clip_info['clip_end']):
                        current_index = idx
                        clip_found = True
                        break
                
                if clip_found:
                    # Display the previous clip
                    clip_data = filtered_df.row(current_index)
                    display_clip(clip_data)
                    print("◀️ Previous clip.")
                else:
                    # Try to find the clip in the full audio_db
                    full_clip_found = False
                    previous_clip_row = None
                    
                    for row in audio_db.df.iter_rows():
                        file_path = row[1]
                        clip_start = row[3]
                        clip_end = row[4]
                        
                        if (file_path == previous_clip_info['file_path'] and 
                            clip_start == previous_clip_info['clip_start'] and 
                            clip_end == previous_clip_info['clip_end']):
                            previous_clip_row = row
                            full_clip_found = True
                            break
                    
                    if full_clip_found and previous_clip_row:
                        # The clip exists but is no longer in filtered_df (probably because it was annotated)
                        # Display it anyway for reference
                        display_clip(previous_clip_row)
                        print("◀️ Previous clip (already annotated, view only).")
                        print("Note: This clip has already been annotated and is shown for reference only.")
                    else:
                        # If clip not found anywhere, remove it from history
                        viewed_clips.pop(current_position)
                        current_position = max(0, current_position - 1)
                        print("⚠️ Previous clip no longer available.")
                        print("Going back further...")
                        
                        # Try to go back again recursively
                        back_clip(None)
                        return
                    
        # Display appropriate information based on review mode
        if review_mode == "cde review":
            current_target = min(bin_target, bin_counts[current_bin])
            
            # Count uncertain annotations for this bin (for information purposes)
            uncertain_clips = audio_db.df.filter(
                (pl.col("file_path").str.contains(f"/{current_bin}/")) & 
                (pl.col("annotation") == 3)  # Uncertain annotations
            )
            uncertain_count = len(uncertain_clips)
            
            print(f"Bin {current_bin}: {bin_progress[current_bin]}/{current_target} definitive labels")
            if uncertain_count > 0:
                print(f"  + {uncertain_count} uncertain annotations (not counted toward target)")
            print(f"Mode: CDE Review (bin target: {bin_target} definitive labels)")
        else:
            print(f"Remaining clips: {len(filtered_df)}")
            if review_mode == "sorted":
                print(f"Mode: Sorted (by score, highest first)")
            else:
                print(f"Mode: Random selection")
    
    # Function to load the next clip
    def load_next_clip():
        nonlocal current_index, filtered_df, viewed_clips, current_position, sorted_position, cde_bins, current_bin
        
        # Make sure we're working with the most up-to-date filtered data
        refresh_filtered_data()
        
        if filtered_df is None or len(filtered_df) == 0:
            with status_output:
                clear_output(wait=True)
                
                if review_mode == "cde review":
                    # For CDE review mode, check if we should try another bin
                    all_done = True
                    for bin_name in cde_bins:
                        if bin_progress[bin_name] < min(bin_target, bin_counts[bin_name]):
                            all_done = False
                            
                            # Switch to this bin if it's not the current one
                            if bin_name != current_bin:
                                print(f"Switching to bin {bin_name} which needs more annotations.")
                                current_bin = bin_name
                                bin_dropdown.value = bin_name  # This will trigger select_cde_bin
                                return
                                
                    if all_done:
                        print("✓ All bins have reached their targets or have no more clips!")
                        print("Click 'Start Annotating' to begin a new annotation session.")
                else:
                    print("No clips match the filter criteria or all clips have been annotated.")
            return
        
        # Get the next clip based on review mode
        if review_mode == "random" or review_mode == "cde review":
            # Random mode - select random clip
            clip_data, clip_idx = get_next_clip()
            
            if clip_data is None:
                with status_output:
                    clear_output(wait=True)
                    
                    if review_mode == "cde review":
                        # Check other bins
                        print("No more clips in the current bin.")
                        update_cde_progress()
                        
                        # Try to find another bin that needs annotations
                        for bin_name in sorted(cde_bins.keys()):
                            if bin_progress[bin_name] < min(bin_target, bin_counts[bin_name]) and len(cde_bins[bin_name]) > 0:
                                print(f"Switching to bin {bin_name} which needs more annotations.")
                                current_bin = bin_name
                                bin_dropdown.value = bin_name  # This will trigger select_cde_bin
                                return
                        
                        print("All bins have reached their targets or have no more clips!")
                    else:
                        print("No clips remain to be annotated in the selected range.")
                return
                
            current_index = clip_idx
            
            # Store this clip in history for random mode
            clip_info = {
                'file_path': clip_data[1],
                'clip_start': clip_data[3],
                'clip_end': clip_data[4]
            }
            
            # If we're navigating back and then forward, truncate the history
            if current_position < len(viewed_clips) - 1:
                viewed_clips = viewed_clips[:current_position + 1]
                
            viewed_clips.append(clip_info)
            current_position = len(viewed_clips) - 1
        else:
            # Sorted mode - get clip at current position
            clip_data, clip_idx = get_clip_by_index(sorted_position)
            
            if clip_data is None:
                with status_output:
                    clear_output(wait=True)
                    print("No clips remain to be annotated in the selected range.")
                return
                
            current_index = clip_idx
        
        # Display the clip
        display_clip(clip_data)
        
        with status_output:
            clear_output(wait=True)
            
            if review_mode == "sorted":
                print(f"Please annotate this clip ({sorted_position + 1} of {len(filtered_df)})")
            elif review_mode == "cde review":
                # Count uncertain annotations for this bin (for information purposes)
                uncertain_clips = audio_db.df.filter(
                    (pl.col("file_path").str.contains(f"/{current_bin}/")) & 
                    (pl.col("annotation") == 3)  # Uncertain annotations
                )
                uncertain_count = len(uncertain_clips)
                
                current_target = min(bin_target, bin_counts[current_bin])
                print(f"Please annotate this clip - Bin {current_bin}")
                print(f"Progress: {bin_progress[current_bin]}/{current_target} definitive labels")
                if uncertain_count > 0:
                    print(f"  + {uncertain_count} uncertain annotations (not counted toward target)")
                print("Note: Uncertain (3) annotations don't count toward the target")
            else:
                print(f"Please annotate this clip ({len(filtered_df)} remaining)")
                
            if review_mode == "sorted":
                print(f"Mode: Sorted (by score, highest first)")
            elif review_mode == "cde review":
                print(f"Mode: CDE Review (bin target: {bin_target} definitive labels)")
            else:
                print(f"Mode: Random selection")
    
    # Function to handle filtering and start the annotation process
    def start_annotation(b):
        nonlocal annotation_active, filtered_df, viewed_clips, current_position, sorted_position, cde_bins, current_bin
        
        # Reset history when starting a new annotation session
        viewed_clips = []
        current_position = -1
        sorted_position = 0  # Start at first item for sorted mode
        
        with filter_output:
            clear_output(wait=True)
            
            # For CDE review mode, identify bins first
            if review_mode == "cde review":
                print("Identifying CDE review bins...")
                cde_bins = identify_cde_bins()
                
                if cde_bins:
                    all_done = True
                    for bin_name in cde_bins:
                        if bin_progress[bin_name] < min(bin_target, bin_counts[bin_name]):
                            all_done = False
                            break
                            
                    if all_done:
                        print("✓ All bins have already reached their targets!")
                        print("If you want to annotate more clips, please increase the bin_target value.")
                        annotation_active = False
                        return
                        
                    # Find first bin that needs more annotations and select it
                    for bin_name in sorted(cde_bins.keys()):
                        if bin_progress[bin_name] < min(bin_target, bin_counts[bin_name]):
                            print(f"Starting with bin {bin_name} which needs more annotations.")
                            bin_dropdown.value = bin_name  # This will trigger select_cde_bin
                            annotation_active = True
                            break
                else:
                    print("No bins found in the audio database.")
                    print("Please check that your audio files are organized in bin subdirectories.")
                    annotation_active = False
                    return
                    
                print(f"Review mode: CDE Review (bin target: {bin_target} definitive labels per bin)")
                update_cde_progress()
            else:
                # Apply filters and get updated filtered data
                filtered_df = refresh_filtered_data()
                
                # Display filtered results stats
                total_clips = len(audio_db.df)
                filtered_clips = len(filtered_df)
                min_score, max_score = score_range_slider.value
                
                print(f"Total: {total_clips}, Filtered: {filtered_clips} (scores: {min_score:.2f}-{max_score:.2f})")
                print(f"Review mode: {'Sorted by score (highest first)' if review_mode == 'sorted' else 'Random selection'}")
                
                if filtered_clips > 0:
                    annotation_active = True
                    print("Starting annotation process...")
                else:
                    print("No clips match the filter criteria. Please adjust the score range.")
                    annotation_active = False
                    return
            
            load_next_clip()
    
    # Function to handle stopping annotation
    def stop_annotation(b):
        nonlocal annotation_active
        annotation_active = False
        
        with filter_output:
            clear_output()
            print("Annotation session ended.")
            
            # Get summary of annotation progress
            total = len(audio_db.df)
            not_reviewed = len(audio_db.df.filter(pl.col("annotation") == 4))
            reviewed = total - not_reviewed
            
            print(f"Total clips: {total}")
            print(f"Reviewed: {reviewed} ({reviewed/total*100:.1f}%)")
            
            if reviewed > 0:
                # Count by annotation type
                not_present = len(audio_db.df.filter(pl.col("annotation") == 0))
                present = len(audio_db.df.filter(pl.col("annotation") == 1))
                uncertain = len(audio_db.df.filter(pl.col("annotation") == 3))
                
                print(f"Not Present: {not_present} ({not_present/reviewed*100:.1f}%)")
                print(f"Present: {present} ({present/reviewed*100:.1f}%)")
                print(f"Uncertain: {uncertain} ({uncertain/reviewed*100:.1f}%)")
                
                # For CDE review mode, show progress per bin
                if review_mode == "cde review" and cde_bins:
                    print("\nCDE Review Progress:")
                    for bin_name in sorted(cde_bins.keys()):
                        definitive = bin_progress[bin_name]
                        target = min(bin_target, bin_counts[bin_name])
                        status = "✓" if definitive >= target else " "
                        
                        # Count uncertain annotations for this bin
                        uncertain_clips = audio_db.df.filter(
                            (pl.col("file_path").str.contains(f"/{bin_name}/")) & 
                            (pl.col("annotation") == 3)  # Uncertain annotations
                        )
                        uncertain_count = len(uncertain_clips)
                        
                        print(f"[{status}] {bin_name}: {definitive}/{target} definitive labels ({definitive/target*100:.1f}%)")
                        if uncertain_count > 0:
                            print(f"    + {uncertain_count} uncertain annotations (not counted toward target)")
        
        with spectrogram_output:
            clear_output()
        
        with audio_info_output:
            clear_output()
        
        with status_output:
            clear_output()
            
        with cde_progress_output:
            clear_output()
    
    # Connect the buttons to their functions
    filter_button.on_click(start_annotation)
    stop_button.on_click(stop_annotation)
    next_button.on_click(next_clip)
    back_button.on_click(back_clip)
    
    # Connect annotation buttons
    not_present_button.on_click(lambda b: update_annotation(0))
    present_button.on_click(lambda b: update_annotation(1))
    uncertain_button.on_click(lambda b: update_annotation(3))
    
    # Create a compact title
    title_text = f"Audio Clip Annotation Tool ({review_mode.capitalize()} Mode)"
    title = widgets.HTML(f"<h3 style='margin:0px;'>{title_text}</h3>")
    
    # Create navigation buttons layout
    nav_buttons = widgets.HBox([
        back_button,
        next_button,
        stop_button
    ], layout=widgets.Layout(
        display='flex',
        flex_flow='row',
        justify_content='flex-end',
        width='auto',
        margin='0px 0px'
    ))
    
    # Create filter controls based on review mode
    if review_mode == "cde review":
        # For CDE review mode, we use bin dropdown instead of score slider
        filter_section = widgets.VBox([
            widgets.HTML("<p style='margin:5px 0px 2px 0px;'><b>CDE Review Mode</b>:</p>"),
            widgets.HTML(f"<p style='margin:2px 0px;'>Target: {bin_target} definitive labels per bin</p>"),
            bin_dropdown,
            filter_button,
            filter_output,
            cde_progress_output
        ], layout=widgets.Layout(margin='0px 10px 0px 0px'))
    else:
        # For other modes, use the standard score filter
        filter_section = widgets.VBox([
            widgets.HTML("<p style='margin:5px 0px 2px 0px;'><b>Filter Clips</b> (not yet reviewed):</p>"),
            score_range_slider,
            filter_button,
            filter_output
        ], layout=widgets.Layout(margin='0px 10px 0px 0px'))
    
    # Create compact annotation section
    annotation_section = widgets.VBox([
        widgets.HTML("<p style='margin:5px 0px 2px 0px;text-align:right;'><b>Annotation Controls</b>:</p>"),
        annotation_buttons,
        status_output
    ], layout=widgets.Layout(
        align_items='flex-end',
        margin='0px 0px 0px 10px'
    ))
    
    # Create a layout with the navigation buttons at the top
    top_bar = widgets.HBox([
        title,
        nav_buttons
    ], layout=widgets.Layout(width='100%', margin='0px 0px 5px 0px'))
    
    # Adjust the layout for larger spectrogram - give it more space
    audio_display_section = widgets.HBox([
        widgets.VBox([
            widgets.HTML("<p style='margin:2px 0px;'><b>Spectrogram</b>:</p>"),
            spectrogram_output
        ], layout=widgets.Layout(width='70%', margin='0px')),  # Increased from 60% to 70%
        
        # Small spacer
        widgets.HTML("<div style='width: 10px;'></div>"),
        
        widgets.VBox([
            audio_info_output
        ], layout=widgets.Layout(width='30%', margin='0px'))  # Decreased from 40% to 30%
    ], layout=widgets.Layout(margin='0px'))
    
    # Create a compact layout
    main_layout = widgets.VBox([
        top_bar,
        widgets.HBox([
            widgets.VBox([filter_section], layout=widgets.Layout(width='50%')),
            widgets.VBox([annotation_section], layout=widgets.Layout(width='50%'))
        ], layout=widgets.Layout(margin='0px 0px 5px 0px')),
        audio_display_section
    ], layout=widgets.Layout(padding='5px'))
    
    # Display the widgets
    display(main_layout)