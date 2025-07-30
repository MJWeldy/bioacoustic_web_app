import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import soundfile as sf
import librosa

from modules import config as cfg

BACKEND = os.environ.get("BACKEND", "PERCH")

#def load_embedding_model():
#    if BACKEND == "Perch" or BACKEND == "PERCH_IGNORE_SR":
#        e_model = hub.load(
#            f"https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/{cfg.PERCH_V}"
#        )
#    if BACKEND == "BirdNET_2.4":
#        from tensorflow.lite.python import interpreter as tfl_interpreter
#        e_model = tfl_interpreter.Interpreter(
#            model_path=cfg.MODEL_PATH,
#            num_threads=cfg.TFLITE_THREADS,
#            experimental_preserve_all_tensors=True
#        )
#
#    return e_model

def load_audio(file_path, start_stop):
    """Load audio file"""
    if not start_stop: 
        audio, sample_rate = sf.read(file_path)
    else:
        start, stop, sr = start_stop
        audio, sample_rate = sf.read(file_path, start = int(start*sr), stop = int(stop*sr) )
    if sample_rate != cfg.TARGET_SR:
        audio = librosa.resample(audio.T, orig_sr=sample_rate, target_sr=cfg.TARGET_SR)
        audio = audio.T
    return np.array(audio)  # tf.squeeze(audio)

@tf.function
def normalize_audio(audio, norm_factor):
    """Normalize the audio at the peak values used in Perch model training"""
    audio = tf.identity(audio)
    audio -= tf.reduce_mean(audio, axis=-1, keepdims=True)
    peak_norm = tf.reduce_max(tf.abs(audio), axis=-1, keepdims=True)
    audio = tf.where(peak_norm > 0.0, audio / peak_norm, audio)
    audio = audio * norm_factor
    return audio

@tf.function
def frame_audio(audio_array: np.ndarray) -> np.ndarray: #,
                #window_size_s: float = 5.0,
                #hop_size_s: float = 5.0
    """Framing audio for inference"""
    if cfg.WINDOW is None or cfg.WINDOW < 0:
        return audio_array[tf.newaxis, :]  # np.newaxis

    frame_length = cfg.MODEL_CONTEXT_FRAME #int(window_size_s * cfg.MODEL_SR)
    hop_length = cfg.HOP_SIZE #int(hop_size_s * cfg.MODEL_SR)

    num_frames = int(tf.math.ceil(tf.shape(audio_array)[0] / frame_length))

    framed_audio = tf.signal.frame(
        audio_array, frame_length, hop_length, pad_end=False, pad_value=0.0
    )
    # if the last frame of audio is shorter than frame_length pad it by concatenating the frame multiple times
    if tf.shape(framed_audio)[0] < num_frames:
        tail = audio_array[((num_frames - 1) * frame_length) :]
        num_repeats = int(tf.math.ceil(frame_length / tf.shape(tail)[0]))  # np.ciel
        last_audio_frame = tf.tile(tail, [num_repeats])[tf.newaxis, :frame_length]
        framed_audio = tf.concat([framed_audio, last_audio_frame], 0)

    return framed_audio  # tf.squeeze(framed_audio, axis=0)

def load_and_preprocess(files):
    if BACKEND == "PERCH" or BACKEND == "PERCH_IGNORE_SR":
        e_model = hub.load(
            f"https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/{cfg.PERCH_V}"
        )
        e = list(map(lambda f: process_with_perch(f, e_model), files))
    elif BACKEND == "BirdNET_2.4":
        e = process_with_birdnet(files)
    return e

def process_with_perch(file_path, e_model):
    audio = load_audio(file_path, None)  
    #if len(audio.shape) < 2:
    #    audio = audio[np.newaxis,]
    audio = tf.cast(audio, tf.float32)
    normalized_audio = normalize_audio(audio, 0.25)
    framed_audio = frame_audio(normalized_audio)
    if len(framed_audio.shape) > 2:
        framed_audio = tf.squeeze(framed_audio)
    
    e = e_model.infer_tf(framed_audio)
    e = e["embedding"]
    return e

def process_with_birdnet(files):
    """Process audio frames with BirdNET TFLite model
    
    This function handles all TFLite-specific operations in one place
    """
    for file in files:
        audio = load_audio(file, None)
        audio = tf.cast(audio, tf.float32)
        framed_audio = frame_audio(audio)
        if len(framed_audio.shape) > 2:
            framed_audio = tf.squeeze(framed_audio)
        e = []
        file_e = []
        # Process each frame with a fresh interpreter
        for frame in framed_audio:
            # Create a completely new interpreter for each frame
            #from ai_edge_litert.interpreter import Interpreter
            from tensorflow.lite.python import interpreter as tfl_interpreter
            interpreter = tfl_interpreter.Interpreter(
                model_path=cfg.MODEL_PATH,
                num_threads=cfg.TFLITE_THREADS,
                experimental_preserve_all_tensors=True
            )
            # Allocate tensors
            interpreter.allocate_tensors()
            # Get input and output details
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]
            # Prepare input data
            interpreter.set_tensor(
                input_details['index'], np.float32(frame)[np.newaxis, :]
            )
            # Run inference
            interpreter.invoke()
            
            # Get embedding from penultimate layer
            # Adjust this based on your model architecture
            output_index = output_details['index'] - 1
            interpreter.allocate_tensors()
            embedding = interpreter.get_tensor(output_index)
            
            file_e.append(embedding)
    e.append(np.array(file_e).squeeze())
    return e

def flatten_pred_list(nested_list):
    flat_list = []
    for i, l in enumerate(nested_list):
        for e in l:
            flat_list.append(e.numpy().item())
    return flat_list

def get_classifier_predictions(embeddings, classifier, class_map, sound_name):
    idx = class_map[sound_name]
    logits = list(map(classifier, embeddings))
    preds = list(map(tf.sigmoid, logits))
    pred_idx = list(map(lambda arr: arr[:,idx] if len(arr) > idx else None, preds))
    return flatten_pred_list(pred_idx)

def get_label(CLASS_MAP, FILE_PATH):
    tmp = np.zeros(len(set(CLASS_MAP.values())))
    for i, key in enumerate(CLASS_MAP):
        if key in str(FILE_PATH):
            # tmp[i] = 1
            tmp[CLASS_MAP[key]] = 1
    return tmp

def get_label_with_strength(CLASS_MAP, FILE_PATH, metadata=None):
    """
    Enhanced label extraction that includes label strength information.
    Prioritizes metadata over filename parsing for accuracy.
    
    Args:
        CLASS_MAP: Dictionary mapping class names to indices
        FILE_PATH: Path to the audio file
        metadata: Optional metadata dict with detailed label information
    
    Returns:
        tuple: (labels, label_strength) where both are numpy arrays
    """
    import os
    import json
    
    # Initialize arrays
    num_classes = len(set(CLASS_MAP.values()))
    labels = np.zeros(num_classes)
    label_strength = np.zeros(num_classes)  # 0 = weak, 1 = strong
    
    filename = os.path.basename(FILE_PATH)
    
    # PRIORITY 1: Get info from metadata if available (most accurate)
    if metadata and 'clips' in metadata:
        for clip_info in metadata['clips']:
            if clip_info['filename'] == filename:
                # Found metadata for this clip - use binary vectors directly
                metadata_labels = clip_info.get('labels', [])
                metadata_strengths = clip_info.get('label_strengths', [])
                
                # Map from metadata class indices to current CLASS_MAP indices
                # Use metadata class_map if available, otherwise assume same ordering
                metadata_class_map = metadata.get('class_map', CLASS_MAP)
                
                # Create mapping from metadata indices to current indices
                for meta_class_name, meta_class_idx in metadata_class_map.items():
                    if meta_class_name in CLASS_MAP:
                        current_class_idx = CLASS_MAP[meta_class_name]
                        
                        # Copy label and strength if within bounds
                        if meta_class_idx < len(metadata_labels):
                            labels[current_class_idx] = metadata_labels[meta_class_idx]
                        if meta_class_idx < len(metadata_strengths):
                            label_strength[current_class_idx] = metadata_strengths[meta_class_idx]
                
                return labels, label_strength
    
    # PRIORITY 2: Parse from filename (only strong labels)
    # Expected format: originalname_clipstart-annotation_slug-strong.wav
    if '-strong.wav' in filename:
        # Extract class information from filename (strong labels only)
        for class_name, class_idx in CLASS_MAP.items():
            if class_name in filename:
                labels[class_idx] = 1
                label_strength[class_idx] = 1  # Filename implies strong label
            elif 'empty' in filename:
                # Strong negative sample
                labels[class_idx] = 0  
                label_strength[class_idx] = 1  # Strong absence
    else:
        # PRIORITY 3: Legacy format - assume strong labels
        for class_name, class_idx in CLASS_MAP.items():
            if class_name in filename:
                labels[class_idx] = 1
                label_strength[class_idx] = 1  # Legacy = strong
    
    return labels, label_strength

def load_training_data_with_strength(folder_path, class_map):
    """
    Load training data from exported clips with label strength information.
    Uses enhanced metadata format with full binary vectors.
    
    Args:
        folder_path: Path to folder containing exported audio clips
        class_map: Dictionary mapping class names to indices
        
    Returns:
        tuple: (file_paths, labels, label_strengths, metadata)
    """
    import os
    import glob
    import json
    
    # Look for metadata file
    metadata_path = os.path.join(folder_path, 'export_metadata.json')
    metadata = None
    loading_mode = "filename_only"
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Found enhanced metadata file: {metadata_path}")
        
        # Check if this has the new binary vector format
        if metadata.get('clips') and len(metadata['clips']) > 0:
            first_clip = metadata['clips'][0]
            if 'labels' in first_clip and 'label_strengths' in first_clip:
                loading_mode = "metadata_vectors"
                print(f"✓ Using binary vector metadata (multiclass-compatible)")
            else:
                loading_mode = "metadata_legacy"
                print(f"✓ Using legacy metadata format")
    else:
        print("⚠ No metadata file found, will parse from filenames only")
    
    # Find all wav files
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
    
    if not wav_files:
        raise ValueError(f"No WAV files found in {folder_path}")
    
    file_paths = []
    all_labels = []
    all_label_strengths = []
    
    print(f"Processing {len(wav_files)} audio files using {loading_mode} mode...")
    
    for file_path in wav_files:
        labels, label_strength = get_label_with_strength(class_map, file_path, metadata)
        
        # Include all files (even those with no positive labels - negative examples are important)
        file_paths.append(file_path)
        all_labels.append(labels)
        all_label_strengths.append(label_strength)
    
    if not file_paths:
        raise ValueError("No audio files found in the dataset")
    
    # Convert to numpy arrays
    labels_array = np.array(all_labels)
    strengths_array = np.array(all_label_strengths)
    
    # Detailed statistics
    total_labels = np.sum(labels_array)
    strong_labels = np.sum(strengths_array == 1)
    weak_labels = np.sum(strengths_array == 0)
    positive_samples = np.sum(np.any(labels_array == 1, axis=1))
    negative_samples = len(file_paths) - positive_samples
    
    print(f"✓ Loaded {len(file_paths)} audio clips:")
    print(f"  - {positive_samples} clips with positive labels")
    print(f"  - {negative_samples} clips with only negative labels")
    print(f"  - {total_labels} total labels across all classes")
    print(f"  - {strong_labels} strong labels, {weak_labels} weak labels")
    
    if loading_mode == "metadata_vectors":
        num_classes = labels_array.shape[1]
        print(f"  - {num_classes} classes detected from metadata")
        for class_name, class_idx in class_map.items():
            if class_idx < num_classes:
                class_positives = np.sum(labels_array[:, class_idx] == 1)
                print(f"    • {class_name}: {class_positives} positive samples")
    
    return file_paths, labels_array, strengths_array, metadata