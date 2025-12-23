# Bug Report & Code Analysis

## 1. Bugs

### Backend (FastAPI)
*   **Memory-Intensive Audio Slicing (`backend/main.py`)**: The `/api/audio/` endpoint (function `get_audio_clip`) calls `u.load_audio(file_path, None)`, which loads the *entire* audio file into memory (and potentially resamples it) before slicing out the requested segment. For long recordings (e.g., 1 hour+), this causes massive memory spikes and slow response times.
*   **Non-Thread-Safe Global State (`backend/main.py`)**: The application relies on global dictionaries (`app_state`, `eval_state`, `training_state`) to manage database instances and model references. This is not thread-safe. Concurrent requests could lead to race conditions or inconsistent state.
*   **Silent Failures in Background Threads (`backend/main.py`)**: `build_dataset_thread` and `training_thread` catch general exceptions but only update a status string. If the Python process crashes or is killed, the frontend will be left in a permanent "building" or "training" state with no way to recover without a refresh.
*   **Inefficient TFLite Inference (`backend/modules/utilities.py`)**: In `process_with_birdnet`, the TFLite interpreter is re-initialized for *every single frame* (144000 samples). This adds significant overhead (~50-100ms per frame) and makes processing long files with BirdNET extremely slow.
*   **Potential Directory Traversal (`backend/main.py`)**: The `/api/audio/{file_path:path}` endpoint takes a raw file path from the user. While `pathlib` offers some protection, it's generally unsafe to accept arbitrary absolute paths without validation against a whitelist of allowed directories.

### Frontend (React)
*   **Unnecessary Re-renders (`frontend/src/App.js`)**: The `Tabs` component is configured with `forceRenderTabPanel={true}`. This keeps all tab components (Active Learning, Database Viewer, etc.) mounted and active in the background, consuming browser memory and potentially running `useEffect` hooks even when hidden.
*   **Missing Error Boundaries**: There are no React Error Boundaries. If a single component crashes (e.g., due to malformed data), the entire application screen will go blank (White Screen of Death).

## 2. Redundant & Slow Code

### Backend
*   **Redundant Database Joins (`backend/modules/database.py`)**: The `Audio_DB.df` property performs a Polars `join` between `clips_df` and `files_df` *every time* it is accessed. This is used frequently (e.g., in `get_label_statistics`, `find_similar_clips`). For large datasets, this join is expensive and should be cached or done lazily.
*   **Unoptimized Resampling (`backend/modules/utilities.py`)**: The `load_audio` function uses `librosa.resample`. If `offset` and `duration` are provided, it's efficient. However, if they are *not* provided (like in the current `/api/audio` implementation), it resamples the entire file. It should use `soundfile`'s native resampling or `librosa.load`'s streaming capabilities more effectively.
*   **Repeated Metadata Parsing (`backend/modules/validation_db.py`)**: In `load_pnw_cnet_predictions`, the code iterates through rows and calls `_parse_pnw_cnet_filename` for every row. Since many rows share the same filename (different species), this is redundant. It should parse unique filenames once and map them.

### Frontend
*   **Large Bundle Size**: The frontend imports full libraries (like `plotly.js`) which might be overkill if only basic plotting is needed. This increases initial load time.

## 3. Recommendations

1.  **Optimize Audio Loading**:
    *   Refactor `get_audio_clip` to pass `offset` and `duration` directly to `u.load_audio`.
    *   Update `u.load_audio` to use `soundfile.read(..., start=..., stop=...)` to only read the necessary bytes from disk.

2.  **Fix TFLite Performance**:
    *   Refactor `process_with_birdnet` to initialize the TFLite interpreter *once* (e.g., in a class or context manager) and reuse it for all frames in a batch.

3.  **Improve State Management**:
    *   Move from global `app_state` dictionaries to a singleton class pattern or use FastAPI's dependency injection system to manage the `Audio_DB` instance safely.

4.  **Database Optimization**:
    *   Cache the "joined" view in `Audio_DB` and only update it when `clips_df` or `files_df` changes.
    *   Use Polars' `LazyFrame` API for complex queries to allow the query optimizer to work.

5.  **Frontend Performance**:
    *   Disable `forceRenderTabPanel` in `App.js`.
    *   Wrap heavy components (`ActiveLearning`, `DatabaseViewer`) in `React.memo` or use `useMemo` for expensive computations.

6.  **Security**:
    *   Implement a `ALLOWED_DIRECTORIES` configuration. Validate that any requested `file_path` starts with one of these allowed paths before accessing the file system.
