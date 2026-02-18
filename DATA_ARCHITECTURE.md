# Data Storage Architectures

This document describes the data storage architectures for the Active Learning and Validation workflows in the Bioacoustics Web Application.

---

## Overview

The application uses two distinct database architectures optimized for different workflows:

1. **Active Learning Database** (`modules/database.py`) - For iterative model training and dataset building
2. **Validation Database** (`modules/validation_db.py`) - For systematic validation of model predictions

Both use **Polars DataFrames** stored as Parquet files for efficient columnar storage and fast query performance.

---

## Active Learning Database Architecture

### Purpose
Build high-quality training datasets through iterative annotation, model training, and active learning cycles.

### Three-Table Structure

#### 1. **Files Table** (`files.parquet`)
Stores metadata about audio files in the dataset.

| Column | Type | Description |
|--------|------|-------------|
| `file_id` | String (UUID) | Primary key: unique identifier for each file |
| `file_name` | String | Name of the audio file (without path) |
| `file_path` | String | Full path to the audio file |
| `duration_sec` | Float32 | Total duration of the audio file in seconds |
| `sampling_rate` | Int32 | Audio sampling rate in Hz |
| `created_at` | Datetime | When file was added to database |

**Key Characteristics:**
- One row per audio file
- Tracks file-level metadata
- Referenced by clips via `file_id` foreign key

#### 2. **Clips Table** (`clips.parquet`)
Stores segmented clips from audio files with annotation status and predictions.

| Column | Type | Description |
|--------|------|-------------|
| `clip_id` | String (UUID) | Primary key: unique identifier for each clip |
| `file_id` | String (UUID) | Foreign key to files table |
| `clip_start` | Float32 | Start time of the clip in seconds |
| `clip_end` | Float32 | End time of the clip in seconds |
| `annotation_status` | List[Int32] | Per-class annotation status array (0=present, 1=not_present, 2=uncertain, 4=unreviewed) |
| `confidence_predictions` | List[Float32] | Per-class confidence scores array from model predictions |
| `label_strength` | List[Int32] | Per-class label strength array (0=weak, 1=strong) |
| `embedding_index` | Int64 | Index into embeddings array for model features |
| `created_at` | Datetime | When clip was created |

**Key Characteristics:**
- Multiple rows per file (one per clip segment)
- **Multiclass support**: Arrays enable simultaneous tracking of multiple species per clip
- Links to embeddings via `embedding_index` for model training
- Annotation status tracked per class

#### 3. **Annotations Table** (`annotations.parquet`)
Stores individual human annotations for specific classes on clips.

| Column | Type | Description |
|--------|------|-------------|
| `annotation_id` | String (UUID) | Primary key: unique identifier for each annotation |
| `clip_id` | String (UUID) | Foreign key to clips table |
| `class_name` | String | Name of the class being annotated |
| `label` | String | Label value: 'present', 'not_present', 'uncertain' |
| `annotated_at` | Datetime | When annotation was made |

**Key Characteristics:**
- Multiple rows per clip (one per annotated class)
- Normalized structure: separate row for each class annotation
- Supports selective annotation (annotate only target classes)

### Additional Storage

#### Embeddings Array (`embeddings.pkl`)
- Stored as NumPy array (not Polars)
- One embedding vector per clip
- Indexed via `embedding_index` in clips table
- Used for model training and active learning queries

#### Metadata (`metadata.json`)
```json
{
  "embedding_model": "BirdNET",
  "embedding_dim": 1280,
  "num_classes": 5,
  "class_names": ["Species1", "Species2", ...],
  "created_at": "2024-01-01T00:00:00",
  "total_clips": 1000
}
```

### Data Flow

```
Audio Files
    ↓
[Embedding Generation]
    ↓
Files Table ← file metadata
    ↓
Clips Table ← clip segments + embedding_index
    ↓
[Active Learning Interface]
    ↓
Annotations Table ← human labels
    ↓
[Model Training]
    ↓
Updated confidence_predictions in Clips Table
```

### Storage Format
```
project_directory/
├── audio_database.parquet  # Legacy (deprecated)
├── files.parquet           # Files table
├── clips.parquet           # Clips table
├── annotations.parquet     # Annotations table
├── embeddings.pkl          # NumPy array of embeddings
└── metadata.json           # Database metadata
```

---

## Validation Database Architecture

### Purpose
Systematically validate model predictions across different strata (sites, dates, regions) to assess model performance and create validation datasets.

### Four-Table Structure

#### 1. **Predictions Table** (`predictions.parquet`)
Stores model predictions to be validated.

| Column | Type | Description |
|--------|------|-------------|
| `prediction_id` | String (UUID) | Primary key: unique identifier for each prediction |
| `filename` | String | Audio filename |
| `start_time` | Float32 | Start time in seconds |
| `end_time` | Float32 | End time in seconds |
| `species_name` | String | Species/class name predicted |
| `confidence` | Float32 | Model confidence score [0-1] |
| `model_name` | String | Source model name (BirdNET, PERCH, etc.) |
| `audio_file_path` | String | Full path to audio file |
| `strata_id` | String (UUID) | Foreign key to strata definitions |
| `strata` | String | User-provided strata grouping |
| `created_at` | Datetime | Import timestamp |

**Key Characteristics:**
- One row per prediction (species × clip combination)
- Flat structure optimized for filtering and sorting
- Strata column enables grouping by site, date, region, etc.

#### 2. **Validation Annotations Table** (`annotations.parquet`)
Stores human validation decisions for predictions.

| Column | Type | Description |
|--------|------|-------------|
| `annotation_id` | String (UUID) | Primary key |
| `prediction_id` | String (UUID) | Foreign key to predictions |
| `filename` | String | Audio filename (denormalized for performance) |
| `start_time` | Float32 | Start time (denormalized) |
| `end_time` | Float32 | End time (denormalized) |
| `species_name` | String | Species name (denormalized) |
| `original_confidence` | Float32 | Original model confidence |
| `validation_state` | String | confirmed/rejected/uncertain/skipped |
| `validation_confidence` | Int32 | User confidence 1-5 |
| `annotator_id` | String | Who made the annotation |
| `validated_at` | Datetime | When annotation was made |
| `strata_id` | String (UUID) | Foreign key to strata |
| `notes` | String | Optional user notes |

**Key Characteristics:**
- One row per validated prediction
- Denormalized fields for fast querying without joins
- Tracks validation state and user confidence
- Supports multiple annotators

#### 3. **Strata Definitions Table** (`strata_definitions.parquet`)
Defines validation groupings for systematic coverage.

| Column | Type | Description |
|--------|------|-------------|
| `strata_id` | String (UUID) | Primary key |
| `strata_name` | String | Human-readable name (e.g., "Site_A", "Week_15") |
| `strata_type` | String | Type of strata (user_provided, temporal, spatial) |
| `confidence_threshold` | Float32 | Minimum confidence threshold for this strata |
| `created_at` | Datetime | When strata was defined |

**Key Characteristics:**
- Defines groupings for systematic validation
- Enables balanced sampling across sites, times, conditions
- Configurable per-strata confidence thresholds

#### 4. **Validation Progress Table** (`validation_progress.parquet`)
Tracks validation progress per strata/species combination.

| Column | Type | Description |
|--------|------|-------------|
| `strata_id` | String (UUID) | Foreign key to strata |
| `strata_name` | String | Denormalized strata name |
| `species_name` | String | Species being validated |
| `total_clips` | Int32 | Total clips for this strata/species |
| `validated_clips` | Int32 | Number validated so far |
| `confirmed_clips` | Int32 | Number confirmed as correct |
| `rejected_clips` | Int32 | Number rejected as incorrect |
| `uncertain_clips` | Int32 | Number marked uncertain |
| `skipped_clips` | Int32 | Number skipped |
| `target_confirmations` | Int32 | Target number of confirmations |
| `last_updated` | Datetime | Last progress update |

**Key Characteristics:**
- One row per strata/species combination
- Real-time progress tracking
- Enables quota-based validation (e.g., "Get 10 confirmations per species per site")

### Data Flow

```
Model Predictions (CSV)
    ↓
[Load Predictions]
    ↓
Predictions Table ← predictions with strata
    ↓
[Create Strata]
    ↓
Strata Definitions Table ← unique strata
Validation Progress Table ← initialize progress counters
    ↓
[Validation Interface]
    ↓
Validation Annotations Table ← validation decisions
    ↓
[Update Progress]
    ↓
Validation Progress Table ← updated counters
```

### Storage Format
```
validation_project_directory/
├── predictions.parquet              # Predictions table
├── annotations.parquet              # Validation annotations
├── strata_definitions.parquet       # Strata definitions
├── validation_progress.parquet      # Progress tracking
└── project_metadata.json            # Project metadata
```

### Input Formats Supported

#### Standard CSV (Wide Format)
```csv
filename,start_time,end_time,strata,Species1,Species2,Species3
audio1.wav,0.0,3.0,site_A,0.85,0.12,0.03
```

#### Standard CSV (Long Format)
```csv
filename,start_time,end_time,strata,species_name,confidence
audio1.wav,0.0,3.0,site_A,Species1,0.85
audio1.wav,0.0,3.0,site_A,Species2,0.12
```

#### PNW-CNet Format (Automatic Parsing)
```csv
Filename,Species1,Species2,Species3
JCN_40758-12_20220614_123409_part_001.png,0.85,0.12,0.03
```
*Automatically extracts strata from filename pattern*

---

## Key Differences

| Aspect | Active Learning Database | Validation Database |
|--------|-------------------------|---------------------|
| **Primary Use Case** | Build training datasets through iterative annotation | Validate model predictions systematically |
| **Multiclass Support** | Native (arrays in clips table) | Via separate rows per species |
| **Embeddings** | Required (for model training) | Not used |
| **Strata Concept** | Not used | Core organizational principle |
| **Progress Tracking** | Annotation counts per class | Detailed per-strata/species quotas |
| **Data Normalization** | Highly normalized (3 tables) | Partially denormalized (4 tables with redundancy) |
| **Primary Key** | Clip-based (one clip → many classes) | Prediction-based (one prediction per species) |
| **Typical Size** | 100s-1000s of clips | 10,000s-100,000s of predictions |
| **Query Pattern** | Join-heavy (files ← clips ← annotations) | Filter-heavy (flat predictions table) |
| **Update Frequency** | High (continuous annotation + training) | Low (batch imports + periodic validation) |

---

## When to Use Each Architecture

### Use Active Learning Database When:
- Building a new training dataset from scratch
- Performing iterative model training
- Need multiclass annotation support
- Working with embeddings for active learning
- Dataset size: 100s to 10,000s of clips
- Focus on annotation quality and efficiency

### Use Validation Database When:
- Validating existing model predictions
- Assessing model performance across different conditions
- Need systematic coverage (sites, dates, regions)
- Working with large-scale prediction outputs
- Dataset size: 10,000s to 1,000,000s of predictions
- Focus on validation coverage and progress tracking

---

## Data Persistence

Both architectures use:
- **Polars DataFrames** for efficient columnar operations
- **Parquet files** for compressed, columnar storage
- **JSON metadata** for configuration and versioning

### Advantages of Parquet Format:
- Columnar storage: fast filtering on specific columns
- Compression: ~10x smaller than CSV
- Schema preservation: datatypes maintained
- Cross-platform compatibility
- Fast I/O with Polars

### Performance Characteristics:
- **Active Learning**: Optimized for frequent updates (annotation, training)
- **Validation**: Optimized for bulk loading and filtered queries
- Both support datasets too large for memory (chunked processing available)

---

## Migration Between Systems

### Active Learning → Validation
Export annotated clips with confidence scores:
```python
# Export from Active Learning database
confirmed_clips = db.get_annotated_clips(label='present')

# Create CSV for Validation import
csv_data = {
    'filename': clip_filenames,
    'start_time': clip_starts,
    'end_time': clip_ends,
    'strata': clip_strata,
    'species_name': species_names,
    'confidence': confidence_scores
}
```

### Validation → Active Learning
Import validated predictions as training data:
```python
# Export from Validation database
confirmed = validation_db.get_annotations(validation_state='confirmed')

# Import to Active Learning database
for clip in confirmed:
    db.add_annotation(
        clip_id=clip.clip_id,
        class_name=clip.species_name,
        label='present'
    )
```

---

## Best Practices

### Active Learning Database
1. **Regular saves**: Auto-save after each annotation session
2. **Backup embeddings**: Regeneration is expensive (2-3 min startup)
3. **Class management**: Define all classes upfront to avoid schema changes
4. **Embedding versioning**: Track which model generated embeddings

### Validation Database
1. **Strata design**: Choose meaningful groupings (site, date, region)
2. **Quota planning**: Set realistic target confirmations per strata/species
3. **Progress monitoring**: Track coverage across all strata
4. **Multiple projects**: Use separate projects for different model versions

### General
1. **Consistent file paths**: Use absolute paths or project-relative paths
2. **Regular exports**: Export to CSV periodically for backup
3. **Version control**: Track metadata.json and project_metadata.json in git
4. **File organization**: Keep database files separate from audio files

---

## Future Enhancements

### Potential Improvements
- **Unified database**: Single architecture supporting both workflows
- **SQLite backend**: Optional SQL support for complex queries
- **Cloud storage**: S3/GCS integration for large-scale deployments
- **Incremental updates**: Partial saves for large datasets
- **Multi-user support**: Concurrent annotation with conflict resolution
- **Audit logging**: Track all database modifications

---

## Technical Notes

### Polars vs Pandas
This application uses **Polars** instead of Pandas for:
- **Speed**: 5-10x faster for filtering and aggregations
- **Memory efficiency**: Lazy evaluation and columnar storage
- **Type safety**: Strict schema enforcement
- **Modern API**: Expression-based queries vs method chaining

### Schema Evolution
Both databases support schema migrations:
- **Version detection**: Check for missing columns on load
- **Automatic migration**: Add new columns with default values
- **Backwards compatibility**: Maintain support for older formats

### Performance Optimization
- **Lazy evaluation**: Queries optimized before execution
- **Predicate pushdown**: Filters applied during file read
- **Columnar scanning**: Only read needed columns
- **Parallel processing**: Multi-threaded operations where possible

---

## Summary

The application uses two specialized database architectures:

1. **Active Learning Database**: Normalized, multiclass-friendly structure optimized for iterative annotation and model training with embeddings.

2. **Validation Database**: Denormalized, strata-based structure optimized for systematic validation of large-scale model predictions.

Both leverage Polars and Parquet for efficient storage and fast querying, with designs tailored to their specific workflows and query patterns.
