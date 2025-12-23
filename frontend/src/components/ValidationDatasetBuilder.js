import React, { useState } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';

const ValidationDatasetBuilder = () => {
  // Validation workflow type
  const [validationType, setValidationType] = useState('');

  // Project persistence states (common to all workflows)
  const [projectName, setProjectName] = useState('');
  const [saveLocation, setSaveLocation] = useState('');

  // Workflow 1: Unvalidated Clips
  const [audioDirectory, setAudioDirectory] = useState('');
  const [clipWindowLength, setClipWindowLength] = useState(3.0);
  const [targetClasses, setTargetClasses] = useState('');
  const [strataColumn, setStrataColumn] = useState('');

  // Workflow 2 & 3: Prediction Sets (Standard and PNW-CNet)
  const [predictionsFile, setPredictionsFile] = useState('');
  const [predictionAudioDirectory, setPredictionAudioDirectory] = useState('');
  const [modelName, setModelName] = useState('');
  const [formatType, setFormatType] = useState('auto');
  const [recursive, setRecursive] = useState(true);
  const [pnwCnetStrataField, setPnwCnetStrataField] = useState('site_station');

  // Workflow 4: Call Density Estimation
  const [densityAudioDirectory, setDensityAudioDirectory] = useState('');
  const [densityClipLength, setDensityClipLength] = useState(3.0);
  const [densityTargetClass, setDensityTargetClass] = useState('');
  const [samplingInterval, setSamplingInterval] = useState(60);
  const [clipsPerInterval, setClipsPerInterval] = useState(5);

  // Loading and preview states
  const [isLoading, setIsLoading] = useState(false);
  const [loadSummary, setLoadSummary] = useState(null);
  const [strataCreated, setStrataCreated] = useState(false);
  const [strataSummary, setStrataSummary] = useState(null);

  const formatOptions = [
    { value: 'auto', label: 'Auto-detect' },
    { value: 'wide', label: 'Wide format (species as columns)' },
    { value: 'long', label: 'Long format (species in rows)' }
  ];

  const temporalUnits = [
    { value: 'week', label: 'Week' },
    { value: 'month', label: 'Month' },
    { value: 'year', label: 'Year' }
  ];

  const loadValidationDataset = async () => {
    if (!validationType) {
      toast.error('Please select a validation type');
      return;
    }

    setIsLoading(true);
    setLoadSummary(null);

    // Validate save location if provided
    if (saveLocation.trim() && !saveLocation.trim().startsWith('/')) {
      toast.error('❌ Save location must be an absolute path starting with "/" (e.g., /home/user/projects)', { autoClose: 8000 });
      setIsLoading(false);
      return;
    }

    try {
      let response;

      if (validationType === 'unvalidated_clips') {
        // Workflow 1: Unvalidated Clips
        if (!audioDirectory.trim()) {
          toast.error('Please specify an audio directory');
          setIsLoading(false);
          return;
        }
        if (!targetClasses.trim()) {
          toast.error('Please specify at least one target class');
          setIsLoading(false);
          return;
        }

        response = await axios.post('/api/validation/load-unvalidated-clips', {
          audio_directory: audioDirectory,
          clip_window_length: clipWindowLength,
          target_classes: targetClasses.split(',').map(c => c.trim()).filter(c => c),
          strata_column: strataColumn.trim() || null,
          save_location: saveLocation.trim() || null
        });

      } else if (validationType === 'prediction_sets') {
        // Workflow 2: Standard Prediction Sets
        if (!predictionsFile.trim()) {
          toast.error('Please specify a predictions file');
          setIsLoading(false);
          return;
        }
        if (!modelName.trim()) {
          toast.error('Please specify a model name');
          setIsLoading(false);
          return;
        }

        response = await axios.post('/api/validation/load-predictions', {
          predictions_path: predictionsFile,
          audio_directory: predictionAudioDirectory || null,
          model_name: modelName,
          format_type: formatType,
          recursive: recursive,
          use_pnw_cnet_format: false,
          save_location: saveLocation.trim() || null
        });

      } else if (validationType === 'pnw_cnet') {
        // Workflow 3: PNW-CNet Predictions
        if (!predictionsFile.trim()) {
          toast.error('Please specify a PNW-CNet predictions file');
          setIsLoading(false);
          return;
        }

        response = await axios.post('/api/validation/load-predictions', {
          predictions_path: predictionsFile,
          audio_directory: predictionAudioDirectory || null,
          model_name: modelName || 'PNW-CNet',
          format_type: 'auto',
          recursive: recursive,
          use_pnw_cnet_format: true,
          pnw_cnet_strata_field: pnwCnetStrataField,
          save_location: saveLocation.trim() || null
        });

      } else if (validationType === 'call_density') {
        // Workflow 4: Call Density Estimation
        if (!densityAudioDirectory.trim()) {
          toast.error('Please specify an audio directory');
          setIsLoading(false);
          return;
        }
        if (!densityTargetClass.trim()) {
          toast.error('Please specify a target class');
          setIsLoading(false);
          return;
        }

        response = await axios.post('/api/validation/load-density-estimation', {
          audio_directory: densityAudioDirectory,
          clip_length: densityClipLength,
          target_class: densityTargetClass.trim(),
          sampling_interval: samplingInterval,
          clips_per_interval: clipsPerInterval,
          save_location: saveLocation.trim() || null
        });
      }

      if (response && response.data.status === 'success') {
        setLoadSummary(response.data);
        const itemCount = response.data.total_predictions || response.data.total_clips || 0;
        toast.success(`Loaded ${itemCount} items successfully`);

        // Show auto-save notification if project was auto-saved
        if (response.data.auto_saved && response.data.project_path) {
          toast.info(`✅ Project auto-saved to: ${response.data.project_path}`, { autoClose: 8000 });
        } else if (response.data.auto_save_warning) {
          toast.warning(`⚠️ Auto-save failed: ${response.data.auto_save_warning}`);
        } else if (response.data.no_save_location) {
          toast.error('⚠️ No save location specified! Your validations will NOT be saved. Please specify a "Project Save Location" and reload the dataset.', { autoClose: false });
        }

        // Automatically create strata after successful loading
        await createStrataAutomatic();
      } else {
        // Handle API response with error status
        const errorMsg = response?.data?.message || 'Unknown error occurred';
        console.error('Load validation dataset failed:', errorMsg);
        toast.error(`Failed to load validation dataset: ${errorMsg}`);
      }
    } catch (error) {
      // Handle network errors or exceptions
      console.error('Load validation dataset error:', error);
      const message = error.response?.data?.detail || error.response?.data?.message || error.message || 'Network error or server unavailable';
      toast.error(`Failed to load validation dataset: ${message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const createStrata = async () => {
    if (!loadSummary) {
      toast.error('Please load predictions first');
      return;
    }

    setIsLoading(true);

    try {
      const response = await axios.post('/api/validation/create-strata', {});

      if (response.data.status === 'success') {
        setStrataCreated(true);
        setStrataSummary(response.data);
        toast.success(`Created ${response.data.strata_created} validation strata`);

        // Automatically save project after strata creation if save location is provided
        if (saveLocation.trim()) {
          await saveProjectAutomatic();
        }
      } else {
        toast.error(response.data.message || 'Failed to create strata');
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to create strata';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const createStrataAutomatic = async () => {
    try {
      const response = await axios.post('/api/validation/create-strata', {});

      if (response.data.status === 'success') {
        setStrataCreated(true);
        setStrataSummary(response.data);
        toast.success(`Automatically created ${response.data.strata_created} validation strata`);

        // Automatically save project after strata creation if save location is provided
        if (saveLocation.trim()) {
          await saveProjectAutomatic();
        }
      } else {
        toast.error(response.data.message || 'Failed to create strata automatically');
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to create strata automatically';
      toast.error(message);
    }
  };

  const saveProject = async () => {
    if (!saveLocation.trim()) {
      toast.error('Please specify a save location');
      return;
    }

    if (!loadSummary && !strataSummary) {
      toast.error('No validation data to save');
      return;
    }

    setIsLoading(true);

    try {
      const response = await axios.post('/api/validation/save-project', {
        base_path: saveLocation,
        project_name: projectName.trim() || undefined
      });

      if (response.data.status === 'success') {
        toast.success(`Project saved successfully: ${response.data.project_name}`);
        setProjectName(response.data.project_name);
      } else {
        toast.error(response.data.message || 'Failed to save project');
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to save project';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const saveProjectAutomatic = async () => {
    try {
      const response = await axios.post('/api/validation/save-project', {
        base_path: saveLocation,
        project_name: projectName.trim() || undefined
      });

      if (response.data.status === 'success') {
        toast.success(`Project automatically saved: ${response.data.project_name}`);
        setProjectName(response.data.project_name);
      } else {
        toast.error(response.data.message || 'Failed to save project automatically');
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to save project automatically';
      toast.error(message);
    }
  };


  return (
    <div>
      {/* Validation Type Selection */}
      <div className="card">
        <div className="card-header">
          <h3>Select Validation Type</h3>
          <p>Choose the type of validation workflow you want to perform</p>
        </div>

        <div className="grid grid-2">
          <div
            onClick={() => setValidationType('unvalidated_clips')}
            style={{
              border: validationType === 'unvalidated_clips' ? '2px solid #6e7cb9' : '2px solid #ddd',
              borderRadius: '8px',
              padding: '1.5rem',
              cursor: 'pointer',
              backgroundColor: validationType === 'unvalidated_clips' ? '#f0f2ff' : 'white',
              transition: 'all 0.2s'
            }}
          >
            <h4 style={{ margin: '0 0 0.5rem 0', color: '#6e7cb9' }}>Unvalidated Clips</h4>
            <p style={{ margin: 0, fontSize: '0.875rem', color: '#666' }}>
              Validate clips from audio files by subdividing them into fixed-length windows for target class annotation
            </p>
          </div>

          <div
            onClick={() => setValidationType('prediction_sets')}
            style={{
              border: validationType === 'prediction_sets' ? '2px solid #6e7cb9' : '2px solid #ddd',
              borderRadius: '8px',
              padding: '1.5rem',
              cursor: 'pointer',
              backgroundColor: validationType === 'prediction_sets' ? '#f0f2ff' : 'white',
              transition: 'all 0.2s'
            }}
          >
            <h4 style={{ margin: '0 0 0.5rem 0', color: '#6e7cb9' }}>Prediction Sets (Standard)</h4>
            <p style={{ margin: 0, fontSize: '0.875rem', color: '#666' }}>
              Validate model predictions from CSV files in wide or long format with optional strata column
            </p>
          </div>

          <div
            onClick={() => setValidationType('pnw_cnet')}
            style={{
              border: validationType === 'pnw_cnet' ? '2px solid #6e7cb9' : '2px solid #ddd',
              borderRadius: '8px',
              padding: '1.5rem',
              cursor: 'pointer',
              backgroundColor: validationType === 'pnw_cnet' ? '#f0f2ff' : 'white',
              transition: 'all 0.2s'
            }}
          >
            <h4 style={{ margin: '0 0 0.5rem 0', color: '#6e7cb9' }}>PNW-CNet Predictions</h4>
            <p style={{ margin: 0, fontSize: '0.875rem', color: '#666' }}>
              Validate default PNW-CNet prediction output with automatic filename parsing (NWFP format)
            </p>
          </div>

          <div
            onClick={() => setValidationType('call_density')}
            style={{
              border: validationType === 'call_density' ? '2px solid #6e7cb9' : '2px solid #ddd',
              borderRadius: '8px',
              padding: '1.5rem',
              cursor: 'pointer',
              backgroundColor: validationType === 'call_density' ? '#f0f2ff' : 'white',
              transition: 'all 0.2s'
            }}
          >
            <h4 style={{ margin: '0 0 0.5rem 0', color: '#6e7cb9' }}>Call Density Estimation</h4>
            <p style={{ margin: 0, fontSize: '0.875rem', color: '#666' }}>
              Validate clips for call density estimation using systematic temporal sampling
            </p>
          </div>
        </div>
      </div>

      {/* Project Configuration Section */}
      {validationType && (
        <div className="card">
          <div className="card-header">
            <h3>Project Configuration</h3>
            <p>Configure save location and project name for automatic project saving</p>
          </div>

        <div className="grid grid-2">
          <div className="form-group">
            <label htmlFor="saveLocation">Project Save Location *</label>
            <input
              type="text"
              id="saveLocation"
              className="form-control"
              placeholder="/path/to/validation/projects"
              value={saveLocation}
              onChange={(e) => setSaveLocation(e.target.value)}
              disabled={isLoading}
            />
            <small style={{ color: '#666', fontSize: '0.875rem' }}>
              Directory where validation projects will be saved automatically
            </small>
          </div>

          <div className="form-group">
            <label htmlFor="projectName">Project Name (Optional)</label>
            <input
              type="text"
              id="projectName"
              className="form-control"
              placeholder="my_validation_project"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              disabled={isLoading}
            />
            <small style={{ color: '#666', fontSize: '0.875rem' }}>
              Leave blank for auto-generated name
            </small>
          </div>
        </div>

          {!saveLocation.trim() && (
            <div style={{
              padding: '0.75rem',
              backgroundColor: '#fff3cd',
              border: '1px solid #ffeaa7',
              borderRadius: '4px',
              color: '#856404',
              marginTop: '1rem'
            }}>
              <strong>Note:</strong> Please specify a save location for automatic project saving after dataset creation.
            </div>
          )}
        </div>
      )}

      {/* Workflow 1: Unvalidated Clips */}
      {validationType === 'unvalidated_clips' && (
        <div className="card">
          <div className="card-header">
            <h3>Unvalidated Clips Configuration</h3>
            <p>Configure clip generation from audio files for validation</p>
          </div>

          <div className="grid grid-2">
            <div className="form-group">
              <label htmlFor="audioDirectory">Audio Files Directory *</label>
              <input
                type="text"
                id="audioDirectory"
                className="form-control"
                placeholder="/path/to/audio/files"
                value={audioDirectory}
                onChange={(e) => setAudioDirectory(e.target.value)}
                disabled={isLoading}
              />
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                Directory containing audio files to subdivide into clips
              </small>
            </div>

            <div className="form-group">
              <label htmlFor="clipWindowLength">Clip Window Length (seconds) *</label>
              <input
                type="number"
                id="clipWindowLength"
                className="form-control"
                placeholder="3.0"
                value={clipWindowLength}
                onChange={(e) => setClipWindowLength(parseFloat(e.target.value))}
                disabled={isLoading}
                step="0.1"
                min="0.1"
              />
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                Length of each clip window in seconds
              </small>
            </div>
          </div>

          <div className="grid grid-2">
            <div className="form-group">
              <label htmlFor="targetClasses">Target Class Names *</label>
              <input
                type="text"
                id="targetClasses"
                className="form-control"
                placeholder="Species1, Species2, Species3"
                value={targetClasses}
                onChange={(e) => setTargetClasses(e.target.value)}
                disabled={isLoading}
              />
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                Comma-separated list of target class names for validation
              </small>
            </div>

            <div className="form-group">
              <label htmlFor="strataColumn">Strata Grouping (Optional)</label>
              <input
                type="text"
                id="strataColumn"
                className="form-control"
                placeholder="site, location, date, etc."
                value={strataColumn}
                onChange={(e) => setStrataColumn(e.target.value)}
                disabled={isLoading}
              />
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                Optional strata field for grouping (extracted from filename or metadata)
              </small>
            </div>
          </div>

          <div style={{ textAlign: 'center', marginTop: '1.5rem' }}>
            <button
              onClick={loadValidationDataset}
              disabled={isLoading || !audioDirectory.trim() || !targetClasses.trim()}
              className="btn btn-primary btn-lg"
            >
              {isLoading ? 'Creating Validation Clips...' : 'Create Validation Dataset'}
            </button>
          </div>
        </div>
      )}

      {/* Workflow 2: Standard Prediction Sets */}
      {validationType === 'prediction_sets' && (
        <div className="card">
          <div className="card-header">
            <h3>Standard Prediction Sets Configuration</h3>
            <p>Load model predictions from CSV files in wide or long format</p>
          </div>

          <div className="grid grid-2">
            <div className="form-group">
              <label htmlFor="predictionsFile">Predictions File/Directory *</label>
              <input
                type="text"
                id="predictionsFile"
                className="form-control"
                placeholder="/path/to/predictions.csv"
                value={predictionsFile}
                onChange={(e) => setPredictionsFile(e.target.value)}
                disabled={isLoading}
              />
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                CSV file or directory containing prediction files
              </small>
            </div>

            <div className="form-group">
              <label htmlFor="predictionAudioDirectory">Audio Files Directory (Optional)</label>
              <input
                type="text"
                id="predictionAudioDirectory"
                className="form-control"
                placeholder="/path/to/audio/files"
                value={predictionAudioDirectory}
                onChange={(e) => setPredictionAudioDirectory(e.target.value)}
                disabled={isLoading}
              />
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                Directory containing original audio files for linking
              </small>
            </div>
          </div>

          <div className="grid grid-3">
            <div className="form-group">
              <label htmlFor="modelName">Model Name *</label>
              <input
                type="text"
                id="modelName"
                className="form-control"
                placeholder="BirdNET, PERCH, CustomModel"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                disabled={isLoading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="formatType">Data Format</label>
              <select
                id="formatType"
                className="form-control"
                value={formatType}
                onChange={(e) => setFormatType(e.target.value)}
                disabled={isLoading}
              >
                {formatOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '1.5rem' }}>
                <input
                  type="checkbox"
                  id="recursive"
                  checked={recursive}
                  onChange={(e) => setRecursive(e.target.checked)}
                  disabled={isLoading}
                  style={{
                    width: '18px',
                    height: '18px',
                    accentColor: '#6e7cb9'
                  }}
                />
                <label htmlFor="recursive" style={{ margin: 0, cursor: 'pointer' }}>
                  Recursive Search
                </label>
              </div>
            </div>
          </div>

          <div style={{ textAlign: 'center', marginTop: '1.5rem' }}>
            <button
              onClick={loadValidationDataset}
              disabled={isLoading || !predictionsFile.trim() || !modelName.trim()}
              className="btn btn-primary btn-lg"
            >
              {isLoading ? 'Loading Predictions...' : 'Load Prediction Dataset'}
            </button>
          </div>
        </div>
      )}

      {/* Workflow 3: PNW-CNet Predictions */}
      {validationType === 'pnw_cnet' && (
        <div className="card">
          <div className="card-header">
            <h3>PNW-CNet Predictions Configuration</h3>
            <p>Load default PNW-CNet prediction output with automatic filename parsing</p>
          </div>

          <div className="grid grid-2">
            <div className="form-group">
              <label htmlFor="predictionsFile">PNW-CNet Predictions File/Directory *</label>
              <input
                type="text"
                id="predictionsFile"
                className="form-control"
                placeholder="/path/to/predictions.csv or /path/to/predictions/directory"
                value={predictionsFile}
                onChange={(e) => setPredictionsFile(e.target.value)}
                disabled={isLoading}
              />
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                CSV file or directory (will recursively load all CSV files)
              </small>
              {predictionsFile && predictionsFile.trim() && (
                <small style={{ color: '#1976d2', fontSize: '0.75rem', display: 'block', marginTop: '0.25rem' }}>
                  Make sure this path exists and is accessible from this system
                </small>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="predictionAudioDirectory">Audio Files Directory (Optional)</label>
              <input
                type="text"
                id="predictionAudioDirectory"
                className="form-control"
                placeholder="/path/to/audio/files"
                value={predictionAudioDirectory}
                onChange={(e) => setPredictionAudioDirectory(e.target.value)}
                disabled={isLoading}
              />
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                Directory containing original audio files (.wav)
              </small>
            </div>
          </div>

          <div className="grid grid-2">
            <div className="form-group">
              <label htmlFor="modelName">Model Name (Optional)</label>
              <input
                type="text"
                id="modelName"
                className="form-control"
                placeholder="PNW-CNet"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                disabled={isLoading}
              />
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                Defaults to "PNW-CNet" if left blank
              </small>
            </div>

            <div className="form-group">
              <label htmlFor="pnwCnetStrataField">Strata Grouping Field</label>
              <select
                id="pnwCnetStrataField"
                className="form-control"
                value={pnwCnetStrataField}
                onChange={(e) => setPnwCnetStrataField(e.target.value)}
                disabled={isLoading}
              >
                <option value="site_station">Site-Station (e.g., 40758-12)</option>
                <option value="site">Site Only (e.g., 40758)</option>
                <option value="region">Region Only (e.g., JCN)</option>
                <option value="region_week">Region-Week (e.g., JCN-W24)</option>
                <option value="site_week">Site-Week (e.g., 40758-W24)</option>
                <option value="site_station_week">Site-Station-Week (e.g., 40758-12-W24)</option>
              </select>
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                Choose which field to use for validation strata (week = ISO week of year)
              </small>
            </div>
          </div>

          <div style={{
            padding: '1rem',
            backgroundColor: '#e3f2fd',
            border: '1px solid #bbdefb',
            borderRadius: '4px',
            marginTop: '1rem'
          }}>
            <h4 style={{ margin: '0 0 0.5rem 0', color: '#1565c0', fontSize: '0.9rem' }}>Filename Format:</h4>
            <p style={{ margin: 0, color: '#1565c0', fontSize: '0.875rem' }}>
              Expected format: <code>REGION_SITE-STATION_YYYYMMDD_HHMMSS_part_NNN.png</code><br/>
              Example: <code>JCN_40758-12_20220614_123409_part_001.png</code><br/>
              Clip times are automatically calculated from part numbers (12-second windows)
            </p>
          </div>

          <div style={{ textAlign: 'center', marginTop: '1.5rem' }}>
            <button
              onClick={loadValidationDataset}
              disabled={isLoading || !predictionsFile.trim()}
              className="btn btn-primary btn-lg"
            >
              {isLoading ? 'Loading PNW-CNet Predictions...' : 'Load PNW-CNet Dataset'}
            </button>
          </div>
        </div>
      )}

      {/* Workflow 4: Call Density Estimation */}
      {validationType === 'call_density' && (
        <div className="card">
          <div className="card-header">
            <h3>Call Density Estimation Configuration</h3>
            <p>Configure systematic temporal sampling for call density estimation</p>
          </div>

          <div style={{
            padding: '3rem',
            textAlign: 'center',
            backgroundColor: '#fff3cd',
            border: '2px solid #ffc107',
            borderRadius: '8px',
            margin: '2rem 0'
          }}>
            <h2 style={{ color: '#856404', marginBottom: '1rem' }}>⚠️ Feature Not Yet Implemented</h2>
            <p style={{ color: '#856404', fontSize: '1.1rem', marginBottom: '1.5rem' }}>
              The Call Density Estimation workflow is currently under development and will be available in a future release.
            </p>
            <p style={{ color: '#856404', fontSize: '0.9rem' }}>
              Please select one of the other validation workflows:
            </p>
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', marginTop: '1.5rem' }}>
              <button
                onClick={() => setValidationType('unvalidated_clips')}
                className="btn btn-secondary"
              >
                Unvalidated Clips
              </button>
              <button
                onClick={() => setValidationType('prediction_sets')}
                className="btn btn-secondary"
              >
                Prediction Sets
              </button>
              <button
                onClick={() => setValidationType('pnw_cnet')}
                className="btn btn-secondary"
              >
                PNW-CNet
              </button>
            </div>
          </div>

          {/* Hidden form for future implementation */}
          <div style={{ display: 'none' }}>
          <div className="grid grid-2">
            <div className="form-group">
              <label htmlFor="densityAudioDirectory">Audio Files Directory *</label>
              <input
                type="text"
                id="densityAudioDirectory"
                className="form-control"
                placeholder="/path/to/audio/files"
                value={densityAudioDirectory}
                onChange={(e) => setDensityAudioDirectory(e.target.value)}
                disabled={isLoading}
              />
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                Directory containing audio files for density estimation
              </small>
            </div>

            <div className="form-group">
              <label htmlFor="densityTargetClass">Target Class *</label>
              <input
                type="text"
                id="densityTargetClass"
                className="form-control"
                placeholder="Species name"
                value={densityTargetClass}
                onChange={(e) => setDensityTargetClass(e.target.value)}
                disabled={isLoading}
              />
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                Single target class for density estimation
              </small>
            </div>
          </div>

          <div className="grid grid-3">
            <div className="form-group">
              <label htmlFor="densityClipLength">Clip Length (seconds) *</label>
              <input
                type="number"
                id="densityClipLength"
                className="form-control"
                placeholder="3.0"
                value={densityClipLength}
                onChange={(e) => setDensityClipLength(parseFloat(e.target.value))}
                disabled={isLoading}
                step="0.1"
                min="0.1"
              />
            </div>

            <div className="form-group">
              <label htmlFor="samplingInterval">Sampling Interval (seconds)</label>
              <input
                type="number"
                id="samplingInterval"
                className="form-control"
                placeholder="60"
                value={samplingInterval}
                onChange={(e) => setSamplingInterval(parseInt(e.target.value))}
                disabled={isLoading}
                step="1"
                min="1"
              />
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                Time interval for systematic sampling
              </small>
            </div>

            <div className="form-group">
              <label htmlFor="clipsPerInterval">Clips per Interval</label>
              <input
                type="number"
                id="clipsPerInterval"
                className="form-control"
                placeholder="5"
                value={clipsPerInterval}
                onChange={(e) => setClipsPerInterval(parseInt(e.target.value))}
                disabled={isLoading}
                step="1"
                min="1"
              />
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                Number of clips to sample per interval
              </small>
            </div>
          </div>

          <div style={{ textAlign: 'center', marginTop: '1.5rem' }}>
            <button
              onClick={loadValidationDataset}
              disabled={isLoading || !densityAudioDirectory.trim() || !densityTargetClass.trim()}
              className="btn btn-primary btn-lg"
            >
              {isLoading ? 'Creating Density Dataset...' : 'Create Density Estimation Dataset'}
            </button>
          </div>
          </div>
          {/* End hidden form */}
        </div>
      )}

      {/* Load Summary */}
      {loadSummary && (
        <div className="card">
          <div className="card-header">
            <h3>Load Summary</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div className="status-indicator status-success"></div>
              <span style={{ color: '#059669', fontWeight: '600' }}>
                Successfully loaded {loadSummary.total_predictions} predictions
              </span>
            </div>
          </div>

          <div className="grid grid-2">
            <div>
              <h4>Dataset Overview</h4>
              <ul style={{ margin: '0', paddingLeft: '20px' }}>
                <li><strong>Total Predictions:</strong> {loadSummary.total_predictions}</li>
                <li><strong>Unique Files:</strong> {loadSummary.unique_files}</li>
                <li><strong>Unique Species:</strong> {loadSummary.unique_species}</li>
                <li><strong>Format Detected:</strong> {loadSummary.format_detected}</li>
                <li><strong>Audio Files Linked:</strong> {loadSummary.audio_files_linked}</li>
                {strataSummary && <li><strong>Validation Strata:</strong> {strataSummary.strata_created}</li>}
                {projectName && <li><strong>Project Saved:</strong> {projectName}</li>}
              </ul>
            </div>
            <div>
              <h4>Species Detected</h4>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                {loadSummary.species_list?.map(species => (
                  <span
                    key={species}
                    style={{
                      backgroundColor: '#e3f2fd',
                      color: '#1565c0',
                      padding: '0.25rem 0.5rem',
                      borderRadius: '4px',
                      fontSize: '0.875rem',
                      border: '1px solid #bbdefb'
                    }}
                  >
                    {species}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div style={{ marginTop: '1rem' }}>
            <h4>Confidence Range</h4>
            <p>
              Min: {loadSummary.confidence_range?.[0]?.toFixed(3)} |
              Max: {loadSummary.confidence_range?.[1]?.toFixed(3)}
            </p>
          </div>
        </div>
      )}

      {/* Validation Ready Message */}
      {strataSummary && (
        <div className="card">
          <div className="card-header">
            <h3>Validation Dataset Ready</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div className="status-indicator status-success"></div>
              <span style={{ color: '#059669', fontWeight: '600' }}>
                Validation dataset created and saved successfully!
              </span>
            </div>
          </div>

          <div style={{ textAlign: 'center', padding: '2rem' }}>
            <p style={{
              color: '#059669',
              fontWeight: '600',
              fontSize: '1.1rem',
              marginBottom: '1rem'
            }}>
              ✓ Dataset is ready for validation! Switch to the Validation tab to begin reviewing predictions.
            </p>
            <p style={{ color: '#666', fontSize: '0.9rem' }}>
              Created {strataSummary.strata_created} validation strata with {strataSummary.total_predictions_assigned} total predictions.
            </p>
          </div>
        </div>
      )}

      {/* Instructions */}
      {validationType && (
        <div className="card">
          <div className="card-header">
            <h3>Workflow Instructions</h3>
          </div>
          <div style={{ lineHeight: '1.6' }}>
            {validationType === 'unvalidated_clips' && (
              <>
                <h4 style={{ color: '#6e7cb9', marginTop: 0 }}>Unvalidated Clips Workflow</h4>
                <ol>
                  <li><strong>Configure Project:</strong> Set save location and project name for automatic saving</li>
                  <li><strong>Select Audio Directory:</strong> Specify directory containing audio files to validate</li>
                  <li><strong>Set Clip Length:</strong> Define the window length for subdividing recordings</li>
                  <li><strong>Specify Target Classes:</strong> Enter target class names (comma-separated)</li>
                  <li><strong>Create Dataset:</strong> System will generate clips and create validation strata</li>
                </ol>
                <div style={{
                  marginTop: '1rem',
                  padding: '1rem',
                  backgroundColor: '#e3f2fd',
                  border: '1px solid #bbdefb',
                  borderRadius: '4px'
                }}>
                  <p style={{ margin: '0', color: '#1565c0' }}>
                    <strong>Use case:</strong> Validating audio files without pre-existing predictions by creating fixed-length clips for manual annotation.
                  </p>
                </div>
              </>
            )}

            {validationType === 'prediction_sets' && (
              <>
                <h4 style={{ color: '#6e7cb9', marginTop: 0 }}>Standard Prediction Sets Workflow</h4>
                <ol>
                  <li><strong>Configure Project:</strong> Set save location and project name</li>
                  <li><strong>Load Predictions:</strong> Provide path to CSV file(s) with predictions</li>
                  <li><strong>Link Audio:</strong> Optionally provide directory containing original audio files</li>
                  <li><strong>Configure Format:</strong> Specify model name and data format</li>
                  <li><strong>Create Strata:</strong> System automatically creates validation strata</li>
                </ol>
                <div style={{
                  marginTop: '1rem',
                  padding: '1rem',
                  backgroundColor: '#e3f2fd',
                  border: '1px solid #bbdefb',
                  borderRadius: '4px'
                }}>
                  <h4 style={{ margin: '0 0 0.5rem 0', color: '#1565c0' }}>Data Format Requirements:</h4>
                  <p style={{ margin: '0', color: '#1565c0' }}>
                    <strong>Wide Format:</strong> Columns for each species (filename, start_time, end_time, strata, species1, species2, ...)<br/>
                    <strong>Long Format:</strong> Species in rows (filename, start_time, end_time, strata, species_name, confidence)<br/>
                    <strong>Strata Column:</strong> Optional column defining validation groups (e.g., site names, time periods)
                  </p>
                </div>
              </>
            )}

            {validationType === 'pnw_cnet' && (
              <>
                <h4 style={{ color: '#6e7cb9', marginTop: 0 }}>PNW-CNet Predictions Workflow</h4>
                <ol>
                  <li><strong>Configure Project:</strong> Set save location and project name</li>
                  <li><strong>Load PNW-CNet File:</strong> Provide path to default PNW-CNet prediction CSV</li>
                  <li><strong>Select Strata Field:</strong> Choose grouping (site-station, site, or region)</li>
                  <li><strong>Link Audio (Optional):</strong> Provide directory for audio playback during validation</li>
                  <li><strong>Create Dataset:</strong> System parses filenames and creates strata automatically</li>
                </ol>
                <div style={{
                  marginTop: '1rem',
                  padding: '1rem',
                  backgroundColor: '#e3f2fd',
                  border: '1px solid #bbdefb',
                  borderRadius: '4px'
                }}>
                  <h4 style={{ margin: '0 0 0.5rem 0', color: '#1565c0' }}>Filename Format:</h4>
                  <p style={{ margin: '0', color: '#1565c0' }}>
                    Expected: <code>REGION_SITE-STATION_YYYYMMDD_HHMMSS_part_NNN.png</code><br/>
                    Example: <code>JCN_40758-12_20220614_123409_part_001.png</code><br/>
                    Clip times calculated from part numbers (12-second windows)
                  </p>
                </div>
              </>
            )}

            {validationType === 'call_density' && (
              <>
                <h4 style={{ color: '#6e7cb9', marginTop: 0 }}>Call Density Estimation Workflow</h4>
                <ol>
                  <li><strong>Configure Project:</strong> Set save location and project name</li>
                  <li><strong>Select Audio Directory:</strong> Specify directory containing recordings</li>
                  <li><strong>Set Parameters:</strong> Define clip length, sampling interval, and clips per interval</li>
                  <li><strong>Specify Target Class:</strong> Enter single target species for density estimation</li>
                  <li><strong>Create Dataset:</strong> System generates systematically sampled clips</li>
                </ol>
                <div style={{
                  marginTop: '1rem',
                  padding: '1rem',
                  backgroundColor: '#e3f2fd',
                  border: '1px solid #bbdefb',
                  borderRadius: '4px'
                }}>
                  <p style={{ margin: '0', color: '#1565c0' }}>
                    <strong>Use case:</strong> Systematic temporal sampling for estimating call density metrics with unbiased clip selection.
                  </p>
                </div>
              </>
            )}
          </div>
        </div>
      )}

    </div>
  );
};

export default ValidationDatasetBuilder;