import React, { useState } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';

const ModelTraining = () => {
  const [trainingAudioFolder, setTrainingAudioFolder] = useState('');
  const [metadataPath, setMetadataPath] = useState('');
  const [testDataMode, setTestDataMode] = useState('split'); // 'split' or 'folder'
  const [testSplit, setTestSplit] = useState(0.2);
  const [testAudioFolder, setTestAudioFolder] = useState('');
  const [randomState, setRandomState] = useState(42);
  const [modelSavePath, setModelSavePath] = useState('');
  
  // Training parameters
  const [nSteps, setNSteps] = useState(1000);
  const [batchSize, setBatchSize] = useState(128);
  const [learningRate, setLearningRate] = useState(0.001);
  const [modelType, setModelType] = useState(2);
  const [verbose, setVerbose] = useState(true);
  
  const [isLoading, setIsLoading] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingResults, setTrainingResults] = useState(null);
  const [trainingLogs, setTrainingLogs] = useState([]);

  const modelTypeOptions = [
    { value: 1, label: 'Type 1 - Basic' },
    { value: 2, label: 'Type 2 - With Dropout' },
    { value: 3, label: 'Type 3 - Single Dense Layer' },
    { value: 4, label: 'Type 4 - Dense + Dropout' },
    { value: 5, label: 'Type 5 - Large Dense (2048)' },
    { value: 6, label: 'Type 6 - Large Dense + Dropout' },
    { value: 7, label: 'Type 7 - Medium Dense (512)' },
    { value: 8, label: 'Type 8 - Medium Dense + Dropout' },
  ];

  const validateForm = () => {
    if (!trainingAudioFolder.trim()) {
      toast.error('Please specify a training audio folder');
      return false;
    }
    if (!metadataPath.trim()) {
      toast.error('Please specify a metadata file path');
      return false;
    }
    if (!modelSavePath.trim()) {
      toast.error('Please specify a model save path');
      return false;
    }
    if (testDataMode === 'split') {
      if (testSplit < 0 || testSplit > 1) {
        toast.error('Test split must be between 0 and 1');
        return false;
      }
    } else if (testDataMode === 'folder') {
      if (!testAudioFolder.trim()) {
        toast.error('Please specify a test audio folder');
        return false;
      }
    }
    return true;
  };

  const startTraining = async () => {
    if (!validateForm()) return;

    setIsLoading(true);
    setTrainingLogs([]);
    setTrainingResults(null);
    
    try {
      const config = {
        training_audio_folder: trainingAudioFolder,
        metadata_path: metadataPath,
        test_data_mode: testDataMode,
        test_split: testDataMode === 'split' ? testSplit : null,
        test_audio_folder: testDataMode === 'folder' ? testAudioFolder : null,
        random_state: randomState,
        model_save_path: modelSavePath,
        training_params: {
          n_steps: nSteps,
          batch_size: batchSize,
          learning_rate: learningRate,
          model_type: modelType,
          verbose: verbose
        }
      };

      const response = await axios.post('/api/model-training/start', config);
      
      if (response.data.status === 'started') {
        toast.success('Model training started!');
        setTrainingStatus('training');
        // Start polling for training progress
        pollTrainingStatus();
      } else if (response.data.status === 'success') {
        toast.success('Model training completed!');
        setTrainingResults(response.data.results);
        setTrainingStatus('completed');
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to start training';
      toast.error(message);
      setTrainingStatus('error');
    } finally {
      setIsLoading(false);
    }
  };

  const pollTrainingStatus = async () => {
    try {
      const response = await axios.get('/api/model-training/status');
      const data = response.data;
      
      setTrainingStatus(data.status);
      
      if (data.logs && data.logs.length > 0) {
        setTrainingLogs(data.logs);
      }
      
      if (data.status === 'completed') {
        toast.success('Model training completed successfully!');
        setTrainingResults(data.results);
        setIsLoading(false);
      } else if (data.status === 'error') {
        toast.error(data.message || 'Training failed');
        setIsLoading(false);
      } else if (data.status === 'training') {
        // Continue polling
        setTimeout(pollTrainingStatus, 3000);
      }
    } catch (error) {
      console.error('Failed to check training status:', error);
      setTimeout(pollTrainingStatus, 5000); // Retry after longer delay
    }
  };

  const stopTraining = async () => {
    try {
      await axios.post('/api/model-training/stop');
      toast.info('Training stop requested');
      setTrainingStatus('stopping');
    } catch (error) {
      toast.error('Failed to stop training');
    }
  };

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <h3>Model Training</h3>
          <p>Train a new classifier using the fit_w_tape function</p>
        </div>

        <div className="grid grid-2">
          <div className="form-group">
            <label htmlFor="trainingAudioFolder">Training Audio Folder</label>
            <input
              type="text"
              id="trainingAudioFolder"
              className="form-control"
              placeholder="/path/to/training/audio/files"
              value={trainingAudioFolder}
              onChange={(e) => setTrainingAudioFolder(e.target.value)}
              disabled={isLoading}
            />
            <small style={{ color: '#666', fontSize: '0.875rem' }}>
              Folder containing labeled audio files for training
            </small>
          </div>

          <div className="form-group">
            <label htmlFor="metadataPath">Dataset Metadata File</label>
            <input
              type="text"
              id="metadataPath"
              className="form-control"
              placeholder="/path/to/dataset/metadata.json"
              value={metadataPath}
              onChange={(e) => setMetadataPath(e.target.value)}
              disabled={isLoading}
            />
            <small style={{ color: '#666', fontSize: '0.875rem' }}>
              Metadata file from existing dataset for backend settings and class map
            </small>
          </div>
        </div>

        <div className="form-group">
          <label>Test Data Configuration</label>
          <div style={{ display: 'flex', gap: '20px', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <input
                type="radio"
                id="testSplit"
                name="testDataMode"
                value="split"
                checked={testDataMode === 'split'}
                onChange={(e) => setTestDataMode(e.target.value)}
                disabled={isLoading}
                style={{ accentColor: '#6e7cb9' }}
              />
              <label htmlFor="testSplit" style={{ margin: 0, cursor: 'pointer' }}>
                Use Train/Test Split
              </label>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <input
                type="radio"
                id="testFolder"
                name="testDataMode"
                value="folder"
                checked={testDataMode === 'folder'}
                onChange={(e) => setTestDataMode(e.target.value)}
                disabled={isLoading}
                style={{ accentColor: '#6e7cb9' }}
              />
              <label htmlFor="testFolder" style={{ margin: 0, cursor: 'pointer' }}>
                Use Separate Test Folder
              </label>
            </div>
          </div>

          {testDataMode === 'split' ? (
            <div className="grid grid-2">
              <div className="form-group">
                <label htmlFor="testSplitValue">Test Data Split</label>
                <input
                  type="number"
                  id="testSplitValue"
                  className="form-control"
                  min="0"
                  max="1"
                  step="0.05"
                  value={testSplit}
                  onChange={(e) => setTestSplit(parseFloat(e.target.value))}
                  disabled={isLoading}
                />
                <small style={{ color: '#666', fontSize: '0.875rem' }}>
                  Fraction of data to use for validation (0.0 - 1.0)
                </small>
              </div>

              <div className="form-group">
                <label htmlFor="randomState">Random State</label>
                <input
                  type="number"
                  id="randomState"
                  className="form-control"
                  value={randomState}
                  onChange={(e) => setRandomState(parseInt(e.target.value))}
                  disabled={isLoading}
                />
                <small style={{ color: '#666', fontSize: '0.875rem' }}>
                  Random seed for reproducible train/test splits
                </small>
              </div>
            </div>
          ) : (
            <div className="form-group">
              <label htmlFor="testAudioFolder">Test Audio Folder</label>
              <input
                type="text"
                id="testAudioFolder"
                className="form-control"
                placeholder="/path/to/test/audio/files"
                value={testAudioFolder}
                onChange={(e) => setTestAudioFolder(e.target.value)}
                disabled={isLoading}
              />
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                Folder containing labeled audio files for testing/validation
              </small>
            </div>
          )}
        </div>

        <div className="form-group">
          <label htmlFor="modelSavePath">Model Save Path</label>
          <input
            type="text"
            id="modelSavePath"
            className="form-control"
            placeholder="/path/to/save/my_classifier_model.keras"
            value={modelSavePath}
            onChange={(e) => setModelSavePath(e.target.value)}
            disabled={isLoading}
          />
          <small style={{ color: '#666', fontSize: '0.875rem' }}>
            Full file path for saving the trained model (include .keras extension, e.g., /path/to/my_model.keras)
          </small>
        </div>

        <div className="card" style={{ marginTop: '1.5rem' }}>
          <div className="card-header">
            <h4>Training Parameters</h4>
          </div>
          
          <div className="grid grid-2">
            <div className="form-group">
              <label htmlFor="nSteps">Training Steps</label>
              <input
                type="number"
                id="nSteps"
                className="form-control"
                min="100"
                max="10000"
                value={nSteps}
                onChange={(e) => setNSteps(parseInt(e.target.value))}
                disabled={isLoading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="batchSize">Batch Size</label>
              <input
                type="number"
                id="batchSize"
                className="form-control"
                min="1"
                max="512"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value))}
                disabled={isLoading}
              />
            </div>
          </div>

          <div className="grid grid-2">
            <div className="form-group">
              <label htmlFor="learningRate">Learning Rate</label>
              <input
                type="number"
                id="learningRate"
                className="form-control"
                min="0.0001"
                max="1.0"
                step="0.0001"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                disabled={isLoading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="modelType">Model Architecture</label>
              <select
                id="modelType"
                className="form-control"
                value={modelType}
                onChange={(e) => setModelType(parseInt(e.target.value))}
                disabled={isLoading}
              >
                {modelTypeOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="form-group">
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <input
                type="checkbox"
                id="verbose"
                checked={verbose}
                onChange={(e) => setVerbose(e.target.checked)}
                disabled={isLoading}
                style={{ 
                  width: '18px', 
                  height: '18px',
                  accentColor: '#6e7cb9'
                }}
              />
              <label htmlFor="verbose" style={{ margin: 0, cursor: 'pointer' }}>
                Verbose Output
              </label>
            </div>
            <small style={{ color: '#666', fontSize: '0.875rem', display: 'block', marginTop: '0.25rem' }}>
              Show detailed training progress. If unchecked, only final loss and cMAP will be shown.
            </small>
          </div>
        </div>

        <div style={{ textAlign: 'center', marginTop: '2rem' }}>
          {trainingStatus !== 'training' ? (
            <button
              onClick={startTraining}
              disabled={isLoading}
              className="btn btn-primary btn-lg"
            >
              {isLoading ? 'Starting Training...' : 'Start Training'}
            </button>
          ) : (
            <button
              onClick={stopTraining}
              className="btn btn-danger btn-lg"
            >
              Stop Training
            </button>
          )}
        </div>
      </div>

      {trainingStatus && trainingStatus !== 'idle' && (
        <div className="card">
          <div className="card-header">
            <h3>Training Status</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div className={`status-indicator ${
                trainingStatus === 'training' ? 'status-warning' : 
                trainingStatus === 'completed' ? 'status-success' : 'status-error'
              }`}></div>
              <span style={{ textTransform: 'capitalize', fontWeight: '600' }}>
                {trainingStatus}
              </span>
            </div>
          </div>

          {trainingLogs.length > 0 && (
            <div>
              <h4>Training Output</h4>
              <div style={{ 
                backgroundColor: '#f8f9fa', 
                border: '1px solid #e89c81', 
                borderRadius: '4px', 
                padding: '1rem',
                maxHeight: '400px',
                overflowY: 'auto',
                fontFamily: 'monospace',
                fontSize: '0.875rem',
                whiteSpace: 'pre-wrap'
              }}>
                {trainingLogs.join('\n')}
              </div>
            </div>
          )}
        </div>
      )}

      {trainingResults && (
        <div className="card">
          <div className="card-header">
            <h3>Training Results</h3>
          </div>
          <div className="grid grid-2">
            <div>
              <h4>Final Metrics</h4>
              <ul style={{ margin: '0', paddingLeft: '20px' }}>
                <li><strong>Final Loss:</strong> {
                  trainingResults.final_loss !== null && trainingResults.final_loss !== undefined && !isNaN(trainingResults.final_loss) 
                    ? trainingResults.final_loss.toFixed(6) 
                    : 'N/A'
                }</li>
                <li><strong>Best Macro cMAP:</strong> {
                  trainingResults.best_cmap !== null && trainingResults.best_cmap !== undefined && !isNaN(trainingResults.best_cmap)
                    ? trainingResults.best_cmap.toFixed(6) 
                    : 'N/A'
                }</li>
                <li><strong>Training Steps:</strong> {trainingResults.total_steps || 'N/A'}</li>
                <li><strong>Model Saved:</strong> {trainingResults.model_path || 'N/A'}</li>
              </ul>
            </div>
            <div>
              <h4>Training Parameters</h4>
              <ul style={{ margin: '0', paddingLeft: '20px' }}>
                <li><strong>Batch Size:</strong> {trainingResults.batch_size || 'N/A'}</li>
                <li><strong>Learning Rate:</strong> {trainingResults.learning_rate || 'N/A'}</li>
                <li><strong>Model Type:</strong> {trainingResults.model_type || 'N/A'}</li>
                <li><strong>Training Samples:</strong> {trainingResults.train_samples || 'N/A'}</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <h3>Instructions</h3>
        </div>
        <div style={{ lineHeight: '1.6' }}>
          <ol>
            <li><strong>Training Audio Folder:</strong> Select folder containing labeled audio files</li>
            <li><strong>Metadata File:</strong> Load metadata.json from existing dataset for backend settings</li>
            <li><strong>Configure Parameters:</strong> Set test split, model architecture, and training parameters</li>
            <li><strong>Start Training:</strong> Begin training process with fit_w_tape function</li>
            <li><strong>Monitor Progress:</strong> Watch training logs and metrics in real-time</li>
          </ol>
          <p><strong>Note:</strong> Training may take several minutes to hours depending on dataset size and parameters. The trained model will be saved to the specified path.</p>
        </div>
      </div>
    </div>
  );
};

export default ModelTraining;