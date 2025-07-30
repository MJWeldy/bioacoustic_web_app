import React, { useState, useRef } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import Select from 'react-select';

const DatasetBuilder = () => {
  const [audioFolder, setAudioFolder] = useState('');
  const [savePath, setSavePath] = useState('');
  const [backendModel, setBackendModel] = useState('PERCH');
  const [classMap, setClassMap] = useState([{ name: '', value: 0 }]);
  const [isEvaluationDataset, setIsEvaluationDataset] = useState(false);
  const [classMapMode, setClassMapMode] = useState('manual'); // 'manual' or 'dictionary'
  const [classMapDict, setClassMapDict] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [datasetStatus, setDatasetStatus] = useState(null);
  const [buildingProgress, setBuildingProgress] = useState(null);
  const pollIntervalRef = useRef(null);

  const backendOptions = [
    { value: 'PERCH', label: 'PERCH (v8)' },
    { value: 'BirdNET_2.4', label: 'BirdNET 2.4' },
    { value: 'PNWCnet', label: 'PNWCnet' },
    { value: 'PNWCnet_EXPANDED', label: 'PNWCnet Expanded' },
  ];

  const addClassMapEntry = () => {
    const newValue = Math.max(...classMap.map(c => c.value), -1) + 1;
    setClassMap([...classMap, { name: '', value: newValue }]);
  };

  const removeClassMapEntry = (index) => {
    if (classMap.length > 1) {
      setClassMap(classMap.filter((_, i) => i !== index));
    }
  };

  const updateClassMapEntry = (index, field, value) => {
    const updated = [...classMap];
    updated[index][field] = field === 'value' ? parseInt(value) : value;
    setClassMap(updated);
  };

  const parseDictionary = () => {
    try {
      let cleanDict = classMapDict.trim();
      
      // First try to parse as JSON
      let dictData;
      try {
        dictData = JSON.parse(cleanDict);
      } catch (jsonError) {
        // If JSON parsing fails, try manual parsing for Python-style format
        if (cleanDict.startsWith('{') && cleanDict.endsWith('}')) {
          cleanDict = cleanDict.slice(1, -1);
        }
        
        const pairs = cleanDict.split(',').map(pair => pair.trim());
        dictData = {};
        
        for (const pair of pairs) {
          if (!pair) continue;
          
          const colonIndex = pair.indexOf(':');
          if (colonIndex === -1) continue;
          
          let key = pair.substring(0, colonIndex).trim();
          let value = pair.substring(colonIndex + 1).trim();
          
          // Remove quotes from key
          key = key.replace(/^['"]|['"]$/g, '');
          
          // Parse value as integer
          const numValue = parseInt(value);
          if (isNaN(numValue)) continue;
          
          dictData[key] = numValue;
        }
      }
      
      // Convert to array format
      const parsed = [];
      for (const [key, value] of Object.entries(dictData)) {
        if (typeof value === 'number') {
          parsed.push({ name: key, value: value });
        }
      }
      
      if (parsed.length > 0) {
        // Sort by value to maintain consistent ordering
        parsed.sort((a, b) => a.value - b.value);
        setClassMap(parsed);
        toast.success(`Parsed ${parsed.length} classes from dictionary`);
      } else {
        toast.error('No valid class entries found in dictionary');
      }
    } catch (error) {
      toast.error('Invalid dictionary format. Use JSON format: {"class1": 0, "class2": 1}');
    }
  };

  const convertToDict = () => {
    const dict = {};
    classMap.forEach(entry => {
      if (entry.name.trim()) {
        dict[entry.name] = entry.value;
      }
    });
    const dictStr = JSON.stringify(dict, null, 2).replace(/"/g, "'");
    setClassMapDict(dictStr);
  };

  const validateForm = () => {
    if (!audioFolder.trim()) {
      toast.error('Please specify an audio folder');
      return false;
    }
    if (!savePath.trim()) {
      toast.error('Please specify a save path');
      return false;
    }
    // Validate class map based on mode
    if (classMapMode === 'dictionary') {
      if (!classMapDict.trim()) {
        toast.error('Please enter a class map dictionary or switch to manual mode');
        return false;
      }
      // Try to parse the dictionary to validate it
      try {
        parseDictionary();
      } catch (error) {
        toast.error('Invalid dictionary format');
        return false;
      }
    } else {
      if (classMap.some(c => !c.name.trim())) {
        toast.error('Please fill in all class names');
        return false;
      }
    }
    return true;
  };

  const pollBuildingProgress = async () => {
    try {
      const response = await axios.get('/api/dataset/building-status');
      setBuildingProgress(response.data);
      
      if (response.data.status === 'completed' || response.data.status === 'error') {
        setIsLoading(false);
        if (response.data.status === 'completed') {
          toast.success(response.data.message);
          setDatasetStatus(response.data);
        } else {
          toast.error(response.data.message);
        }
        
        // Clear polling interval
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
        setBuildingProgress(null);
      }
    } catch (error) {
      console.error('Failed to check building status:', error);
    }
  };

  const createDataset = async () => {
    if (!validateForm()) return;

    setIsLoading(true);
    setBuildingProgress({ status: 'starting', message: 'Initializing dataset creation...' });
    
    try {
      // Ensure we have the latest class map from dictionary if in dictionary mode
      let finalClassMap = classMap;
      if (classMapMode === 'dictionary' && classMapDict.trim()) {
        parseDictionary(); // This updates the classMap state
        finalClassMap = classMap; // Use the updated classMap
      }

      const config = {
        audio_folder: audioFolder,
        class_map: finalClassMap,
        backend_model: backendModel,
        save_path: savePath,
        is_evaluation_dataset: isEvaluationDataset
      };

      const response = await axios.post('/api/dataset/create', config);
      
      if (response.data.status === 'started') {
        // Start polling for progress
        pollIntervalRef.current = setInterval(() => {
          pollBuildingProgress();
        }, 2000); // Poll every 2 seconds
        
        toast.info('Dataset creation started. This may take several minutes...');
      } else if (response.data.status === 'success') {
        // Immediate completion (e.g., existing embeddings)
        toast.success(response.data.message);
        setDatasetStatus(response.data);
        setIsLoading(false);
        setBuildingProgress(null);
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to create dataset';
      toast.error(message);
      setIsLoading(false);
      setBuildingProgress(null);
      
      // Clear polling interval if it was started
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    }
  };

  const checkDatasetStatus = async () => {
    try {
      const response = await axios.get('/api/dataset/status');
      setDatasetStatus(response.data);
    } catch (error) {
      console.error('Failed to check dataset status:', error);
    }
  };

  React.useEffect(() => {
    checkDatasetStatus();
  }, []);

  // Cleanup polling interval on unmount
  React.useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <h3>Dataset Builder</h3>
          <p>Create a new dataset with embeddings and class mappings for active learning</p>
        </div>

        {datasetStatus && datasetStatus.loaded && (
          <div style={{ 
            padding: '1.5rem', 
            backgroundColor: '#d0eaf1', 
            borderRadius: '8px', 
            marginBottom: '1.5rem',
            border: '1px solid #7bbcd5'
          }}>
            <strong style={{ color: '#6e7cb9', fontSize: '1.1rem' }}>Current Dataset Status</strong>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
              <div>
                <ul style={{ margin: '0', paddingLeft: '20px', color: '#333' }}>
                  <li><strong>Clips:</strong> {datasetStatus.clips_count}</li>
                  <li><strong>Backend Model:</strong> {datasetStatus.backend_model}</li>
                  <li><strong>Classes:</strong> {Object.keys(datasetStatus.class_map || {}).join(', ')}</li>
                </ul>
              </div>
              {datasetStatus.metadata && (
                <div>
                  <ul style={{ margin: '0', paddingLeft: '20px', color: '#333' }}>
                    <li><strong>Dataset Type:</strong> {datasetStatus.dataset_type}</li>
                    <li><strong>Created:</strong> {datasetStatus.creation_date ? new Date(datasetStatus.creation_date).toLocaleDateString() : 'Unknown'}</li>
                    <li><strong>Has Labels:</strong> {datasetStatus.has_labels ? 'Yes' : 'No'}</li>
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}

        {buildingProgress && (
          <div style={{ 
            padding: '1.5rem', 
            backgroundColor: '#d0eaf1', 
            borderRadius: '8px', 
            marginBottom: '1.5rem',
            border: '2px solid #7bbcd5'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '1rem' }}>
              <div style={{
                width: '24px',
                height: '24px',
                border: '3px solid #6e7cb9',
                borderTop: '3px solid transparent',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
              }}></div>
              <strong style={{ color: '#6e7cb9', fontSize: '1.1rem' }}>Building Dataset</strong>
            </div>
            <p style={{ margin: '0', color: '#333', fontSize: '1rem' }}>
              {buildingProgress.message || 'Processing audio files and generating embeddings...'}
            </p>
            {buildingProgress.progress && (
              <div style={{ marginTop: '0.75rem' }}>
                <div style={{ 
                  width: '100%', 
                  height: '8px', 
                  backgroundColor: '#e89c81', 
                  borderRadius: '4px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${buildingProgress.progress}%`,
                    height: '100%',
                    backgroundColor: '#6e7cb9',
                    transition: 'width 0.3s ease'
                  }}></div>
                </div>
                <small style={{ color: '#666', fontSize: '0.875rem' }}>
                  {buildingProgress.progress}% complete
                </small>
              </div>
            )}
          </div>
        )}

        <div className="grid grid-2">
          <div className="form-group">
            <label htmlFor="audioFolder">Audio Folder Path</label>
            <input
              type="text"
              id="audioFolder"
              className="form-control"
              placeholder="/path/to/audio/files"
              value={audioFolder}
              onChange={(e) => setAudioFolder(e.target.value)}
            />
            <small style={{ color: '#666', fontSize: '0.875rem' }}>
              Path to folder containing WAV or MP3 files
            </small>
          </div>

          <div className="form-group">
            <label htmlFor="savePath">Save Location</label>
            <input
              type="text"
              id="savePath"
              className="form-control"
              placeholder="/path/to/save/dataset"
              value={savePath}
              onChange={(e) => setSavePath(e.target.value)}
            />
            <small style={{ color: '#666', fontSize: '0.875rem' }}>
              Location to save embeddings and database
            </small>
          </div>
        </div>

        <div className="grid grid-2">
          <div className="form-group">
            <label htmlFor="backendModel">Backend Model</label>
            <Select
              options={backendOptions}
              value={backendOptions.find(opt => opt.value === backendModel)}
              onChange={(selected) => setBackendModel(selected.value)}
              isSearchable={false}
              styles={{
                control: (base) => ({
                  ...base,
                  border: '2px solid #e89c81',
                  '&:hover': { border: '2px solid #e89c81' },
                  '&:focus-within': { 
                    border: '2px solid #6e7cb9',
                    boxShadow: '0 0 0 3px rgba(110, 124, 185, 0.1)'
                  }
                })
              }}
            />
          </div>

          <div className="form-group">
            <label style={{ display: 'block', marginBottom: '0.5rem' }}>Dataset Type</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '0.5rem' }}>
              <input
                type="checkbox"
                id="isEvaluationDataset"
                checked={isEvaluationDataset}
                onChange={(e) => setIsEvaluationDataset(e.target.checked)}
                style={{ 
                  width: '18px', 
                  height: '18px',
                  accentColor: '#6e7cb9'
                }}
              />
              <label htmlFor="isEvaluationDataset" style={{ margin: 0, cursor: 'pointer' }}>
                Evaluation Dataset
              </label>
            </div>
            <small style={{ color: '#666', fontSize: '0.875rem', display: 'block', marginTop: '0.25rem' }}>
              Check if this dataset contains labeled audio files for evaluation
            </small>
          </div>
        </div>

        <div className="form-group">
          <label>Class Map</label>
          <div style={{ display: 'flex', gap: '20px', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <input
                type="radio"
                id="manualMode"
                name="classMapMode"
                value="manual"
                checked={classMapMode === 'manual'}
                onChange={(e) => setClassMapMode(e.target.value)}
                disabled={isLoading}
                style={{ accentColor: '#6e7cb9' }}
              />
              <label htmlFor="manualMode" style={{ margin: 0, cursor: 'pointer' }}>
                Build Manually
              </label>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <input
                type="radio"
                id="dictMode"
                name="classMapMode"
                value="dictionary"
                checked={classMapMode === 'dictionary'}
                onChange={(e) => setClassMapMode(e.target.value)}
                disabled={isLoading}
                style={{ accentColor: '#6e7cb9' }}
              />
              <label htmlFor="dictMode" style={{ margin: 0, cursor: 'pointer' }}>
                Paste Dictionary
              </label>
            </div>
          </div>

          {classMapMode === 'dictionary' ? (
            <div>
              <textarea
                className="form-control"
                placeholder={`{
  "bird_song": 0,
  "frog_call": 1,
  "insect_chirp": 2,
  "mammal_call": 3,
  "background_noise": 4
}`}
                value={classMapDict}
                onChange={(e) => setClassMapDict(e.target.value)}
                disabled={isLoading}
                rows={6}
                style={{
                  fontFamily: 'monospace',
                  fontSize: '0.875rem',
                  marginBottom: '10px'
                }}
              />
              <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
                <button
                  type="button"
                  onClick={parseDictionary}
                  className="btn btn-primary btn-sm"
                  disabled={isLoading || !classMapDict.trim()}
                >
                  Parse Dictionary
                </button>
                <button
                  type="button"
                  onClick={() => setClassMapMode('manual')}
                  className="btn btn-secondary btn-sm"
                  disabled={isLoading}
                >
                  Switch to Manual
                </button>
              </div>
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                Paste a Python dictionary format like {`{"bird": 0, "frog": 1, "insect": 2}`}
              </small>
            </div>
          ) : (
            <div>
              {classMap.map((entry, index) => (
                <div key={index} style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
                  <input
                    type="text"
                    className="form-control"
                    placeholder="Class name"
                    value={entry.name}
                    onChange={(e) => updateClassMapEntry(index, 'name', e.target.value)}
                    style={{ flex: 1 }}
                    disabled={isLoading}
                  />
                  <input
                    type="number"
                    className="form-control"
                    placeholder="Value"
                    value={entry.value}
                    onChange={(e) => updateClassMapEntry(index, 'value', e.target.value)}
                    style={{ width: '100px' }}
                    disabled={isLoading}
                  />
                  <button
                    type="button"
                    onClick={() => removeClassMapEntry(index)}
                    className="btn btn-danger btn-sm"
                    disabled={classMap.length === 1 || isLoading}
                  >
                    Remove
                  </button>
                </div>
              ))}
              <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
                <button
                  type="button"
                  onClick={addClassMapEntry}
                  className="btn btn-secondary btn-sm"
                  disabled={isLoading}
                >
                  Add Class
                </button>
                <button
                  type="button"
                  onClick={() => {
                    convertToDict();
                    setClassMapMode('dictionary');
                  }}
                  className="btn btn-primary btn-sm"
                  disabled={isLoading || classMap.some(c => !c.name.trim())}
                >
                  Convert to Dictionary
                </button>
              </div>
              <small style={{ color: '#666', fontSize: '0.875rem' }}>
                Build class map manually or convert to dictionary format for easier editing
              </small>
            </div>
          )}
        </div>


        <div style={{ textAlign: 'center', marginTop: '2rem' }}>
          <button
            onClick={createDataset}
            disabled={isLoading}
            className="btn btn-primary btn-lg"
          >
            {isLoading ? 'Creating Dataset...' : 'Create Dataset'}
          </button>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>Instructions</h3>
        </div>
        <div style={{ lineHeight: '1.6' }}>
          <ol>
            <li><strong>Audio Folder:</strong> Select a folder containing your audio files (WAV or MP3 format)</li>
            <li><strong>Class Map:</strong> Define the classes you want to classify. You can either:
              <ul style={{ marginTop: '8px' }}>
                <li><strong>Build Manually:</strong> Add classes one by one with names and values</li>
                <li><strong>Paste Dictionary:</strong> Copy/paste a Python dictionary format for multiclass problems</li>
              </ul>
            </li>
            <li><strong>Backend Model:</strong> Choose the embedding model:
              <ul style={{ marginTop: '8px' }}>
                <li><strong>PERCH:</strong> Google's PERCH model for bird vocalizations</li>
                <li><strong>BirdNET:</strong> BirdNET 2.4 model for bird sounds</li>
                <li><strong>PNWCnet:</strong> Pacific Northwest focused model</li>
              </ul>
            </li>
            <li><strong>Save Location:</strong> Choose where to store the embeddings and database files</li>
            <li><strong>Evaluation Dataset:</strong> Check this if your audio files contain class labels in their filenames for evaluation purposes</li>
          </ol>
          <p><strong>Evaluation Datasets:</strong> When creating an evaluation dataset, labels will be automatically extracted from filenames using the class map. The embeddings file will contain both embeddings and labels for evaluation purposes.</p>
          <p><strong>Note:</strong> Dataset creation may take several minutes depending on the number and size of audio files.</p>
        </div>
      </div>
    </div>
  );
};

export default DatasetBuilder;