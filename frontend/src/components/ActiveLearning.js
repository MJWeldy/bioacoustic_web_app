import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import { Range } from 'react-range';
import Select from 'react-select';

const ActiveLearning = ({ isActive = true }) => {
  const [datasetPath, setDatasetPath] = useState('');
  const [classifierPath, setClassifierPath] = useState('');
  const [isDatasetLoaded, setIsDatasetLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [scoreRange, setScoreRange] = useState([0.0, 1.0]);
  const [colorMode, setColorMode] = useState('viridis');
  const [currentClip, setCurrentClip] = useState(null);
  const [clips, setClips] = useState([]);
  const [currentClipIndex, setCurrentClipIndex] = useState(0);
  const [spectrogram, setSpectrogram] = useState(null);
  const [datasetMetadata, setDatasetMetadata] = useState(null);
  const [availableClasses, setAvailableClasses] = useState([]);
  const [selectedClass, setSelectedClass] = useState(null);
  const [currentClassIndex, setCurrentClassIndex] = useState(0);
  const [labelStatistics, setLabelStatistics] = useState(null);
  const [clipLabels, setClipLabels] = useState(null);
  const [reviewMode, setReviewMode] = useState('random');
  const audioRef = useRef(null);

  const colorModeOptions = [
    { value: 'viridis', label: 'Viridis (Color)' },
    { value: 'gray_r', label: 'Grayscale' },
    { value: 'plasma', label: 'Plasma' },
    { value: 'inferno', label: 'Inferno' },
  ];

  const reviewModeOptions = [
    { value: 'random', label: 'Random' },
    { value: 'top_down', label: 'Top-down (Highest Score First)' },
  ];

  const loadAvailableClasses = async () => {
    try {
      const response = await axios.get('/api/active-learning/classes');
      const classes = response.data.classes.map(cls => ({
        value: cls.value,
        label: cls.name
      }));
      setAvailableClasses(classes);
      
      // Set default selected class
      if (classes.length > 0) {
        const defaultClass = classes[0];
        setSelectedClass(defaultClass);
        setCurrentClassIndex(defaultClass.value);
      }
      
      return classes;
    } catch (error) {
      console.error('Failed to load classes:', error);
      return [];
    }
  };

  const selectClass = async (classIndex) => {
    try {
      setIsLoading(true);
      const response = await axios.post('/api/active-learning/select-class', null, {
        params: { class_index: classIndex }
      });
      
      if (response.data.status === 'success') {
        setCurrentClassIndex(classIndex);
        toast.success(response.data.message);
        
        // Refresh clips to show updated scores for the selected class
        await getClips();
        
        // Update label statistics
        await loadLabelStatistics();
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to select class';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const loadLabelStatistics = async () => {
    try {
      const response = await axios.get('/api/active-learning/label-statistics');
      if (response.data.status === 'success') {
        setLabelStatistics(response.data.statistics);
      }
    } catch (error) {
      console.error('Failed to load label statistics:', error);
    }
  };

  const populateEmbeddingIndices = async () => {
    try {
      setIsLoading(true);
      const response = await axios.post('/api/active-learning/populate-embedding-indices');
      
      if (response.data.status === 'success') {
        toast.success(response.data.message);
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to populate embedding indices';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const loadClipLabels = async (clipId) => {
    if (!clipId || availableClasses.length <= 1) return;
    
    try {
      const response = await axios.get('/api/active-learning/clip-labels', {
        params: { clip_id: clipId }
      });
      if (response.data.status === 'success') {
        setClipLabels(response.data.class_labels);
      }
    } catch (error) {
      console.error('Failed to load clip labels:', error);
      setClipLabels(null);
    }
  };

  const markOtherClassesAsAbsent = async () => {
    if (!currentClip || availableClasses.length <= 1) return;
    
    try {
      const request = {
        clip_id: currentClip.clip_id,
        annotation: 0  // The annotation value doesn't matter for this endpoint
      };

      const response = await axios.post('/api/active-learning/annotate-other-classes', request);
      
      if (response.data.status === 'success') {
        toast.success(response.data.message);
        
        // Refresh clip labels and statistics
        await loadClipLabels(currentClip.clip_id);
        await loadLabelStatistics();
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to mark other classes as absent';
      toast.error(message);
    }
  };

  const changeReviewMode = async (newReviewMode) => {
    try {
      const response = await axios.post('/api/active-learning/set-review-mode', {
        review_mode: newReviewMode
      });
      
      if (response.data.status === 'success') {
        setReviewMode(newReviewMode);
        toast.success(response.data.message);
        
        // Refresh clips with new review mode
        await getClips();
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to set review mode';
      toast.error(message);
    }
  };

  const loadDataset = async () => {
    if (!datasetPath.trim()) {
      toast.error('Please specify a dataset path');
      return;
    }

    setIsLoading(true);
    try {
      const response = await axios.post('/api/active-learning/load-dataset', null, {
        params: { dataset_path: datasetPath }
      });
      
      if (response.data.status === 'success') {
        toast.success(response.data.message);
        setIsDatasetLoaded(true);
        setDatasetMetadata(response.data.metadata);
        
        // Load available classes for multiclass support
        await loadAvailableClasses();
        
        // Load label statistics
        await loadLabelStatistics();
        
        await getClips();
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to load dataset';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const loadClassifier = async () => {
    if (!classifierPath.trim()) {
      toast.error('Please specify a classifier path');
      return;
    }

    setIsLoading(true);
    try {
      const response = await axios.post('/api/active-learning/load-classifier', null, {
        params: { classifier_path: classifierPath }
      });
      
      if (response.data.status === 'success') {
        toast.success(response.data.message);
        await getClips(); // Refresh clips with updated scores
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to load classifier';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const getClips = async () => {
    if (!isDatasetLoaded) return;

    try {
      const filterConfig = {
        score_min: scoreRange[0],
        score_max: scoreRange[1],
        annotation_filter: [4] // Only unreviewed clips
      };

      const response = await axios.post('/api/active-learning/get-clips', filterConfig);
      setClips(response.data.clips);
      
      // Use the next clip from the API if available, otherwise use the first clip
      if (response.data.next_clip) {
        setCurrentClip(response.data.next_clip);
        await loadClip(response.data.next_clip);
      } else if (response.data.clips.length > 0) {
        setCurrentClipIndex(0);
        await loadClip(response.data.clips[0]);
      } else {
        // Don't set currentClip to null immediately - let user know but keep current clip visible
        console.log('DEBUG: No more clips available with current filters');
        toast.info('No more clips match the current filters. You can adjust the score range or annotation filters to see more clips.');
        // Keep the current clip displayed so user can still see what they just annotated
      }
    } catch (error) {
      toast.error('Failed to load clips');
    }
  };

  const loadClip = async (clip) => {
    console.log('DEBUG: Loading clip with data:', clip);
    
    // Ensure clip has clip_id property - construct it if missing
    if (!clip.clip_id && clip.file_path && clip.clip_start !== undefined && clip.clip_end !== undefined) {
      clip.clip_id = `${clip.file_path}|${clip.clip_start}|${clip.clip_end}`;
      console.log('DEBUG: Constructed clip_id:', clip.clip_id);
    }
    
    setCurrentClip(clip);
    setIsLoading(true);

    try {
      // Generate spectrogram
      const spectrogramRequest = {
        file_path: clip.file_path,
        clip_start: clip.clip_start,
        clip_end: clip.clip_end,
        color_mode: colorMode
      };

      const spectrogramResponse = await axios.post('/api/spectrogram', spectrogramRequest);
      setSpectrogram(spectrogramResponse.data.spectrogram);

      // Load audio
      if (audioRef.current) {
        const audioUrl = `/api/audio/${encodeURIComponent(clip.file_path)}?clip_start=${clip.clip_start}&clip_end=${clip.clip_end}`;
        audioRef.current.src = audioUrl;
        audioRef.current.load();
      }

      // Load clip labels for multiclass view
      await loadClipLabels(clip.clip_id);
    } catch (error) {
      toast.error('Failed to load clip');
    } finally {
      setIsLoading(false);
    }
  };

  const annotateClip = async (annotation) => {
    if (!currentClip) {
      console.log('DEBUG: No currentClip available');
      toast.error('No clip selected for annotation');
      return;
    }

    if (!currentClip.clip_id) {
      console.log('DEBUG: currentClip missing clip_id:', currentClip);
      toast.error('Invalid clip data - missing clip_id');
      return;
    }

    try {
      const request = {
        clip_id: currentClip.clip_id,
        annotation: annotation
      };

      console.log('DEBUG: Sending annotation request:', request);
      console.log('DEBUG: currentClip state:', currentClip);

      await axios.post('/api/active-learning/annotate', request);
      
      const annotationText = annotation === 0 ? 'Not Present' : 
                           annotation === 1 ? 'Present' : 'Uncertain';
      toast.success(`Clip annotated as: ${annotationText}`);

      // Update label statistics and clip labels after annotation
      await loadLabelStatistics();
      await loadClipLabels(currentClip.clip_id);

      // Move to next clip
      await nextClip();
    } catch (error) {
      toast.error('Failed to annotate clip');
    }
  };

  const nextClip = async () => {
    if (currentClipIndex < clips.length - 1) {
      const newIndex = currentClipIndex + 1;
      setCurrentClipIndex(newIndex);
      await loadClip(clips[newIndex]);
    } else {
      // For review modes that have a specific order, refresh to get the next clip
      await getClips();
    }
  };

  const previousClip = async () => {
    if (currentClipIndex > 0) {
      const newIndex = currentClipIndex - 1;
      setCurrentClipIndex(newIndex);
      await loadClip(clips[newIndex]);
    }
  };

  const saveDatabase = async () => {
    try {
      const response = await axios.post('/api/active-learning/save-database');
      if (response.data.status === 'success') {
        toast.success('Database saved successfully');
      }
    } catch (error) {
      toast.error('Failed to save database');
    }
  };

  const exportClips = async () => {
    const exportPath = prompt('Enter export path:');
    const annotationSlug = prompt('Enter annotation slug:');
    
    if (!exportPath || !annotationSlug) return;

    try {
      const response = await axios.post('/api/active-learning/export-clips', null, {
        params: { 
          export_path: exportPath,
          annotation_slug: annotationSlug 
        }
      });
      
      if (response.data.status === 'success') {
        toast.success(response.data.message);
      }
    } catch (error) {
      toast.error('Failed to export clips');
    }
  };

  useEffect(() => {
    if (currentClip && colorMode) {
      loadClip(currentClip);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colorMode]);

  useEffect(() => {
    if (isDatasetLoaded) {
      getClips();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scoreRange]);

  return (
    <div style={{ display: isActive ? 'block' : 'none' }}>
      <div className="card">
        <div className="card-header">
          <h3>Active Learning</h3>
          <p>Load a dataset and perform interactive annotation</p>
        </div>

        <div className="grid grid-2">
          <div className="form-group">
            <label htmlFor="datasetPath">Dataset Path</label>
            <div style={{ display: 'flex', gap: '10px' }}>
              <input
                type="text"
                id="datasetPath"
                className="form-control"
                placeholder="/path/to/dataset"
                value={datasetPath}
                onChange={(e) => setDatasetPath(e.target.value)}
                style={{ flex: 1 }}
              />
              <button
                onClick={loadDataset}
                disabled={isLoading}
                className="btn btn-primary"
              >
                Load Dataset
              </button>
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="classifierPath">Pretrained Classifier (Optional)</label>
            <div style={{ display: 'flex', gap: '10px' }}>
              <input
                type="text"
                id="classifierPath"
                className="form-control"
                placeholder="/path/to/classifier.keras"
                value={classifierPath}
                onChange={(e) => setClassifierPath(e.target.value)}
                style={{ flex: 1 }}
              />
              <button
                onClick={loadClassifier}
                disabled={isLoading || !isDatasetLoaded}
                className="btn btn-secondary"
              >
                Load Classifier
              </button>
            </div>
          </div>
        </div>

        {isDatasetLoaded && (
          <div style={{ marginBottom: '1.5rem', padding: '10px', backgroundColor: '#f8f9fa', borderRadius: '6px', border: '1px solid #ddd' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <strong style={{ color: '#6e7cb9' }}>Vector Search Setup</strong>
                <p style={{ margin: '5px 0 0 0', fontSize: '0.9rem', color: '#666' }}>
                  Populate embedding indices to enable vector similarity search functionality.
                </p>
              </div>
              <button
                onClick={populateEmbeddingIndices}
                disabled={isLoading}
                className="btn btn-info"
                style={{ minWidth: '140px' }}
              >
                {isLoading ? 'Processing...' : 'Fix Embedding Indices'}
              </button>
            </div>
          </div>
        )}

        {isDatasetLoaded && (
          <div className="form-group" style={{ marginBottom: '1.5rem' }}>
            <label htmlFor="reviewMode">Review Mode</label>
            <Select
              id="reviewMode"
              options={reviewModeOptions}
              value={reviewModeOptions.find(option => option.value === reviewMode)}
              onChange={(selected) => changeReviewMode(selected.value)}
              isSearchable={false}
              isDisabled={isLoading}
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
            <small style={{ color: '#666', fontSize: '0.875rem' }}>
              Select how clips should be presented for annotation. Top-down mode shows unreviewed clips with highest scores first.
            </small>
          </div>
        )}

        {isDatasetLoaded && (
          <div>

            {labelStatistics && (
              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <div style={{ 
                  padding: '1rem', 
                  backgroundColor: '#d0eaf1', 
                  borderRadius: '8px', 
                  border: '1px solid #7bbcd5'
                }}>
                  <h4 style={{ margin: '0 0 0.5rem 0', color: '#6e7cb9' }}>Label Statistics</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                    <div>
                      <strong>Overall:</strong>
                      <ul style={{ margin: '0.25rem 0', paddingLeft: '20px' }}>
                        <li>Total clips: {labelStatistics.total_clips}</li>
                        <li>Strong labels: {labelStatistics.clips_with_strong_labels}</li>
                        <li>Weak labels only: {labelStatistics.clips_with_only_weak_labels}</li>
                      </ul>
                    </div>
                    {availableClasses.length > 1 && (
                      <div>
                        <strong>Current Class ({selectedClass?.label}):</strong>
                        <ul style={{ margin: '0.25rem 0', paddingLeft: '20px' }}>
                          <li>Strong: {labelStatistics.per_class_statistics[`class_${currentClassIndex}`]?.strong_labels || 0}</li>
                          <li>Weak: {labelStatistics.per_class_statistics[`class_${currentClassIndex}`]?.weak_labels || 0}</li>
                        </ul>
                      </div>
                    )}
                  </div>
                  <small style={{ color: '#666', fontSize: '0.875rem', display: 'block', marginTop: '0.5rem' }}>
                    <strong>Strong labels:</strong> Explicitly marked as Present/Not Present. 
                    <strong>Weak labels:</strong> Not yet confirmed for this class.
                  </small>
                </div>
              </div>
            )}
            
            <div className="grid grid-2">
              <div className="form-group">
                <label>Score Range</label>
                <div style={{ padding: '20px 10px' }}>
                  <Range
                    step={0.01}
                    min={0}
                    max={1}
                    values={scoreRange}
                    onChange={(values) => setScoreRange(values)}
                    renderTrack={({ props, children }) => (
                      <div
                        {...props}
                        style={{
                          ...props.style,
                          height: '6px',
                          width: '100%',
                          backgroundColor: '#e89c81',
                          borderRadius: '3px'
                        }}
                      >
                        {children}
                      </div>
                    )}
                    renderThumb={({ props }) => (
                      <div
                        {...props}
                        style={{
                          ...props.style,
                          height: '20px',
                          width: '20px',
                          backgroundColor: '#6e7cb9',
                          borderRadius: '50%'
                        }}
                      />
                    )}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px' }}>
                    <span>{scoreRange[0].toFixed(2)}</span>
                    <span>{scoreRange[1].toFixed(2)}</span>
                  </div>
                </div>
              </div>

              <div className="form-group">
                <label>Spectrogram Color Mode</label>
                <Select
                  options={colorModeOptions}
                  value={colorModeOptions.find(opt => opt.value === colorMode)}
                  onChange={(selected) => setColorMode(selected.value)}
                  isSearchable={false}
                />
              </div>
            </div>

            <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
              <button onClick={saveDatabase} className="btn btn-success">
                Save Database
              </button>
              <button onClick={exportClips} className="btn btn-warning">
                Export Clips
              </button>
            </div>
          </div>
        )}

        {isDatasetLoaded && datasetMetadata && (
          <div style={{ 
            padding: '1rem', 
            backgroundColor: '#f8f9fa', 
            borderRadius: '6px',
            marginTop: '1rem',
            border: '1px solid #e89c81'
          }}>
            <strong style={{ color: '#6e7cb9' }}>Dataset Information:</strong>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginTop: '0.5rem', fontSize: '0.9rem' }}>
              <div><strong>Type:</strong> {datasetMetadata.dataset_info?.dataset_type || 'Unknown'}</div>
              <div><strong>Model:</strong> {datasetMetadata.dataset_info?.backend_model || 'Unknown'}</div>
              <div><strong>Created:</strong> {datasetMetadata.dataset_info?.creation_date ? new Date(datasetMetadata.dataset_info.creation_date).toLocaleDateString() : 'Unknown'}</div>
              <div><strong>Classes:</strong> {Object.keys(datasetMetadata.class_map || {}).join(', ')}</div>
            </div>
          </div>
        )}
      </div>

      {currentClip && (
        <div className="card">
          <div className="card-header">
            <h3>Current Clip</h3>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <p style={{ margin: 0 }}>
                  <strong>File:</strong> {currentClip.file_name} | 
                  <strong> Time:</strong> {currentClip.clip_start.toFixed(1)}s - {currentClip.clip_end.toFixed(1)}s | 
                  <strong> Score:</strong> {currentClip.score.toFixed(3)} |
                  <strong> Clip:</strong> {currentClipIndex + 1} of {clips.length}
                </p>
              </div>
              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  onClick={previousClip}
                  disabled={currentClipIndex === 0}
                  className="btn btn-secondary btn-sm"
                >
                  ← Previous
                </button>
                <button
                  onClick={nextClip}
                  disabled={currentClipIndex >= clips.length - 1}
                  className="btn btn-secondary btn-sm"
                >
                  Next →
                </button>
              </div>
            </div>
            
            {/* Clip Labels Information */}
            {clipLabels && availableClasses.length > 1 && (
              <div style={{ 
                marginTop: '1rem', 
                padding: '0.75rem', 
                backgroundColor: '#f8f9fa', 
                borderRadius: '6px',
                border: '1px solid #ddd'
              }}>
                <h5 style={{ margin: '0 0 0.5rem 0', fontSize: '0.9rem', color: '#6e7cb9' }}>
                  Class Labels for This Clip:
                </h5>
                <div style={{ 
                  display: 'grid', 
                  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
                  gap: '0.5rem',
                  fontSize: '0.8rem'
                }}>
                  {clipLabels.map((classLabel, index) => (
                    <div 
                      key={index}
                      style={{ 
                        padding: '0.25rem 0.5rem',
                        backgroundColor: classLabel.is_current ? '#d0eaf1' : 'white',
                        border: classLabel.is_current ? '1px solid #7bbcd5' : '1px solid #e9ecef',
                        borderRadius: '4px',
                        fontSize: '0.75rem'
                      }}
                    >
                      <strong>{classLabel.class_name}:</strong> {classLabel.label_text}
                      {classLabel.is_current && ' (current)'}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div>
            <h4>Spectrogram</h4>
            {spectrogram ? (
              <img 
                src={spectrogram} 
                alt="Spectrogram" 
                style={{ 
                  width: '100%', 
                  height: '600px', 
                  objectFit: 'contain',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  backgroundColor: '#f8f9fa'
                }}
              />
            ) : (
              <div className="loading" style={{ height: '600px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                Loading spectrogram...
              </div>
            )}
          </div>

          <div className="grid grid-2" style={{ alignItems: 'start', marginTop: '20px' }}>
            <div>
              <h4>Audio Player</h4>
              <audio
                ref={audioRef}
                controls
                style={{ width: '100%', marginBottom: '20px' }}
                autoPlay
              >
                Your browser does not support the audio element.
              </audio>
              
              {/* Target Class Selector - moved below audio player */}
              {availableClasses.length > 1 && (
                <div className="form-group" style={{ marginTop: '10px' }}>
                  <label htmlFor="classSelect" style={{ fontSize: '0.9rem', marginBottom: '5px', display: 'block' }}>Target Class</label>
                  <Select
                    id="classSelect"
                    options={availableClasses}
                    value={selectedClass}
                    onChange={(selected) => {
                      setSelectedClass(selected);
                      selectClass(selected.value);
                    }}
                    isSearchable={false}
                    isDisabled={isLoading || !isDatasetLoaded}
                    styles={{
                      control: (base) => ({
                        ...base,
                        minHeight: '32px',
                        fontSize: '0.875rem',
                        border: '2px solid #e89c81',
                        '&:hover': { border: '2px solid #e89c81' },
                        '&:focus-within': { 
                          border: '2px solid #6e7cb9',
                          boxShadow: '0 0 0 3px rgba(110, 124, 185, 0.1)'
                        }
                      }),
                      indicatorsContainer: (base) => ({
                        ...base,
                        height: '32px'
                      }),
                      valueContainer: (base) => ({
                        ...base,
                        padding: '2px 8px'
                      })
                    }}
                  />
                  <small style={{ color: '#666', fontSize: '0.75rem', display: 'block', marginTop: '3px' }}>
                    Select which class to annotate
                  </small>
                </div>
              )}
            </div>

            <div>
              <h4>Annotation</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <button
                  onClick={() => annotateClip(0)}
                  className="btn btn-danger"
                >
                  Not Present (0)
                </button>
                <button
                  onClick={() => annotateClip(1)}
                  className="btn btn-success"
                >
                  Present (1)
                </button>
                <button
                  onClick={() => annotateClip(3)}
                  className="btn btn-warning"
                >
                  Uncertain (3)
                </button>
                
                {availableClasses.length > 1 && (
                  <div style={{ marginTop: '15px', paddingTop: '15px', borderTop: '1px solid #ddd' }}>
                    <button
                      onClick={markOtherClassesAsAbsent}
                      className="btn btn-info"
                      style={{ fontSize: '0.9rem' }}
                    >
                      Mark Other Classes as "Not Present"
                    </button>
                    <small style={{ 
                      display: 'block', 
                      color: '#666', 
                      fontSize: '0.8rem', 
                      marginTop: '5px',
                      lineHeight: '1.3'
                    }}>
                      Use when only the current class is present in this clip
                    </small>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {!currentClip && isDatasetLoaded && !isLoading && (
        <div className="card">
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <h3>No clips available</h3>
            <p>Try adjusting the score range or check if there are unreviewed clips in the dataset.</p>
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <h3>Instructions</h3>
        </div>
        <div style={{ lineHeight: '1.6' }}>
          <ol>
            <li><strong>Load Dataset:</strong> Enter the path to a dataset created with the Dataset Builder</li>
            <li><strong>Select Target Class (Multiclass):</strong> Choose which class to annotate from the dropdown</li>
            <li><strong>Load Classifier (Optional):</strong> Load a pretrained classifier to update scores</li>
            <li><strong>Adjust Score Range:</strong> Filter clips by their classification scores</li>
            <li><strong>Review Clips:</strong> Listen to audio and examine spectrograms</li>
            <li><strong>Annotate:</strong> Mark clips as "Present" or "Not Present" for strong labels, or "Uncertain" for weak labels</li>
            <li><strong>Save Progress:</strong> Regularly save your annotations to the database</li>
            <li><strong>Export Results:</strong> Export annotated clips as WAV files when done</li>
          </ol>
          <p><strong>Strong vs Weak Labels:</strong> When you mark a clip as "Present" or "Not Present" for a class, it becomes a strong label for that class only. Other classes remain weakly labeled (not confirmed). This allows selective annotation without assuming presence/absence of all classes simultaneously.</p>
          <p><strong>Keyboard shortcuts:</strong> Use the annotation buttons or navigate with Previous/Next</p>
        </div>
      </div>
    </div>
  );
};

export default ActiveLearning;