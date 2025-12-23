import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';

const ValidationInterface = () => {
  // Project loading states
  const [availableProjects, setAvailableProjects] = useState([]);
  const [showLoadModal, setShowLoadModal] = useState(false);
  const [projectSearchLocation, setProjectSearchLocation] = useState('');

  // Strata and species selection
  const [availableStrata, setAvailableStrata] = useState([]);
  const [selectedStrata, setSelectedStrata] = useState('');
  const [availableSpecies, setAvailableSpecies] = useState([]);
  const [selectedSpecies, setSelectedSpecies] = useState('');

  // Validation settings
  const [validationRules, setValidationRules] = useState({
    target_confirmations: 1,
    confidence_threshold: 0.5,
    auto_advance: true
  });
  
  // Session mode: 'validate' (default) or 'review'
  const [sessionMode, setSessionMode] = useState('validate');

  // Current validation state
  const [currentClip, setCurrentClip] = useState(null);
  const [validationQueue, setValidationQueue] = useState([]);
  const [queueIndex, setQueueIndex] = useState(0);
  const [sessionProgress, setSessionProgress] = useState(null);

  // Audio playback
  const audioRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioCurrentTime, setAudioCurrentTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const lastTimeUpdateRef = useRef(0);

  // Loading states
  const [isLoading, setIsLoading] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // Spectrogram (separate state to avoid re-renders when updating currentClip)
  const [spectrogramUrl, setSpectrogramUrl] = useState(null);

  // Spectrogram color mode
  const [colorMode, setColorMode] = useState('viridis');
  
  // Hotkeys
  const [hotkeysEnabled, setHotkeysEnabled] = useState(false);

  const colorModeOptions = [
    { value: 'viridis', label: 'Viridis (Color)' },
    { value: 'gray_r', label: 'Grayscale' },
    { value: 'plasma', label: 'Plasma' },
    { value: 'inferno', label: 'Inferno' },
  ];

  // Helper function to get annotation status badge
  const getAnnotationStatusBadge = (clip) => {
    // Check for either annotation_status or validation_state
    const rawStatus = clip?.annotation_status || clip?.validation_state;
    
    if (!clip || !rawStatus) {
      return null;
    }

    const status = rawStatus.toLowerCase();
    const colors = {
      confirmed: { bg: '#dcfce7', color: '#166534', text: 'Previously Confirmed', icon: '✓' },
      rejected: { bg: '#fee2e2', color: '#991b1b', text: 'Previously Rejected', icon: '✗' },
      uncertain: { bg: '#fef3c7', color: '#92400e', text: 'Previously Marked Uncertain', icon: '?' },
      skipped: { bg: '#f3f4f6', color: '#374151', text: 'Previously Skipped', icon: '⊘' }
    };

    const style = colors[status] || { bg: '#e0e0e0', color: '#666', text: `Previously: ${status}`, icon: '◉' };

    return (
      <div style={{
        backgroundColor: style.bg,
        color: style.color,
        padding: '0.75rem 1rem',
        borderRadius: '8px',
        marginBottom: '1rem',
        border: `2px solid ${style.color}`,
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        fontSize: '0.95rem',
        fontWeight: '600'
      }}>
        <span style={{ fontSize: '1.2rem' }}>{style.icon}</span>
        <span>{style.text}</span>
        {clip.annotation_timestamp && (
          <span style={{ fontSize: '0.8rem', fontWeight: 'normal', marginLeft: 'auto' }}>
            {new Date(clip.annotation_timestamp).toLocaleString()}
          </span>
        )}
      </div>
    );
  };

  // Load available strata on component mount
  useEffect(() => {
    loadAvailableStrata();
  }, []);

  // Load species when strata is selected
  useEffect(() => {
    if (selectedStrata) {
      loadAvailableSpecies(selectedStrata);
    }
  }, [selectedStrata]);

  // Reload spectrogram when color mode changes
  useEffect(() => {
    if (currentClip && currentClip.audio_file_path && colorMode) {
      loadClip(currentClip);
    }
  }, [colorMode]);

  const loadAvailableStrata = async () => {
    console.log('🔄 Refresh Strata button clicked - loading strata...');

    try {
      const response = await axios.get('/api/validation/strata');
      console.log('Strata API response:', response.data);
      console.log('Strata count:', response.data.strata ? response.data.strata.length : 0);

      const strata = response.data.strata || [];
      setAvailableStrata(strata);

      // Clear selected strata if it's no longer in the list
      if (selectedStrata && strata.length > 0) {
        const strataExists = strata.some(s => s.strata_id === selectedStrata);
        if (!strataExists) {
          console.log('Selected strata no longer exists, clearing selections');
          setSelectedStrata('');
          setAvailableSpecies([]);
          setSelectedSpecies('');
        }
      } else if (strata.length === 0) {
        // Clear all selections if no strata available
        console.log('No strata available, clearing all selections');
        setSelectedStrata('');
        setAvailableSpecies([]);
        setSelectedSpecies('');
        toast.info('No strata found in current validation database');
      } else {
        console.log(`Loaded ${strata.length} strata successfully`);
        toast.success(`Loaded ${strata.length} strata`);
      }
    } catch (error) {
      console.error('Error loading strata:', error);
      const message = error.response?.data?.detail || error.message || 'Failed to load strata';
      toast.error(`Failed to refresh strata: ${message}`);

      // Clear strata on error (likely means no validation DB loaded)
      setAvailableStrata([]);
      setSelectedStrata('');
      setAvailableSpecies([]);
      setSelectedSpecies('');
    }
  };

  const loadAvailableSpecies = async (strataId) => {
    try {
      const response = await axios.get(`/api/validation/strata/${strataId}/species`);
      setAvailableSpecies(response.data.species || []);
    } catch (error) {
      toast.error('Failed to load species for selected strata');
    }
  };

  const listProjects = async () => {
    if (!projectSearchLocation.trim()) {
      toast.error('Please specify a location to search for projects');
      return;
    }

    try {
      const response = await axios.get(`/api/validation/list-projects?base_path=${encodeURIComponent(projectSearchLocation)}`);

      if (response.data.status === 'success') {
        setAvailableProjects(response.data.projects);
        setShowLoadModal(true);
      } else {
        toast.error(response.data.message || 'Failed to list projects');
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to list projects';
      toast.error(message);
    }
  };

  const loadProject = async (projectPath) => {
    setIsLoading(true);

    try {
      const response = await axios.post('/api/validation/load-project', {
        project_path: projectPath
      });

      if (response.data.status === 'success') {
        toast.success(`Project loaded successfully: ${response.data.project_name}`);
        setShowLoadModal(false);

        // Refresh available strata after loading project
        loadAvailableStrata();
      } else {
        toast.error(response.data.message || 'Failed to load project');
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to load project';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const startValidationSession = async (overrideStrataId = null, overrideSpeciesName = null) => {
    const strataId = overrideStrataId || selectedStrata;
    const speciesName = overrideSpeciesName || selectedSpecies;

    console.log('startValidationSession called');
    console.log('Strata ID:', strataId);
    console.log('Species:', speciesName);

    if (!strataId || !speciesName) {
      toast.error('Please select both strata and species');
      return;
    }

    setIsLoading(true);

    try {
      // Create a clean copy of validation rules to avoid circular reference issues
      const cleanRules = {
        target_confirmations: Number(validationRules.target_confirmations),
        confidence_threshold: Number(validationRules.confidence_threshold),
        auto_advance: Boolean(validationRules.auto_advance)
      };

      console.log('Validation rules:', cleanRules);

      const response = await axios.post('/api/validation/start-session', {
        strata_id: strataId,
        species_name: speciesName,
        validation_rules: cleanRules,
        review_mode: sessionMode === 'review'
      });

      console.log('Session started successfully');

      if (response.data.status === 'success') {
        setValidationQueue(response.data.validation_queue);
        setQueueIndex(0);
        setSessionProgress(response.data.session_progress);

        if (response.data.validation_queue.length > 0) {
          // Small delay to ensure audio element is ready
          setTimeout(() => {
            loadClip(response.data.validation_queue[0]);
          }, 100);
        } else {
          toast.info('No clips available for validation in this strata/species combination');
        }
      } else {
        const errorMsg = response.data.message || 'Unknown error occurred';
        console.error('Session failed:', errorMsg);
        toast.error(`Failed to start validation session: ${errorMsg}`);
      }
    } catch (error) {
      console.error('Session error:', error.message);
      const message = error.response?.data?.detail || error.response?.data?.message || error.message || 'Network error or server unavailable';
      toast.error(`Failed to start validation session: ${message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const advanceToNextStrata = async () => {
    if (!selectedSpecies || availableStrata.length === 0) {
      toast.error('No strata available to advance to');
      return;
    }

    // Find current strata index
    const currentIndex = availableStrata.findIndex(s => s.strata_id === selectedStrata);

    // Move to next strata (wrap around to beginning if at end)
    const nextIndex = (currentIndex + 1) % availableStrata.length;
    const nextStrata = availableStrata[nextIndex];

    // Update selected strata state for UI display
    setSelectedStrata(nextStrata.strata_id);

    // Start session with the new strata ID directly (don't wait for state update)
    await startValidationSession(nextStrata.strata_id, selectedSpecies);
  };

  const saveProject = async () => {
    setIsSaving(true);

    try {
      const response = await axios.post('/api/validation/save-project', {});

      if (response.data.status === 'success') {
        toast.success(`Project saved to ${response.data.project_path}`);
      } else {
        toast.error(response.data.message || 'Failed to save project');
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to save project';
      toast.error(message);
    } finally {
      setIsSaving(false);
    }
  };

  const loadClip = async (clipData) => {
    if (!clipData) return;

    console.log('Loading clip:', clipData);
    console.log('audioRef.current:', audioRef.current);
    console.log('audio_file_path:', clipData.audio_file_path);
    console.log('All clip data keys:', Object.keys(clipData));

    // Set current clip (single state update - no spectrogram embedded)
    setCurrentClip(clipData);

    // Clear previous spectrogram immediately
    setSpectrogramUrl(null);

    // Stop and clear any currently playing audio
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }

    // Load audio if available
    if (clipData.audio_file_path && clipData.audio_file_path.trim()) {
      console.log('Audio file path exists, checking audioRef...');

      if (!audioRef.current) {
        console.error('audioRef.current is null!');
        toast.error('Audio player not ready');
        return;
      }

      try {
        // Use backend slicing with query parameters to serve only the specific clip
        // This ensures the audio player timeline matches the clip duration (e.g. 5s) instead of the full file (e.g. 1h)
        // The backend endpoint is now non-blocking (def instead of async def), so this slicing won't freeze the server
        const audioUrl = `/api/validation/audio/${clipData.audio_file_path}?clip_start=${clipData.start_time || 0}&clip_end=${clipData.end_time || 0}`;
        console.log('Setting audio URL:', audioUrl);

        audioRef.current.src = audioUrl;
        console.log('Audio src set, calling load()');
        audioRef.current.load(); // Force reload

        // Try to play (may fail due to browser autoplay policies)
        try {
          console.log('Attempting autoplay...');
          await audioRef.current.play();
          console.log('Autoplay succeeded');
        } catch (playError) {
          console.log('Autoplay prevented:', playError.message);
          // This is expected - user must interact first
        }
      } catch (error) {
        console.error('Error loading audio:', error);
        toast.error('Failed to load audio: ' + error.message);
      }
    } else {
      console.log('No audio_file_path in clip data');
    }

    // Load spectrogram
    if (clipData.audio_file_path) {
      try {
        const spectrogramResponse = await axios.post('/api/spectrogram', {
          file_path: clipData.audio_file_path,
          clip_start: clipData.start_time || 0,
          clip_end: clipData.end_time || 0,
          color_mode: colorMode
        });

        if (spectrogramResponse.data.spectrogram) {
          setSpectrogramUrl(spectrogramResponse.data.spectrogram);
        }
      } catch (error) {
        console.error('Error loading spectrogram:', error);
        toast.error('Failed to load spectrogram');
      }
    }
  };

  const submitValidation = async (validationState, confidence = 3, notes = '') => {
    if (!currentClip) return;

    setIsValidating(true);

    try {
      const response = await axios.post('/api/validation/submit-annotation', {
        prediction_id: currentClip.prediction_id,
        validation_state: validationState,
        validation_confidence: confidence,
        notes: notes,
        strata_id: selectedStrata,
        species_name: selectedSpecies
      });

      if (response.data.status === 'success') {
        // Update session progress
        setSessionProgress(response.data.session_progress);

        // Update local queue with new annotation
        const updatedQueue = [...validationQueue];
        const clipIndex = updatedQueue.findIndex(c => c.prediction_id === currentClip.prediction_id);
        
        if (clipIndex !== -1) {
          updatedQueue[clipIndex] = {
            ...updatedQueue[clipIndex],
            annotation_status: validationState,
            annotation_timestamp: new Date().toISOString()
          };
          setValidationQueue(updatedQueue);
          
          // Also update current clip state immediately so badge appears
          setCurrentClip(prev => ({
            ...prev,
            annotation_status: validationState,
            annotation_timestamp: new Date().toISOString()
          }));
        }

        // Check if target met for this strata/species
        if (response.data.target_met) {
          toast.success(`Target confirmations reached for ${selectedSpecies} in this strata!`);
          // Could auto-advance to next species or strata
        }

        // Move to next clip
        if (validationRules.auto_advance) {
          // Use the updated queue to find next clip
          if (queueIndex < updatedQueue.length - 1) {
            const nextIndex = queueIndex + 1;
            setQueueIndex(nextIndex);
            loadClip(updatedQueue[nextIndex]);
          } else {
            toast.info('No more clips in queue for this session');
          }
        } else {
          toast.success('Validation recorded successfully');
        }
      } else {
        toast.error(response.data.message || 'Failed to record validation');
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to record validation';
      toast.error(message);
    } finally {
      setIsValidating(false);
    }
  };

  const advanceToNextClip = () => {
    if (queueIndex < validationQueue.length - 1) {
      const nextIndex = queueIndex + 1;
      setQueueIndex(nextIndex);
      loadClip(validationQueue[nextIndex]);
    } else {
      toast.info('No more clips in queue for this session');
    }
  };

  const goToPreviousClip = () => {
    if (queueIndex > 0) {
      const prevIndex = queueIndex - 1;
      setQueueIndex(prevIndex);
      loadClip(validationQueue[prevIndex]);
    }
  };

  // Audio controls
  const togglePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleAudioTimeUpdate = () => {
    if (audioRef.current) {
      // Throttle updates to every 100ms to reduce re-renders and prevent stuttering
      const now = Date.now();
      if (now - lastTimeUpdateRef.current >= 100) {
        setAudioCurrentTime(audioRef.current.currentTime);
        lastTimeUpdateRef.current = now;
      }
    }
  };

  const handleAudioLoadedMetadata = () => {
    if (audioRef.current) {
      setAudioDuration(audioRef.current.duration);
    }
  };

  const seekToTime = (time) => {
    if (audioRef.current) {
      audioRef.current.currentTime = time;
      setAudioCurrentTime(time);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Hotkey handler
  useEffect(() => {
    if (!hotkeysEnabled || !currentClip) return;

    const handleKeyDown = (e) => {
      // Ignore if user is typing in an input field
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
        return;
      }

      switch (e.key.toLowerCase()) {
        case ' ':
          e.preventDefault(); // Prevent scrolling
          togglePlayPause();
          break;
        case 'c':
        case 'y':
          submitValidation('confirmed', 5);
          break;
        case 'r':
        case 'n':
          submitValidation('rejected', 5);
          break;
        case 'u':
            submitValidation('uncertain', 3);
            break;
        case 's':
            submitValidation('skipped', 1);
            break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [hotkeysEnabled, currentClip, togglePlayPause, submitValidation]);

  return (
    <div>
      {/* Load Project Section */}
      <div className="card">
        <div className="card-header">
          <h3>Load Validation Project</h3>
          <p>Load an existing validation project to continue work</p>
        </div>

        <div className="grid grid-2">
          <div className="form-group">
            <label htmlFor="projectSearchLocation">Project Location</label>
            <input
              type="text"
              id="projectSearchLocation"
              className="form-control"
              placeholder="/path/to/validation/projects"
              value={projectSearchLocation}
              onChange={(e) => setProjectSearchLocation(e.target.value)}
              disabled={isLoading}
            />
            <small style={{ color: '#666', fontSize: '0.875rem' }}>
              Directory containing validation projects
            </small>
          </div>

          <div className="form-group" style={{ display: 'flex', alignItems: 'end' }}>
            <button
              onClick={listProjects}
              disabled={isLoading || !projectSearchLocation.trim()}
              className="btn btn-secondary"
              style={{ width: '100%' }}
            >
              Browse Projects
            </button>
          </div>
        </div>
      </div>

      {/* Session Configuration */}
      <div className="card">
        <div className="card-header">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <h3>Validation Session Setup</h3>
              <p>Select strata and species for systematic validation</p>
            </div>
            <button
              onClick={loadAvailableStrata}
              className="btn btn-outline btn-sm"
              disabled={isLoading}
              title="Refresh strata list from current validation database"
            >
              🔄 Refresh Strata
            </button>
          </div>
          {availableStrata.length === 0 && (
            <div style={{
              padding: '0.75rem',
              backgroundColor: '#fff3cd',
              border: '1px solid #ffeaa7',
              borderRadius: '4px',
              color: '#856404',
              marginTop: '1rem'
            }}>
              <strong>No validation strata available.</strong> Please go to the Dataset Builder tab to create a validation dataset, or load an existing project above.
            </div>
          )}
        </div>

        <div className="grid grid-2">
          <div className="form-group">
            <label htmlFor="strataSelect">Validation Strata</label>
            <select
              id="strataSelect"
              className="form-control"
              value={selectedStrata}
              onChange={(e) => setSelectedStrata(e.target.value)}
              disabled={isLoading}
            >
              <option value="">
                {availableStrata.length === 0 ? "No strata available - load dataset first" : "Select a strata..."}
              </option>
              {availableStrata
                .sort((a, b) => a.strata_name.localeCompare(b.strata_name))
                .map(strata => (
                <option key={strata.strata_id} value={strata.strata_id}>
                  {strata.strata_name} ({strata.total_clips} clips, {strata.species_count} species)
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="speciesSelect">Target Species</label>
            <select
              id="speciesSelect"
              className="form-control"
              value={selectedSpecies}
              onChange={(e) => setSelectedSpecies(e.target.value)}
              disabled={isLoading || !selectedStrata}
            >
              <option value="">
                {!selectedStrata ? "Select strata first" : availableSpecies.length === 0 ? "No species in selected strata" : "Select a species..."}
              </option>
              {availableSpecies
                .sort((a, b) => a.species_name.localeCompare(b.species_name))
                .map(species => (
                <option key={species.species_name} value={species.species_name}>
                  {species.species_name} ({species.total_clips} clips, {species.confirmed_clips} confirmed)
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="grid grid-3">
          <div className="form-group">
            <label htmlFor="targetConfirmations">Target Confirmations</label>
            <select
              id="targetConfirmations"
              className="form-control"
              value={validationRules.target_confirmations}
              onChange={(e) => setValidationRules({
                ...validationRules,
                target_confirmations: parseInt(e.target.value)
              })}
              disabled={isLoading}
            >
              <option value={1}>1 confirmation</option>
              <option value={2}>2 confirmations</option>
              <option value={3}>3 confirmations</option>
              <option value={5}>5 confirmations</option>
              <option value={10}>10 confirmations</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="confidenceThreshold">Min Confidence</label>
            <input
              type="number"
              id="confidenceThreshold"
              className="form-control"
              min="0"
              max="1"
              step="0.01"
              value={validationRules.confidence_threshold}
              onChange={(e) => setValidationRules({
                ...validationRules,
                confidence_threshold: parseFloat(e.target.value)
              })}
              disabled={isLoading}
            />
            <small style={{ color: '#666', fontSize: '0.875rem' }}>
              Only include predictions above this confidence threshold
            </small>
          </div>

          <div className="form-group">
            <label htmlFor="colorMode">Spectrogram Color</label>
            <select
              id="colorMode"
              className="form-control"
              value={colorMode}
              onChange={(e) => setColorMode(e.target.value)}
            >
              {colorModeOptions.map(option => (
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
                id="autoAdvance"
                checked={validationRules.auto_advance}
                onChange={(e) => setValidationRules({
                  ...validationRules,
                  auto_advance: e.target.checked
                })}
                disabled={isLoading}
                style={{
                  width: '18px',
                  height: '18px',
                  accentColor: '#6e7cb9'
                }}
              />
              <label htmlFor="autoAdvance" style={{ margin: 0, cursor: 'pointer' }}>
                Auto-advance
              </label>
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '0.5rem' }}>
              <input
                type="checkbox"
                id="enableHotkeys"
                checked={hotkeysEnabled}
                onChange={(e) => setHotkeysEnabled(e.target.checked)}
                style={{
                  width: '18px',
                  height: '18px',
                  accentColor: '#0d6efd'
                }}
              />
              <label htmlFor="enableHotkeys" style={{ margin: 0, cursor: 'pointer' }}>
                Enable Hotkeys (Space=Play, C/Y=Confirm, R/N=Reject)
              </label>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '0.5rem' }}>
              <input
                type="checkbox"
                id="reviewMode"
                checked={sessionMode === 'review'}
                onChange={(e) => setSessionMode(e.target.checked ? 'review' : 'validate')}
                disabled={isLoading}
                style={{
                  width: '18px',
                  height: '18px',
                  accentColor: '#dc3545'
                }}
              />
              <label htmlFor="reviewMode" style={{ margin: 0, cursor: 'pointer', color: sessionMode === 'review' ? '#dc3545' : 'inherit', fontWeight: sessionMode === 'review' ? 'bold' : 'normal' }}>
                Review Mode (Edit existing validations)
              </label>
            </div>
          </div>
        </div>

        <div style={{ textAlign: 'center', marginTop: '1.5rem', display: 'flex', gap: '1rem', justifyContent: 'center' }}>
          <button
            onClick={() => startValidationSession()}
            disabled={isLoading || !selectedStrata || !selectedSpecies}
            className={`btn btn-lg ${sessionMode === 'review' ? 'btn-danger' : 'btn-primary'}`}
          >
            {isLoading ? 'Starting Session...' : sessionMode === 'review' ? 'Start Review Session' : 'Start Validation Session'}
          </button>
          <button
            onClick={() => advanceToNextStrata()}
            disabled={isLoading || !selectedSpecies || availableStrata.length === 0}
            className="btn btn-secondary btn-lg"
          >
            Next Strata
          </button>
        </div>
      </div>

      {/* Session Progress */}
      {sessionProgress && (
        <div className="card">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3>Session Progress</h3>
            <button
              onClick={saveProject}
              disabled={isSaving}
              className="btn btn-success"
            >
              {isSaving ? 'Saving...' : 'Save Project'}
            </button>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '20px' }}>
            {/* Progress bar section */}
            <div style={{ flex: 1 }}>
              <div style={{ marginBottom: '0.5rem' }}>
                <span style={{ fontWeight: '600', fontSize: '0.9rem' }}>
                  {sessionProgress.reviewed_clips}/{sessionProgress.total_clips} clips reviewed
                  <span style={{ color: '#666', marginLeft: '10px' }}>
                    ({Math.round((sessionProgress.reviewed_clips / sessionProgress.total_clips) * 100)}%)
                  </span>
                </span>
              </div>
              <div style={{
                backgroundColor: '#e9ecef',
                borderRadius: '4px',
                overflow: 'hidden',
                height: '8px'
              }}>
                <div style={{
                  backgroundColor: '#6e7cb9',
                  height: '100%',
                  width: `${(sessionProgress.reviewed_clips / sessionProgress.total_clips) * 100}%`,
                  transition: 'width 0.3s ease'
                }}></div>
              </div>
            </div>

            {/* Counts section */}
            <div style={{ display: 'flex', gap: '15px', fontSize: '0.875rem' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontWeight: 'bold', color: '#059669' }}>{sessionProgress.confirmed_clips}</div>
                <div style={{ fontSize: '0.75rem', color: '#666' }}>Confirmed</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontWeight: 'bold', color: '#dc3545' }}>{sessionProgress.rejected_clips}</div>
                <div style={{ fontSize: '0.75rem', color: '#666' }}>Rejected</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontWeight: 'bold', color: '#ffc107' }}>{sessionProgress.uncertain_clips}</div>
                <div style={{ fontSize: '0.75rem', color: '#666' }}>Uncertain</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontWeight: 'bold', color: '#6c757d' }}>{sessionProgress.skipped_clips}</div>
                <div style={{ fontSize: '0.75rem', color: '#666' }}>Skipped</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Current Clip Validation */}
      {currentClip && (
        <div className="card">
          <div className="card-header">
            <h3>Current Clip</h3>

            {/* Annotation Status Indicator */}
            {getAnnotationStatusBadge(currentClip)}

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ marginBottom: '5px' }}>
                  <strong>File:</strong> {currentClip.filename}
                </div>
                <div style={{ fontSize: '0.875rem', color: '#666' }}>
                  <strong>Time:</strong> {currentClip.start_time?.toFixed(1)}s - {currentClip.end_time?.toFixed(1)}s |
                  <strong> Species:</strong> {currentClip.species_name} |
                  <strong> Model Score:</strong> {currentClip.confidence?.toFixed(6)} |
                  <strong> Model:</strong> {currentClip.model_name} |
                  <strong> Clip:</strong> {queueIndex + 1} of {validationQueue.length}
                </div>
              </div>
              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  onClick={goToPreviousClip}
                  disabled={queueIndex === 0}
                  className="btn btn-secondary btn-sm"
                >
                  ← Previous
                </button>
                <button
                  onClick={advanceToNextClip}
                  disabled={queueIndex >= validationQueue.length - 1}
                  className="btn btn-secondary btn-sm"
                >
                  Next →
                </button>
              </div>
            </div>
          </div>

          <div>
            <h4>Spectrogram</h4>
            {spectrogramUrl ? (
              <img
                src={spectrogramUrl}
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
                onTimeUpdate={handleAudioTimeUpdate}
                onLoadedMetadata={handleAudioLoadedMetadata}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onError={(e) => {
                  console.error('Audio error:', e);
                  console.error('Audio error code:', audioRef.current?.error?.code);
                  console.error('Audio error message:', audioRef.current?.error?.message);
                  console.error('Audio src:', audioRef.current?.src);
                  toast.error('Audio playback error - check browser console');
                }}
                onLoadStart={() => console.log('Audio load started')}
                onLoadedData={() => console.log('Audio data loaded')}
                onCanPlay={() => console.log('Audio can play')}
              >
                Your browser does not support the audio element.
              </audio>
            </div>

            <div>
              <h4>Annotation</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <button
                  onClick={() => submitValidation('confirmed', 5)}
                  disabled={isValidating}
                  className="btn btn-success"
                >
                  ✓ Confirm {hotkeysEnabled && '(C)'}
                </button>
                <button
                  onClick={() => submitValidation('rejected', 5)}
                  disabled={isValidating}
                  className="btn btn-danger"
                >
                  ✗ Reject {hotkeysEnabled && '(R)'}
                </button>
                <button
                  onClick={() => submitValidation('uncertain', 3)}
                  disabled={isValidating}
                  className="btn btn-warning"
                >
                  ? Uncertain {hotkeysEnabled && '(U)'}
                </button>
                <button
                  onClick={() => submitValidation('skipped', 1)}
                  disabled={isValidating}
                  className="btn btn-secondary"
                >
                  ⏭️ Skip {hotkeysEnabled && '(S)'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="card">
        <div className="card-header">
          <h3>Validation Instructions</h3>
        </div>
        <div style={{ lineHeight: '1.6' }}>
          <ol>
            <li><strong>Select Strata:</strong> Choose a validation strata (time/site grouping)</li>
            <li><strong>Choose Species:</strong> Select the target species for this validation session</li>
            <li><strong>Set Rules:</strong> Configure target confirmations and confidence thresholds</li>
            <li><strong>Review Clips:</strong> Listen to audio clips and validate model predictions</li>
            <li><strong>Make Decisions:</strong> Confirm, reject, mark uncertain, or skip each prediction</li>
          </ol>
          <div style={{
            marginTop: '1rem',
            padding: '1rem',
            backgroundColor: '#e8f5e8',
            border: '1px solid #c8e6c9',
            borderRadius: '4px'
          }}>
            <h4 style={{ margin: '0 0 0.5rem 0', color: '#2e7d2e' }}>Validation States:</h4>
            <p style={{ margin: '0', color: '#2e7d2e' }}>
              <strong>Confirm:</strong> Model prediction is correct<br/>
              <strong>Reject:</strong> Model prediction is incorrect<br/>
              <strong>Uncertain:</strong> Cannot determine if prediction is correct<br/>
              <strong>Skip:</strong> Move to next clip without decision
            </p>
          </div>
        </div>
      </div>

      {/* Load Project Modal */}
      {showLoadModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'white',
            borderRadius: '8px',
            padding: '2rem',
            maxWidth: '600px',
            width: '90%',
            maxHeight: '80vh',
            overflowY: 'auto'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3>Load Validation Project</h3>
              <button
                onClick={() => setShowLoadModal(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '1.5rem',
                  cursor: 'pointer',
                  padding: '0',
                  color: '#666'
                }}
              >
                ×
              </button>
            </div>

            {availableProjects.length === 0 ? (
              <p style={{ textAlign: 'center', color: '#666' }}>
                No validation projects found in the specified location.
              </p>
            ) : (
              <div>
                <p style={{ marginBottom: '1rem' }}>
                  Select a validation project to load:
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {availableProjects.map((project, index) => (
                    <div
                      key={index}
                      style={{
                        border: '1px solid #ddd',
                        borderRadius: '4px',
                        padding: '1rem',
                        cursor: 'pointer',
                        transition: 'background-color 0.2s',
                        backgroundColor: 'white'
                      }}
                      onClick={() => loadProject(project.project_path)}
                      onMouseEnter={(e) => e.target.style.backgroundColor = '#f5f5f5'}
                      onMouseLeave={(e) => e.target.style.backgroundColor = 'white'}
                    >
                      <div style={{ fontWeight: 'bold', marginBottom: '0.25rem' }}>
                        {project.project_name}
                      </div>
                      <div style={{ fontSize: '0.875rem', color: '#666' }}>
                        Created: {new Date(project.created_at).toLocaleString()}<br/>
                        Predictions: {project.total_predictions} |
                        Annotations: {project.total_annotations} |
                        Strata: {project.total_strata}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div style={{ marginTop: '1rem', textAlign: 'center' }}>
              <button
                onClick={() => setShowLoadModal(false)}
                className="btn btn-secondary"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ValidationInterface;