import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import SpectrogramViewer, { formatTime, formatFreq } from './SpectrogramViewer';
import SpectrogramOptions from './SpectrogramOptions';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  TextField,
  Button,
  Grid,
  Typography,
  Divider,
  Stack,
  Collapse,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  FormControlLabel,
  Checkbox,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
  Slider,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  FolderOpen as LoadIcon,
  Refresh as RefreshIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  CheckCircle as ConfirmIcon,
  Cancel as RejectIcon,
  Help as UncertainIcon,
  SkipNext as SkipIcon,
  NavigateNext as NextIcon,
  NavigateBefore as PrevIcon,
  Save as SaveIcon,
  Tune as TuneIcon,
} from '@mui/icons-material';

const ValidationInterface = ({ isActive = true }) => {
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
  
  // New target mode state
  const [targetMode, setTargetMode] = useState('confirmations'); // 'confirmations' or 'total'
  
  // Session mode
  const [sessionMode, setSessionMode] = useState('validate');
  const [selectionStrategy, setSelectionStrategy] = useState('top_down');
  const [validationStatusFilter, setValidationStatusFilter] = useState('all'); // Filter by validation status

  // Current validation state
  const [currentClip, setCurrentClip] = useState(null);
  const [validationQueue, setValidationQueue] = useState([]);
  const [queueIndex, setQueueIndex] = useState(0);
  const [sessionProgress, setSessionProgress] = useState(null);
  const [overallProgress, setOverallProgress] = useState(null);

  // Audio playback
  const audioRef = useRef(null);
  const preloadAudioRef = useRef(null); // For preloading next clip
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioCurrentTime, setAudioCurrentTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const lastTimeUpdateRef = useRef(0);

  // Loading states
  const [isLoading, setIsLoading] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // Spectrogram
  const [spectrogramUrl, setSpectrogramUrl] = useState(null);
  const [spectrogramMetadata, setSpectrogramMetadata] = useState(null);
  const [colorMode, setColorMode] = useState('viridis');

  // Spectrogram options state
  const [spectrogramOptions, setSpectrogramOptions] = useState({
    color_mode: 'viridis',
    freq_scale: 'mel',
    n_mels: 256,
    n_fft: 2048,
    hop_length: 128,
    window_size: null,
    fmin: null,  // null means use model default
    fmax: null,  // null means use model default
    bandpass_min: null,
    bandpass_max: null
  });

  // Hotkeys
  const [hotkeysEnabled, setHotkeysEnabled] = useState(false);

  // Scroll management
  const validationAreaRef = useRef(null);

  const colorModeOptions = [
    { value: 'viridis', label: 'Viridis' },
    { value: 'gray_r', label: 'Grayscale' },
    { value: 'plasma', label: 'Plasma' },
    { value: 'inferno', label: 'Inferno' },
  ];

  // Effects
  useEffect(() => { loadAvailableStrata(); }, []);
  useEffect(() => { if (selectedStrata) loadAvailableSpecies(selectedStrata); }, [selectedStrata]);
  
  // Load media resources whenever currentClip changes
  useEffect(() => {
    if (currentClip) {
        loadAudio(currentClip);
        loadSpectrogram(currentClip);
        // Preload next clip after a short delay
        setTimeout(() => preloadNextClip(), 100);

        // Prevent jumping to top by scrolling validation area into view
        if (validationAreaRef.current) {
            // Use setTimeout to ensure the DOM has updated
            setTimeout(() => {
                validationAreaRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }, 50);
        }
    }
  }, [currentClip]);

  // Reload spectrogram when spectrogram options change
  useEffect(() => {
    if (currentClip) {
        loadSpectrogram(currentClip);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spectrogramOptions]);

  // --- API Handlers ---
  const loadAvailableStrata = async () => {
    try {
      const response = await axios.get('/api/validation/strata');
      const strata = response.data.strata || [];
      setAvailableStrata(strata);
      
      if (selectedStrata && strata.length > 0 && !strata.some(s => s.strata_id === selectedStrata)) {
          setSelectedStrata('');
          setAvailableSpecies([]);
          setSelectedSpecies('');
      } else if (strata.length === 0) {
          toast.info('No strata found in current validation database');
      }
    } catch (error) {
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
    if (!projectSearchLocation.trim()) { toast.error('Please specify a location'); return; }
    try {
      const response = await axios.get(`/api/validation/list-projects?base_path=${encodeURIComponent(projectSearchLocation)}`);
      if (response.data.status === 'success') {
        setAvailableProjects(response.data.projects);
        setShowLoadModal(true);
      }
    } catch (error) { toast.error('Failed to list projects'); }
  };

  const loadProject = async (projectPath) => {
    setIsLoading(true);
    try {
      const response = await axios.post('/api/validation/load-project', { project_path: projectPath });
      if (response.data.status === 'success') {
        toast.success(`Project loaded: ${response.data.project_name}`);
        setShowLoadModal(false);
        loadAvailableStrata();
      } else {
        // API returned an error status
        const errorMsg = response.data.message || 'Unknown error';
        console.error('Project load failed:', errorMsg);
        toast.error(`Failed to load project: ${errorMsg}`);
      }
    } catch (error) {
      // Network or other error
      const errorMsg = error.response?.data?.message || error.message || 'Network error';
      console.error('Project load error:', error);
      toast.error(`Failed to load project: ${errorMsg}`);
    }
    finally { setIsLoading(false); }
  };

  const startValidationSession = async (overrideStrataId = null, overrideSpeciesName = null) => {
    const strataId = overrideStrataId || selectedStrata;
    const speciesName = overrideSpeciesName || selectedSpecies;

    if (!strataId || !speciesName) { toast.error('Please select both strata and species'); return; }

    setIsLoading(true);
    try {
      // Map validation status filter to array of statuses
      let statusFilterArray = null;
      if (validationStatusFilter !== 'all') {
        if (validationStatusFilter === 'unvalidated') {
          statusFilterArray = ['unvalidated'];
        } else {
          statusFilterArray = [validationStatusFilter];
        }
      }

      const response = await axios.post('/api/validation/start-session', {
        strata_id: strataId,
        species_name: speciesName,
        validation_rules: {
            target_confirmations: Number(validationRules.target_confirmations),
            confidence_threshold: Number(validationRules.confidence_threshold),
            auto_advance: Boolean(validationRules.auto_advance)
        },
        review_mode: sessionMode === 'review',
        selection_strategy: selectionStrategy,
        validation_status_filter: statusFilterArray
      });

      if (response.data.status === 'success') {
        setValidationQueue(response.data.validation_queue);
        setQueueIndex(0);
        setSessionProgress(response.data.session_progress);
        setOverallProgress(response.data.overall_progress);

        if (response.data.validation_queue.length > 0) {
          // Just set the clip - the useEffect will handle loading media
          setCurrentClip(response.data.validation_queue[0]);
        } else {
          toast.info('No clips available for validation');
          setCurrentClip(null);
        }
      }
    } catch (error) { toast.error('Failed to start session'); } 
    finally { setIsLoading(false); }
  };

  const advanceToNextStrata = async () => {
    if (!selectedSpecies || availableStrata.length === 0) return;
    const currentIndex = availableStrata.findIndex(s => s.strata_id === selectedStrata);
    const nextIndex = (currentIndex + 1) % availableStrata.length;
    const nextStrata = availableStrata[nextIndex];
    setSelectedStrata(nextStrata.strata_id);
    await startValidationSession(nextStrata.strata_id, selectedSpecies);
  };

  const saveProject = async () => {
    setIsSaving(true);
    try {
      const response = await axios.post('/api/validation/save-project', {});
      if (response.data.status === 'success') toast.success('Project saved');
    } catch (error) { toast.error('Failed to save project'); }
    finally { setIsSaving(false); }
  };

  const saveSpectrogram = () => {
    if (!spectrogramUrl) return;
    const link = document.createElement('a');
    link.href = spectrogramUrl;
    link.download = `spectrogram_${currentClip?.clip_id || 'clip'}.png`;
    link.click();
  };

  const saveAudio = () => {
    if (!currentClip) return;
    const audioUrl = `/api/audio/${currentClip.audio_file_path}?clip_start=${currentClip.start_time || 0}&clip_end=${currentClip.end_time || 0}`;
    const link = document.createElement('a');
    link.href = audioUrl;
    link.download = `audio_${currentClip.clip_id || 'clip'}.wav`;
    link.click();
  };

  const toggleStrataCompletion = async (isCompleted) => {
    if (!selectedStrata || !selectedSpecies) {
      toast.error('No active validation session');
      return;
    }

    // Optimistic update - update UI immediately for instant feedback
    const previousState = sessionProgress?.is_completed;
    setSessionProgress(prev => ({
      ...prev,
      is_completed: isCompleted
    }));

    try {
      // Fire and forget the API call - don't wait for auto-save
      axios.post('/api/validation/toggle-strata-completion', {
        strata_id: selectedStrata,
        species_name: selectedSpecies,
        is_completed: isCompleted
      }).then(response => {
        if (response.data.status === 'success') {
          toast.success(isCompleted ? 'Strata marked as complete' : 'Strata marked as incomplete', { autoClose: 2000 });
        } else {
          // Rollback on error
          setSessionProgress(prev => ({
            ...prev,
            is_completed: previousState
          }));
          toast.error('Failed to update completion status');
        }
      }).catch(error => {
        // Rollback on error
        setSessionProgress(prev => ({
          ...prev,
          is_completed: previousState
        }));
        toast.error('Failed to update completion status');
        console.error('Toggle completion error:', error);
      });
    } catch (error) {
      // Rollback on immediate error
      setSessionProgress(prev => ({
        ...prev,
        is_completed: previousState
      }));
      toast.error('Failed to update completion status');
      console.error('Toggle completion error:', error);
    }
  };

  const loadAudio = (clipData) => {
    if (!clipData || !clipData.audio_file_path) return;

    // Slight delay to ensure DOM is ready if this is the first clip
    // This handles the "audioRef is null" issue on initial load
    if (!audioRef.current) {
        setTimeout(() => loadAudio(clipData), 50);
        return;
    }

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setAudioCurrentTime(0);

      const audioUrl = `/api/audio/${clipData.audio_file_path}?clip_start=${clipData.start_time || 0}&clip_end=${clipData.end_time || 0}`;
      audioRef.current.src = audioUrl;
      audioRef.current.load();

      // Auto-play without blocking - don't await the promise
      audioRef.current.play().then(() => {
        setIsPlaying(true);
      }).catch((e) => {
        // Silently handle auto-play failures (browser restrictions, etc.)
        setIsPlaying(false);
      });
    }
  };

  const loadSpectrogram = async (clipData) => {
    if (!clipData || !clipData.audio_file_path) return;
    setSpectrogramUrl(null); // Clear previous

    try {
        const specRes = await axios.post('/api/spectrogram', {
            file_path: clipData.audio_file_path,
            clip_start: clipData.start_time || 0,
            clip_end: clipData.end_time || 0,
            ...spectrogramOptions  // Include all spectrogram options
        });
        if (specRes.data.spectrogram) {
            setSpectrogramUrl(specRes.data.spectrogram);
            setSpectrogramMetadata(specRes.data.metadata || null);
        }
    } catch (e) { console.error("Spectrogram error", e); }
  };

  // Preload the next clip's audio in the background
  const preloadNextClip = () => {
    if (!validationQueue || queueIndex >= validationQueue.length - 1) return;

    const nextClip = validationQueue[queueIndex + 1];
    if (!nextClip || !nextClip.audio_file_path) return;

    if (preloadAudioRef.current) {
      const audioUrl = `/api/audio/${nextClip.audio_file_path}?clip_start=${nextClip.start_time || 0}&clip_end=${nextClip.end_time || 0}`;
      preloadAudioRef.current.src = audioUrl;
      preloadAudioRef.current.load();
    }
  };

  // Legacy loadClip wrapper for backward compatibility with button calls
  const loadClip = (clipData) => {
      setCurrentClip(clipData);
  };

  const submitValidation = async (validationState, confidence = 3, notes = '') => {
    if (!currentClip) return;

    // Stop audio immediately when validation button is clicked
    if (audioRef.current) {
      audioRef.current.pause();
      setIsPlaying(false);
    }

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
        setSessionProgress(response.data.session_progress);
        setOverallProgress(response.data.overall_progress);

        const updatedQueue = [...validationQueue];
        const clipIndex = updatedQueue.findIndex(c => c.prediction_id === currentClip.prediction_id);
        if (clipIndex !== -1) {
            updatedQueue[clipIndex] = { ...updatedQueue[clipIndex], annotation_status: validationState, annotation_timestamp: new Date().toISOString() };
            setValidationQueue(updatedQueue);
            setCurrentClip(prev => ({ ...prev, annotation_status: validationState }));
        }

        // Check target condition based on mode
        let targetMet = false;
        const targetValue = validationRules.target_confirmations;
        
        if (targetMode === 'confirmations') {
            targetMet = response.data.target_met;
        } else if (targetMode === 'total') {
            const validatedCount = response.data.session_progress?.validated_clips || 0;
            targetMet = validatedCount >= targetValue;
        }

        if (targetMet) {
            toast.success(`Target ${targetMode === 'total' ? 'validations' : 'confirmations'} reached for ${selectedSpecies}!`);
        }

        if (validationRules.auto_advance) {
            if (queueIndex < updatedQueue.length - 1) {
                const nextIndex = queueIndex + 1;
                setQueueIndex(nextIndex);
                loadClip(updatedQueue[nextIndex]);
            } else {
                toast.info('Session complete');
            }
        }
      }
    } catch (error) { toast.error('Failed to submit validation'); }
    finally { setIsValidating(false); }
  };

  const deleteAnnotation = async () => {
    if (!currentClip) return;
    if (!window.confirm('Are you sure you want to delete this annotation?')) return;

    setIsValidating(true);
    try {
        const response = await axios.delete('/api/active-learning/annotation', {
            params: {
                clip_id: currentClip.prediction_id,
                class_name: selectedSpecies
            }
        });

        if (response.data.status === 'success') {
            toast.success('Annotation deleted');
            
            // Update session progress locally
            if (sessionProgress && currentClip.annotation_status) {
                const status = currentClip.annotation_status.toLowerCase();
                const newProgress = { ...sessionProgress };
                
                if (status === 'confirmed') newProgress.confirmed_clips = Math.max(0, newProgress.confirmed_clips - 1);
                else if (status === 'rejected') newProgress.rejected_clips = Math.max(0, newProgress.rejected_clips - 1);
                else if (status === 'uncertain') newProgress.uncertain_clips = Math.max(0, newProgress.uncertain_clips - 1);
                else if (status === 'skipped') newProgress.skipped_clips = Math.max(0, newProgress.skipped_clips - 1);
                
                newProgress.validated_clips = Math.max(0, newProgress.validated_clips - 1);
                setSessionProgress(newProgress);
            }

            const updatedQueue = [...validationQueue];
            const clipIndex = updatedQueue.findIndex(c => c.prediction_id === currentClip.prediction_id);
            if (clipIndex !== -1) {
                const updatedClip = { ...updatedQueue[clipIndex] };
                delete updatedClip.annotation_status;
                delete updatedClip.annotation_timestamp;
                updatedQueue[clipIndex] = updatedClip;
                setValidationQueue(updatedQueue);
                setCurrentClip(updatedClip);
            }
            if (validationRules.auto_advance) {
                advanceToNextClip();
            }
        }
    } catch (error) {
        toast.error('Failed to delete annotation');
    } finally {
        setIsValidating(false);
    }
  };

  const advanceToNextClip = () => {
    if (queueIndex < validationQueue.length - 1) {
      const nextIndex = queueIndex + 1;
      setQueueIndex(nextIndex);
      loadClip(validationQueue[nextIndex]);
    }
  };

  const goToPreviousClip = () => {
    if (queueIndex > 0) {
      const prevIndex = queueIndex - 1;
      setQueueIndex(prevIndex);
      loadClip(validationQueue[prevIndex]);
    }
  };

  // Audio Handlers
  const togglePlayPause = () => {
    if (audioRef.current) {
        if (isPlaying) audioRef.current.pause();
        else audioRef.current.play();
        setIsPlaying(!isPlaying);
    }
  };
  const handleAudioTimeUpdate = () => {
      if (audioRef.current) {
          const now = Date.now();
          if (now - lastTimeUpdateRef.current >= 100) {
              setAudioCurrentTime(audioRef.current.currentTime);
              lastTimeUpdateRef.current = now;
          }
      }
  };

  // Hotkeys
  useEffect(() => {
    // Only register hotkeys when this component is active, hotkeys are enabled, and there's a clip
    if (!isActive || !hotkeysEnabled || !currentClip) return;
    const handleKeyDown = (e) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
      switch (e.key.toLowerCase()) {
        case ' ': e.preventDefault(); togglePlayPause(); break;
        case 'c': case 'y': submitValidation('confirmed', 5); break;
        case 'r': case 'n': submitValidation('rejected', 5); break;
        case 'u': submitValidation('uncertain', 3); break;
        case 's': submitValidation('skipped', 1); break;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isActive, hotkeysEnabled, currentClip, isPlaying]);

  return (
    <Box sx={{ display: isActive ? 'block' : 'none', pb: 4 }}>
      
      {/* Session Controls */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} md={4}>
            <Card elevation={0} sx={{ border: '1px solid #e0e0e0', height: '100%' }}>
                <CardHeader 
                    title="Load Project" 
                    titleTypographyProps={{ variant: 'subtitle1', fontWeight: 'bold' }}
                />
                <CardContent sx={{ pt: 0 }}>
                    <Stack spacing={1.5}>
                        <Typography variant="body2" color="text.secondary">
                            Specify the directory containing your validation projects.
                        </Typography>
                        <TextField
                            fullWidth
                            size="small"
                            label="Search Location"
                            placeholder="/path/to/validation/projects"
                            value={projectSearchLocation}
                            onChange={(e) => setProjectSearchLocation(e.target.value)}
                            disabled={isLoading}
                        />
                        <Button 
                            variant="outlined" 
                            startIcon={<LoadIcon />} 
                            fullWidth
                            onClick={listProjects}
                            disabled={isLoading || !projectSearchLocation.trim()}
                        >
                            Browse Projects
                        </Button>
                    </Stack>
                </CardContent>
            </Card>
        </Grid>
        <Grid item xs={12} md={8}>
            <Card elevation={0} sx={{ border: '1px solid #e0e0e0', height: '100%' }}>
                <CardContent>
                    <Grid container spacing={3} alignItems="center">
                        <Grid item xs={12} md={4}>
                            <FormControl fullWidth>
                                <InputLabel>Strata</InputLabel>
                                <Select value={selectedStrata} label="Strata" onChange={(e) => setSelectedStrata(e.target.value)}>
                                    {[...availableStrata].sort((a, b) => a.strata_name.localeCompare(b.strata_name)).map(s => <MenuItem key={s.strata_id} value={s.strata_id}>{s.strata_name}</MenuItem>)}
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={12} md={4}>
                            <FormControl fullWidth>
                                <InputLabel>Species</InputLabel>
                                <Select value={selectedSpecies} label="Species" onChange={(e) => setSelectedSpecies(e.target.value)} disabled={!selectedStrata}>
                                    {[...availableSpecies].sort((a, b) => a.species_name.localeCompare(b.species_name)).map(s => <MenuItem key={s.species_name} value={s.species_name}>{s.species_name}</MenuItem>)}
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={12} md={4}>
                            <Stack direction="row" spacing={1}>
                                <Button 
                                    variant="contained" 
                                    fullWidth
                                    size="large"
                                    disabled={!selectedStrata || !selectedSpecies || isLoading}
                                    onClick={() => startValidationSession()}
                                    sx={{ height: 56 }}
                                >
                                    Start
                                </Button>
                                <Button 
                                    variant="outlined" 
                                    fullWidth
                                    size="large"
                                    disabled={isLoading || !selectedSpecies || availableStrata.length === 0}
                                    onClick={() => advanceToNextStrata()}
                                    sx={{ height: 56 }}
                                >
                                    Next Strata
                                </Button>
                            </Stack>
                        </Grid>
                    </Grid>

                    <Divider sx={{ my: 2 }} />
                    
                    <Typography variant="caption" color="text.secondary" fontWeight="bold" sx={{ mb: 1, display: 'block' }}>SESSION OPTIONS</Typography>
                    <Grid container spacing={2}>
                        <Grid item xs={6} md={3}>
                            <FormControl size="small" fullWidth>
                                <InputLabel>Strategy</InputLabel>
                                <Select value={selectionStrategy} label="Strategy" onChange={(e) => setSelectionStrategy(e.target.value)}>
                                    <MenuItem value="top_down">Top-Down (Confidence)</MenuItem>
                                    <MenuItem value="bottom_up">Bottom-Up (Confidence)</MenuItem>
                                    <MenuItem value="sequential">Sequential (Time Order)</MenuItem>
                                    <MenuItem value="random">Random</MenuItem>
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={6} md={3}>
                            <FormControl size="small" fullWidth>
                                <InputLabel>Status Filter</InputLabel>
                                <Select value={validationStatusFilter} label="Status Filter" onChange={(e) => setValidationStatusFilter(e.target.value)}>
                                    <MenuItem value="all">All Clips</MenuItem>
                                    <MenuItem value="unvalidated">Not Validated</MenuItem>
                                    <MenuItem value="confirmed">Confirmed</MenuItem>
                                    <MenuItem value="rejected">Rejected</MenuItem>
                                    <MenuItem value="uncertain">Uncertain</MenuItem>
                                    <MenuItem value="skipped">Skipped</MenuItem>
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={6} md={3}>
                            <FormControl size="small" fullWidth>
                                <InputLabel>Target Mode</InputLabel>
                                <Select value={targetMode} label="Target Mode" onChange={(e) => setTargetMode(e.target.value)}>
                                    <MenuItem value="confirmations">Confirmations</MenuItem>
                                    <MenuItem value="total">Total Validated</MenuItem>
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={6} md={3}>
                            <TextField
                                size="small"
                                fullWidth
                                type="number"
                                label={targetMode === 'confirmations' ? "Confirmed Target" : "Total Target"}
                                value={validationRules.target_confirmations}
                                onChange={(e) => setValidationRules({...validationRules, target_confirmations: parseInt(e.target.value)})}
                                inputProps={{ min: 1 }}
                            />
                        </Grid>
                        <Grid item xs={6} md={3}>
                            <TextField
                                size="small"
                                fullWidth
                                type="number"
                                label="Min Score"
                                value={validationRules.confidence_threshold}
                                onChange={(e) => setValidationRules({...validationRules, confidence_threshold: parseFloat(e.target.value)})}
                                inputProps={{ min: 0, max: 1, step: 0.1 }}
                            />
                        </Grid>
                    </Grid>

                    <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                        <FormControlLabel control={<Checkbox checked={validationRules.auto_advance} onChange={(e) => setValidationRules({...validationRules, auto_advance: e.target.checked})} size="small" />} label={<Typography variant="body2">Auto-advance</Typography>} />
                        <FormControlLabel control={<Checkbox checked={hotkeysEnabled} onChange={(e) => setHotkeysEnabled(e.target.checked)} size="small" />} label={<Typography variant="body2">Hotkeys</Typography>} />
                        <FormControlLabel control={<Checkbox checked={sessionMode === 'review'} onChange={(e) => setSessionMode(e.target.checked ? 'review' : 'validate')} size="small" />} label={<Typography variant="body2">Review Mode</Typography>} />
                        <FormControlLabel control={<Checkbox checked={colorMode !== 'viridis'} onChange={(e) => setColorMode(e.target.checked ? 'gray_r' : 'viridis')} size="small" />} label={<Typography variant="body2">Grayscale</Typography>} />
                    </Box>
                </CardContent>
            </Card>
        </Grid>
      </Grid>

      {/* Progress Bar */}
      {sessionProgress && (
        <Card elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2 }}>
            <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                <Grid container alignItems="center" spacing={3}>
                    <Grid item xs={12} md={4}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                            <Typography variant="caption" fontWeight="bold">
                                {selectedSpecies ? `${selectedSpecies.toUpperCase()} PROGRESS` : 'SESSION PROGRESS'}
                            </Typography>
                            <Typography variant="caption">
                                {sessionProgress.total_strata > 1 ? (
                                    // Multiple strata: show strata completion
                                    `${sessionProgress.completed_strata} / ${sessionProgress.total_strata} strata`
                                ) : (
                                    // Single/no strata: show clip completion
                                    `${sessionProgress.validated_clips} / ${sessionProgress.total_clips} clips`
                                )}
                            </Typography>
                        </Box>
                        <LinearProgress
                            variant="determinate"
                            value={sessionProgress.total_strata > 1 ?
                                (sessionProgress.completed_strata / sessionProgress.total_strata) * 100 :
                                (sessionProgress.validated_clips / sessionProgress.total_clips) * 100
                            }
                            sx={{ height: 8, borderRadius: 4 }}
                        />
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                            {sessionProgress.total_strata > 1 ? (
                                `${sessionProgress.species_validated_clips || sessionProgress.validated_clips} / ${sessionProgress.species_total_clips || sessionProgress.total_clips} total clips across all strata`
                            ) : (
                                `${sessionProgress.confirmed_clips} confirmed, ${sessionProgress.rejected_clips} rejected`
                            )}
                        </Typography>
                    </Grid>
                    <Grid item xs={12} md={8}>
                        <Grid container spacing={1}>
                            <Grid item xs={4} sm={2} textAlign="center">
                                <Typography variant="h6" color="text.secondary">{sessionProgress?.validated_clips || 0}</Typography>
                                <Typography variant="caption" color="text.secondary">Total</Typography>
                            </Grid>
                            <Grid item xs={4} sm={2} textAlign="center">
                                <Typography variant="h6" color="success.main">{sessionProgress?.confirmed_clips || 0}</Typography>
                                <Typography variant="caption" color="text.secondary">Confirm</Typography>
                            </Grid>
                            <Grid item xs={4} sm={2} textAlign="center">
                                <Typography variant="h6" color="error.main">{sessionProgress?.rejected_clips || 0}</Typography>
                                <Typography variant="caption" color="text.secondary">Reject</Typography>
                            </Grid>
                            <Grid item xs={4} sm={2} textAlign="center">
                                <Typography variant="h6" color="warning.main">{sessionProgress?.uncertain_clips || 0}</Typography>
                                <Typography variant="caption" color="text.secondary">Unsure</Typography>
                            </Grid>
                            <Grid item xs={4} sm={2} textAlign="center">
                                <Typography variant="h6" color="text.disabled">{sessionProgress?.skipped_clips || 0}</Typography>
                                <Typography variant="caption" color="text.secondary">Skip</Typography>
                            </Grid>
                            <Grid item xs={4} sm={2} textAlign="center">
                                <Button size="small" variant="outlined" color="success" onClick={saveProject} disabled={isSaving} sx={{ minWidth: 0, px: 1 }}>
                                    <SaveIcon fontSize="small" />
                                </Button>
                                <Typography variant="caption" display="block" color="text.secondary">Save</Typography>
                            </Grid>
                        </Grid>

                        {/* Completion Status */}
                        <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={sessionProgress?.is_completed || false}
                                        onChange={(e) => toggleStrataCompletion(e.target.checked)}
                                        color="success"
                                    />
                                }
                                label={
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <Typography variant="body2">Mark Strata as Complete</Typography>
                                        {sessionProgress?.is_completed && (
                                            <Chip label="COMPLETED" size="small" color="success" />
                                        )}
                                    </Box>
                                }
                            />
                            <Typography variant="caption" color="text.secondary">
                                Target: {sessionProgress?.target_confirmations || 0} confirmations
                            </Typography>
                        </Box>
                    </Grid>
                </Grid>
            </CardContent>
        </Card>
      )}

      {/* Main Validation Area */}
      {currentClip && (
        <Grid container spacing={2} ref={validationAreaRef} sx={{ minHeight: 600 }}>
            {/* Left: Spectrogram & Audio */}
            <Grid item xs={12} md={8}>
                <Card elevation={0} sx={{ border: '1px solid #e0e0e0', height: '100%' }}>
                    <CardHeader 
                        title={
                            <Stack direction="row" alignItems="center" spacing={2}>
                                <Typography variant="h6" noWrap sx={{ maxWidth: 400 }}>{currentClip.filename}</Typography>
                                {currentClip.annotation_status && (
                                    <Chip 
                                        label={currentClip.annotation_status.toUpperCase()} 
                                        color={
                                            currentClip.annotation_status === 'confirmed' ? 'success' : 
                                            currentClip.annotation_status === 'rejected' ? 'error' : 
                                            currentClip.annotation_status === 'uncertain' ? 'warning' : 'default'
                                        }
                                        size="small"
                                        variant="filled"
                                    />
                                )}
                            </Stack>
                        }
                        subheader={
                            <Stack direction="row" spacing={4} sx={{ mt: 1.5, flexWrap: 'wrap' }}>
                                <Box>
                                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontWeight: 'bold', letterSpacing: '0.05em' }}>CLIP RANGE</Typography>
                                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                        {spectrogramMetadata ? `${formatTime(spectrogramMetadata.clip_start)} — ${formatTime(spectrogramMetadata.clip_end)}` : `${currentClip.start_time?.toFixed(1)}s - ${currentClip.end_time?.toFixed(1)}s`}
                                    </Typography>
                                </Box>
                                <Box>
                                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontWeight: 'bold', letterSpacing: '0.05em' }}>DURATION</Typography>
                                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                        {spectrogramMetadata ? `${(spectrogramMetadata.clip_end - spectrogramMetadata.clip_start).toFixed(2)}s` : `${(currentClip.end_time - currentClip.start_time).toFixed(1)}s`}
                                    </Typography>
                                </Box>
                                <Box>
                                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontWeight: 'bold', letterSpacing: '0.05em' }}>FREQ RANGE</Typography>
                                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                        {spectrogramMetadata ? `${formatFreq(spectrogramMetadata.freq_min)} - ${formatFreq(spectrogramMetadata.freq_max)}Hz` : 'N/A'}
                                    </Typography>
                                </Box>
                                <Box>
                                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontWeight: 'bold', letterSpacing: '0.05em' }}>SCORE</Typography>
                                    <Typography variant="body2" sx={{ color: 'primary.main', fontWeight: 'bold' }}>{currentClip.confidence?.toFixed(3)}</Typography>
                                </Box>
                                <Box>
                                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontWeight: 'bold', letterSpacing: '0.05em' }}>SPECIES</Typography>
                                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>{currentClip.species_name}</Typography>
                                </Box>
                            </Stack>
                        }
                        action={
                            <Stack direction="row" spacing={1}>
                                <Button
                                    variant="outlined"
                                    size="small"
                                    onClick={saveSpectrogram}
                                    startIcon={<SaveIcon />}
                                    disabled={!spectrogramUrl}
                                >
                                    Save Image
                                </Button>
                                <Button
                                    variant="outlined"
                                    size="small"
                                    onClick={saveAudio}
                                    startIcon={<SaveIcon />}
                                    disabled={!currentClip}
                                >
                                    Save Audio
                                </Button>
                                <Button
                                    variant="outlined"
                                    size="small"
                                    onClick={goToPreviousClip}
                                    disabled={queueIndex === 0}
                                    startIcon={<PrevIcon />}
                                >
                                    Previous
                                </Button>
                                <Button
                                    variant="outlined"
                                    size="small"
                                    onClick={advanceToNextClip}
                                    disabled={queueIndex >= validationQueue.length - 1}
                                    endIcon={<NextIcon />}
                                >
                                    Next
                                </Button>
                            </Stack>
                        }
                    />
                    <CardContent>
                        {/* Spectrogram Options */}
                        <SpectrogramOptions
                          options={spectrogramOptions}
                          onChange={setSpectrogramOptions}
                          onReset={() => setSpectrogramOptions({
                            color_mode: 'viridis',
                            freq_scale: 'mel',
                            n_mels: 256,
                            n_fft: 2048,
                            hop_length: 128,
                            window_size: null,
                            fmin: null,
                            fmax: null,
                            bandpass_min: null,
                            bandpass_max: null
                          })}
                          modelDefaults={{
                            MIN_FREQ: 60,
                            MAX_FREQ: 10000
                          }}
                        />

                        {/* Spectrogram Viewer */}
                        <Box sx={{ mb: 2 }}>
                            <SpectrogramViewer
                                spectrogramUrl={spectrogramUrl}
                                metadata={spectrogramMetadata}
                                audioCurrentTime={audioCurrentTime}
                                clipDuration={currentClip ? currentClip.end_time - currentClip.start_time : 0}
                                isLoading={!spectrogramUrl}
                                showMetadata={false}
                            />
                        </Box>

                        {/* Audio Controls */}
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                            <IconButton onClick={togglePlayPause} color="primary" size="large" sx={{ border: '2px solid', width: 48, height: 48 }}>
                                {isPlaying ? <PauseIcon /> : <PlayIcon />}
                            </IconButton>
                            <Slider size="small" value={audioCurrentTime} max={audioDuration || 100} onChange={(_, v) => { if(audioRef.current) audioRef.current.currentTime = v; }} />
                            <Typography variant="caption" sx={{ minWidth: 40, textAlign: 'right' }}>
                                {audioCurrentTime.toFixed(1)} / {audioDuration.toFixed(1)}s
                            </Typography>
                            <audio ref={audioRef} onTimeUpdate={handleAudioTimeUpdate} onLoadedMetadata={() => setAudioDuration(audioRef.current?.duration || 0)} onEnded={() => setIsPlaying(false)} style={{ display: 'none' }} />
                            {/* Hidden audio element for preloading next clip */}
                            <audio ref={preloadAudioRef} preload="auto" style={{ display: 'none' }} />
                        </Box>
                    </CardContent>
                </Card>
            </Grid>

            {/* Right: Actions */}
            <Grid item xs={12} md={4}>
                <Card elevation={0} sx={{ border: '1px solid #e0e0e0', height: '100%' }}>
                    <CardHeader title="Validation" subheader="Make a decision for this clip" />
                    <CardContent>
                        <Stack spacing={2}>
                            <Button variant="outlined" fullWidth size="large" onClick={() => submitValidation('confirmed')} startIcon={<ConfirmIcon sx={{ fontSize: 30 }} />} sx={{ justifyContent: 'flex-start', py: 2.5, px: 3, color: 'success.main', borderColor: 'success.main', '&:hover': { bgcolor: '#f0fdf4', borderColor: 'success.dark' } }}>
                                <Box textAlign="left" sx={{ ml: 1 }}>
                                    <Typography variant="subtitle1" fontWeight="bold" display="block">Confirm</Typography>
                                    <Typography variant="caption">Correct Prediction {hotkeysEnabled && '(C)'}</Typography>
                                </Box>
                            </Button>
                            <Button variant="outlined" fullWidth size="large" onClick={() => submitValidation('rejected')} startIcon={<RejectIcon sx={{ fontSize: 30 }} />} sx={{ justifyContent: 'flex-start', py: 2.5, px: 3, color: 'error.main', borderColor: 'error.main', '&:hover': { bgcolor: '#fef2f2', borderColor: 'error.dark' } }}>
                                <Box textAlign="left" sx={{ ml: 1 }}>
                                    <Typography variant="subtitle1" fontWeight="bold" display="block">Reject</Typography>
                                    <Typography variant="caption">Incorrect Prediction {hotkeysEnabled && '(R)'}</Typography>
                                </Box>
                            </Button>
                            <Button variant="outlined" fullWidth size="large" onClick={() => submitValidation('uncertain')} startIcon={<UncertainIcon sx={{ fontSize: 30 }} />} sx={{ justifyContent: 'flex-start', py: 2.5, px: 3, color: 'warning.main', borderColor: 'warning.main', '&:hover': { bgcolor: '#fffbeb', borderColor: 'warning.dark' } }}>
                                <Box textAlign="left" sx={{ ml: 1 }}>
                                    <Typography variant="subtitle1" fontWeight="bold" display="block">Uncertain</Typography>
                                    <Typography variant="caption">Cannot Determine {hotkeysEnabled && '(U)'}</Typography>
                                </Box>
                            </Button>
                            <Button variant="outlined" fullWidth size="large" onClick={() => submitValidation('skipped')} startIcon={<SkipIcon sx={{ fontSize: 30 }} />} sx={{ justifyContent: 'flex-start', py: 2.5, px: 3, color: 'text.secondary', borderColor: 'text.secondary', '&:hover': { bgcolor: '#f3f4f6', borderColor: 'text.primary' } }}>
                                <Box textAlign="left" sx={{ ml: 1 }}>
                                    <Typography variant="subtitle1" fontWeight="bold" display="block">Skip</Typography>
                                    <Typography variant="caption">Next Clip {hotkeysEnabled && '(S)'}</Typography>
                                </Box>
                            </Button>
                            
                            {sessionMode === 'review' && (
                                <Button 
                                    variant="outlined" 
                                    fullWidth 
                                    size="large" 
                                    onClick={deleteAnnotation}
                                    startIcon={<RejectIcon sx={{ fontSize: 30 }} />} 
                                    sx={{ justifyContent: 'flex-start', py: 2.5, px: 3, color: 'error.dark', borderColor: 'error.dark', '&:hover': { bgcolor: '#fef2f2', borderColor: 'error.main' } }}
                                >
                                    <Box textAlign="left" sx={{ ml: 1 }}>
                                        <Typography variant="subtitle1" fontWeight="bold" display="block">Delete</Typography>
                                        <Typography variant="caption">Remove Annotation</Typography>
                                    </Box>
                                </Button>
                            )}
                        </Stack>
                    </CardContent>
                </Card>
            </Grid>
        </Grid>
      )}

      {/* Load Project Dialog */}
      <Dialog open={showLoadModal} onClose={() => setShowLoadModal(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Load Project</DialogTitle>
        <DialogContent dividers>
            <Stack spacing={1}>
                {availableProjects.length === 0 ? <Typography align="center" color="text.secondary">No projects found.</Typography> : 
                 availableProjects.map((p, i) => (
                    <Card key={i} variant="outlined" sx={{ cursor: 'pointer', '&:hover': { bgcolor: '#f5f5f5' } }} onClick={() => loadProject(p.project_path)}>
                        <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                            <Typography variant="subtitle2">{p.project_name}</Typography>
                            <Typography variant="caption" color="text.secondary">{new Date(p.created_at).toLocaleString()}</Typography>
                        </CardContent>
                    </Card>
                 ))}
            </Stack>
        </DialogContent>
        <DialogActions>
            <Button onClick={() => setShowLoadModal(false)}>Cancel</Button>
        </DialogActions>
      </Dialog>

    </Box>
  );
};

export default ValidationInterface;