import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select as MuiSelect,
  MenuItem,
  Grid,
  Paper,
  Typography,
  Divider,
  Stack,
  Slider,
  IconButton,
  Chip,
  Skeleton,
  Tooltip,
} from '@mui/material';
import {
  Folder as FolderIcon,
  Psychology as ModelIcon,
  NavigateNext as NextIcon,
  NavigateBefore as PrevIcon,
  Save as SaveIcon,
  Delete as DeleteIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Help as HelpIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  VolumeUp as VolumeUpIcon,
  VolumeOff as VolumeOffIcon,
  FileDownload as ExportIcon,
} from '@mui/icons-material';
import Select from 'react-select';

const ActiveLearning = ({ isActive = true }) => {
  // State management
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
  const [additionalPositiveClasses, setAdditionalPositiveClasses] = useState([]);
  const [roundProgress, setRoundProgress] = useState(null);
  const [isAnnotating, setIsAnnotating] = useState(false);

  // Scroll management
  const annotationAreaRef = useRef(null);

  // Audio Player State
  const audioRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);

  const colorModeOptions = [
    { value: 'viridis', label: 'Viridis' },
    { value: 'gray_r', label: 'Grayscale' },
    { value: 'plasma', label: 'Plasma' },
    { value: 'inferno', label: 'Inferno' },
  ];

  const reviewModeOptions = [
    { value: 'random', label: 'Random' },
    { value: 'top_down', label: 'Top-down (Highest Score)' },
    { value: 'top_10+score_quantiles', label: 'Top 10+Score Quantiles (50/round)' },
    { value: 'review_annotated', label: 'Review Annotated Clips' },
  ];

  // --- Audio Player Handlers ---
  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      setDuration(audioRef.current.duration);
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
  };

  const handleSliderChange = (event, newValue) => {
    if (audioRef.current) {
      audioRef.current.currentTime = newValue;
      setCurrentTime(newValue);
    }
  };

  const handleVolumeChange = (event, newValue) => {
    setVolume(newValue);
    if (audioRef.current) {
      audioRef.current.volume = newValue;
      setIsMuted(newValue === 0);
    }
  };

  const toggleMute = () => {
    if (audioRef.current) {
      const newMuted = !isMuted;
      setIsMuted(newMuted);
      audioRef.current.muted = newMuted;
    }
  };

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
  };

  // --- Keyboard Shortcuts ---
  useEffect(() => {
    // Only register hotkeys when this component is active
    if (!isActive) return;

    const handleKeyDown = (event) => {
      // Ignore if typing in an input
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;

      switch (event.key.toLowerCase()) {
        case ' ': // Spacebar
          event.preventDefault(); // Prevent scrolling
          togglePlay();
          break;
        case 'c': // Confirm/Present (C or Y)
        case 'y':
          if (currentClip) annotate(1);
          break;
        case 'r': // Reject/Not Present (R or N)
        case 'n':
          if (currentClip) annotate(0);
          break;
        case 'u': // Uncertain
          if (currentClip) annotate(3);
          break;
        case 'arrowright':
          nextClip();
          break;
        case 'arrowleft':
          previousClip();
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isActive, currentClip, isPlaying]); // Add dependencies as needed

  // API Functions
  const loadAvailableClasses = async () => {
    try {
      const response = await axios.get('/api/active-learning/classes');
      const classes = response.data.classes.map(cls => ({
        value: cls.value,
        label: cls.name
      }));
      setAvailableClasses(classes);

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

  const loadClipLabels = async (clipId) => {
    if (!clipId) return;
    try {
      const response = await axios.get('/api/active-learning/clip-labels', {
        params: { clip_id: clipId }
      });
      if (response.data.status === 'success') {
        setClipLabels(response.data.class_labels);
      }
    } catch (error) {
      console.error('Failed to load clip labels:', error);
      setClipLabels([]);
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
        await loadAvailableClasses();
        await loadLabelStatistics();
        // Don't automatically start annotation - user must click "Start Annotation"
        setIsAnnotating(false);
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to load dataset';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const startAnnotation = async () => {
    // Reset state before starting
    setCurrentClipIndex(0);
    setCurrentClip(null);
    setClips([]);
    setIsAnnotating(true);
    await getClips(true);
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
        await getClips();
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to load classifier';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const getClips = async (forceLoad = false) => {
    if (!isDatasetLoaded && !forceLoad) return;

    try {
      let response;
      if (reviewMode === 'review_annotated') {
        response = await axios.post('/api/active-learning/review-clips');
      } else {
        const filterConfig = {
          score_min: scoreRange[0],
          score_max: scoreRange[1],
          annotation_filter: [4]
        };
        response = await axios.post('/api/active-learning/get-clips', filterConfig);
      }

      setClips(response.data.clips);

      if (response.data.next_clip) {
        setCurrentClip(response.data.next_clip);
        setRoundProgress(response.data.round_progress || null);
        // Find the index of next_clip in the clips array, default to 0
        const clipIndex = response.data.clips.findIndex(
          clip => clip.clip_id === response.data.next_clip.clip_id
        );
        setCurrentClipIndex(clipIndex >= 0 ? clipIndex : 0);
        await loadClip(response.data.next_clip);
      } else if (response.data.clips.length > 0) {
        setCurrentClipIndex(0);
        await loadClip(response.data.clips[0]);
      } else {
        // No clips available, reset everything
        setCurrentClipIndex(0);
        setCurrentClip(null);
      }
    } catch (error) {
      console.error('Failed to get clips:', error);
    }
  };

  const loadClip = async (clip) => {
    if (!clip) {
      console.error('loadClip called with undefined clip');
      return;
    }

    // Ensure clip has clip_id property
    if (!clip.clip_id && clip.file_path && clip.clip_start !== undefined && clip.clip_end !== undefined) {
      clip.clip_id = `${clip.file_path}|${clip.clip_start}|${clip.clip_end}`;
    }

    setCurrentClip(clip);
    // Reset audio state for new clip
    setIsPlaying(false);
    setCurrentTime(0);

    // Prevent jumping to top by scrolling annotation area into view
    if (annotationAreaRef.current) {
      setTimeout(() => {
        annotationAreaRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 50);
    }

    // Extract round progress metadata if available
    if (clip.round_position && clip.round_total) {
      setRoundProgress({
        position: clip.round_position,
        total: clip.round_total,
        category: clip.round_category,
        categoryLabel: clip.round_category_label
      });
    } else {
      setRoundProgress(null);
    }

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
        // Auto-play attempt
        try {
            await audioRef.current.play();
            setIsPlaying(true);
        } catch (e) {
            console.warn("Autoplay failed", e);
            setIsPlaying(false);
        }
      }

      // Load clip labels for multiclass view
      await loadClipLabels(clip.clip_id);
    } catch (error) {
      console.error('Failed to load clip:', error);
      toast.error('Failed to load clip');
    } finally {
      setIsLoading(false);
    }
  };

  const annotate = async (value) => {
    if (!currentClip) {
      toast.error('No clip selected for annotation');
      return;
    }

    if (!currentClip.clip_id) {
      toast.error('Invalid clip data - missing clip_id');
      return;
    }

    try {
      const request = {
        clip_id: currentClip.clip_id,
        annotation: value
      };

      await axios.post('/api/active-learning/annotate', request);

      const annotationText = value === 0 ? 'Not Present' :
                           value === 1 ? 'Present' : 'Uncertain';
      toast.success(`Clip annotated as: ${annotationText}`);

      // Update label statistics and clip labels after annotation
      await loadLabelStatistics();
      await loadClipLabels(currentClip.clip_id);

      // Move to next clip
      await nextClip();
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to annotate clip';
      toast.error(message);
    }
  };

  const deleteAnnotation = async () => {
    if (!currentClip) return;

    try {
      const response = await axios.delete('/api/active-learning/annotation', {
        params: { clip_id: currentClip.clip_id }
      });

      if (response.data.status === 'success') {
        toast.success(response.data.message);
        await loadLabelStatistics();
        await loadClipLabels(currentClip.clip_id);
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to delete annotation';
      toast.error(message);
    }
  };

  const nextClip = async () => {
    if (!clips || clips.length === 0) {
      toast.error('No clips available');
      return;
    }

    // For quantile mode and review_annotated mode, cycle through the list
    if ((reviewMode === 'top_10+score_quantiles' || reviewMode === 'review_annotated') && clips.length > 0) {
      const newIndex = (currentClipIndex + 1) % clips.length;
      setCurrentClipIndex(newIndex);
      if (clips[newIndex]) {
        await loadClip(clips[newIndex]);
      }
    } else if (currentClipIndex < clips.length - 1) {
      const newIndex = currentClipIndex + 1;
      setCurrentClipIndex(newIndex);
      if (clips[newIndex]) {
        await loadClip(clips[newIndex]);
      }
    } else {
      // For other review modes, refresh to get the next clip
      await getClips();
    }
  };

  const previousClip = async () => {
    if (!clips || clips.length === 0) {
      toast.error('No clips available');
      return;
    }

    // For quantile mode and review_annotated mode, allow cycling backwards through the list
    if ((reviewMode === 'top_10+score_quantiles' || reviewMode === 'review_annotated') && clips.length > 0) {
      const newIndex = currentClipIndex === 0 ? clips.length - 1 : currentClipIndex - 1;
      setCurrentClipIndex(newIndex);
      if (clips[newIndex]) {
        await loadClip(clips[newIndex]);
      }
    } else if (currentClipIndex > 0) {
      const prevIndex = currentClipIndex - 1;
      setCurrentClipIndex(prevIndex);
      if (clips[prevIndex]) {
        await loadClip(clips[prevIndex]);
      }
    } else {
      // At the first clip - notify user
      toast.info('Already at the first clip');
    }
  };

  const generateNewRound = async () => {
    if (reviewMode !== 'top_10+score_quantiles') return;
    try {
      setIsLoading(true);
      await getClips();
      toast.success('Generated new round of 50 clips');
    } catch (error) {
      toast.error('Failed to generate new round');
    } finally {
      setIsLoading(false);
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
    if (!exportPath) return;

    try {
      const checkResponse = await axios.get('/api/active-learning/check-export-folder', {
        params: { export_path: exportPath }
      });

      if (checkResponse.data.existing_clips_count > 0) {
        if (!window.confirm(`Export folder contains ${checkResponse.data.existing_clips_count} clips. Continue?`)) {
          return;
        }
      }

      const response = await axios.post('/api/active-learning/export-clips', null, {
        params: { export_path: exportPath }
      });

      if (response.data.status === 'success') {
        toast.success(response.data.message);
      }
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to export clips');
    }
  };

  const changeReviewMode = async (newReviewMode) => {
    try {
      const response = await axios.post('/api/active-learning/set-review-mode', {
        review_mode: newReviewMode
      });

      if (response.data.status === 'success') {
        setReviewMode(newReviewMode);
        setIsAnnotating(false); // Reset annotation state when mode changes
        setCurrentClip(null); // Clear current clip
        setCurrentClipIndex(0); // Reset index
        setClips([]); // Clear clips array
        toast.success(response.data.message + ' - Click "Start Annotation" to begin.');
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to set review mode';
      toast.error(message);
    }
  };

  const selectClass = async (classIndex) => {
    try {
      setIsLoading(true);

      // Update the selected class object
      const newSelectedClass = availableClasses.find(cls => cls.value === classIndex);
      if (newSelectedClass) {
        setSelectedClass(newSelectedClass);
      }

      const response = await axios.post('/api/active-learning/select-class', null, {
        params: { class_index: classIndex }
      });

      if (response.data.status === 'success') {
        setCurrentClassIndex(classIndex);
        setIsAnnotating(false); // Reset annotation state when class changes
        setCurrentClip(null); // Clear current clip
        setCurrentClipIndex(0); // Reset index
        setClips([]); // Clear clips array
        await loadLabelStatistics();
        toast.success(response.data.message + ' - Click "Start Annotation" to begin.');
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to select class';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const annotateAdditionalClasses = async () => {
    if (!currentClip || additionalPositiveClasses.length === 0) return;

    try {
      setIsLoading(true);
      for (const classOption of additionalPositiveClasses) {
        const request = {
          clip_id: currentClip.clip_id,
          annotation: 1,
          class_index: classOption.value
        };
        await axios.post('/api/active-learning/annotate-class', request);
      }

      const classNames = additionalPositiveClasses.map(c => c.label).join(', ');
      toast.success(`Marked additional classes as present: ${classNames}`);
      setAdditionalPositiveClasses([]);
      await loadLabelStatistics();
      await loadClipLabels(currentClip.clip_id);
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to annotate additional classes';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const markOtherClassesAsAbsent = async () => {
    if (!currentClip || availableClasses.length <= 1) return;

    try {
      const request = { clip_id: currentClip.clip_id, annotation: 0 };
      const response = await axios.post('/api/active-learning/annotate-other-classes', request);

      if (response.data.status === 'success') {
        toast.success(response.data.message);
        await loadClipLabels(currentClip.clip_id);
        await loadLabelStatistics();
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to mark other classes as absent';
      toast.error(message);
    }
  };

  const saveSpectrogram = () => {
    if (!spectrogram) return;
    const link = document.createElement('a');
    link.href = spectrogram;
    link.download = `spectrogram_${currentClip?.clip_id || 'clip'}.png`;
    link.click();
  };

  // Effects
  useEffect(() => {
    if (currentClip && colorMode) {
      loadClip(currentClip);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colorMode]);

  // Removed automatic loading on scoreRange change - user must click "Start Annotation"
  // useEffect(() => {
  //   if (isDatasetLoaded && isAnnotating) {
  //     getClips();
  //   }
  //   // eslint-disable-next-line react-hooks/exhaustive-deps
  // }, [scoreRange]);

  // Get current label status
  const getCurrentLabelStatus = () => {
    if (!clipLabels) return { text: 'Unlabeled', color: '#666' };
    const targetLabel = clipLabels.find(cl => cl.is_current);
    if (!targetLabel) return { text: 'Unlabeled', color: '#666' };

    switch (targetLabel.label_text) {
      case 'Present':
        return { text: 'Present', color: '#10b981' };
      case 'Not Present':
        return { text: 'Not Present', color: '#ef4444' };
      case 'Uncertain':
        return { text: 'Uncertain', color: '#f59e0b' };
      default:
        return { text: 'Unlabeled', color: '#666' };
    }
  };

  const labelStatus = getCurrentLabelStatus();

  return (
    <Box sx={{ display: isActive ? 'block' : 'none' }}>
      {/* Dataset Configuration */}
      <Card elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
            Dataset Configuration
          </Typography>

          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Stack direction="row" spacing={1}>
                <TextField
                  fullWidth
                  size="small"
                  label="Dataset Path"
                  placeholder="/path/to/dataset"
                  value={datasetPath}
                  onChange={(e) => setDatasetPath(e.target.value)}
                  InputProps={{
                    startAdornment: <FolderIcon sx={{ mr: 1, color: 'text.secondary', fontSize: 20 }} />,
                  }}
                />
                <Button
                  variant="contained"
                  onClick={loadDataset}
                  disabled={isLoading}
                  sx={{ minWidth: 100 }}
                >
                  LOAD
                </Button>
              </Stack>
            </Grid>

            <Grid item xs={12} md={6}>
              <Stack direction="row" spacing={1}>
                <TextField
                  fullWidth
                  size="small"
                  label="Classifier Path (Optional)"
                  placeholder="/path/to/classifier.keras"
                  value={classifierPath}
                  onChange={(e) => setClassifierPath(e.target.value)}
                  disabled={!isDatasetLoaded}
                  InputProps={{
                    startAdornment: <ModelIcon sx={{ mr: 1, color: 'text.secondary', fontSize: 20 }} />,
                  }}
                />
                <Button
                  variant="contained"
                  onClick={loadClassifier}
                  disabled={isLoading || !isDatasetLoaded}
                  sx={{ minWidth: 100 }}
                >
                  LOAD
                </Button>
              </Stack>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Review Settings */}
      {isDatasetLoaded && (
        <Card elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2 }}>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
              Review Settings
            </Typography>

            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={6} md={3}>
                <FormControl fullWidth size="small">
                  <InputLabel>Review Mode</InputLabel>
                  <MuiSelect
                    value={reviewMode}
                    onChange={(e) => changeReviewMode(e.target.value)}
                    label="Review Mode"
                  >
                    {reviewModeOptions.map(opt => (
                      <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
                    ))}
                  </MuiSelect>
                </FormControl>
              </Grid>

              {availableClasses.length > 1 && (
                <Grid item xs={12} sm={6} md={3}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Target Class</InputLabel>
                    <MuiSelect
                      value={currentClassIndex}
                      onChange={(e) => selectClass(e.target.value)}
                      label="Target Class"
                    >
                      {availableClasses.map(cls => (
                        <MenuItem key={cls.value} value={cls.value}>{cls.label}</MenuItem>
                      ))}
                    </MuiSelect>
                  </FormControl>
                </Grid>
              )}

              <Grid item xs={12} sm={6} md={3}>
                <FormControl fullWidth size="small">
                  <InputLabel>Colormap</InputLabel>
                  <MuiSelect
                    value={colorMode}
                    onChange={(e) => setColorMode(e.target.value)}
                    label="Colormap"
                  >
                    {colorModeOptions.map(opt => (
                      <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
                    ))}
                  </MuiSelect>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={3}>
                <Typography variant="caption" color="text.secondary">
                  Score Range: {scoreRange[0].toFixed(2)} - {scoreRange[1].toFixed(2)}
                </Typography>
                <Slider
                  value={scoreRange}
                  onChange={(e, newValue) => setScoreRange(newValue)}
                  valueLabelDisplay="auto"
                  min={0}
                  max={1}
                  step={0.01}
                  size="small"
                />
              </Grid>
            </Grid>

            {/* Start Annotation Button */}
            {!isAnnotating && (
              <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
                <Button
                  variant="contained"
                  size="large"
                  onClick={startAnnotation}
                  disabled={isLoading}
                  sx={{
                    minWidth: 200,
                    height: 48,
                    fontSize: '1.1rem',
                    fontWeight: 600,
                  }}
                >
                  START ANNOTATION
                </Button>
              </Box>
            )}
          </CardContent>
        </Card>
      )}

      {/* Statistics Dashboard */}
      {isDatasetLoaded && labelStatistics && (
        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Paper elevation={0} sx={{ p: 2, border: '1px solid #e0e0e0', textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">Total Clips</Typography>
              <Typography variant="h5" sx={{ fontWeight: 600, color: '#1976d2' }}>
                {labelStatistics.total_clips?.toLocaleString() || 0}
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper elevation={0} sx={{ p: 2, border: '1px solid #e0e0e0', textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">Strong Labels</Typography>
              <Typography variant="h5" sx={{ fontWeight: 600, color: '#10b981' }}>
                {labelStatistics.clips_with_strong_labels?.toLocaleString() || 0}
              </Typography>
            </Paper>
          </Grid>
          
          {availableClasses.length > 1 && selectedClass && (
            <>
              <Grid item xs={12} md={6}>
                 <Paper elevation={0} sx={{ p: 2, border: '1px solid #e0e0e0' }}>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1, textAlign: 'center' }}>
                        Class Stats: {selectedClass.label}
                    </Typography>
                    <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 1 }}>
                        <Box sx={{ flex: 1, textAlign: 'center' }}>
                            <Typography variant="h6" sx={{ fontWeight: 600, color: '#1976d2' }}>
                                {labelStatistics.per_class_statistics?.[`class_${currentClassIndex}`]?.strong_labels || 0}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">Strong</Typography>
                        </Box>
                        <Divider orientation="vertical" flexItem />
                        <Box sx={{ flex: 1, textAlign: 'center' }}>
                             <Typography variant="h6" sx={{ fontWeight: 600, color: '#666' }}>
                                {labelStatistics.per_class_statistics?.[`class_${currentClassIndex}`]?.weak_labels || 0}
                            </Typography>
                             <Typography variant="caption" color="text.secondary">Weak</Typography>
                        </Box>
                    </Stack>
                 </Paper>
              </Grid>
            </>
          )}
        </Grid>
      )}

      {/* Action Buttons */}
      {isDatasetLoaded && isAnnotating && (
        <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
          <Button
            variant="contained"
            color="success"
            onClick={saveDatabase}
            startIcon={<SaveIcon />}
          >
            SAVE DATABASE
          </Button>
          <Button
            variant="contained"
            color="warning"
            onClick={exportClips}
            startIcon={<ExportIcon />}
          >
            EXPORT CLIPS
          </Button>
        </Stack>
      )}

      {/* Clip Information Bar (Horizontal) - SKELETON SUPPORT */}
      {isDatasetLoaded && isAnnotating && (
        <Card elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2 }}>
          <CardContent sx={{ py: 1.5 }}>
            {currentClip ? (
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} md={8}>
                <Stack direction="row" spacing={3} flexWrap="wrap">
                  <Box>
                    <Typography variant="caption" color="text.secondary">File</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {currentClip.file_name}
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Time</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {currentClip.clip_start?.toFixed(2)}s - {currentClip.clip_end?.toFixed(2)}s
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Score</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {currentClip.score?.toFixed(3)}
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Progress</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {currentClipIndex + 1} of {clips.length}
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Status</Typography>
                    <Box>
                      <Chip
                        label={labelStatus.text}
                        size="small"
                        sx={{
                          backgroundColor: `${labelStatus.color}20`,
                          color: labelStatus.color,
                          fontWeight: 600,
                          height: 24,
                        }}
                      />
                    </Box>
                  </Box>

                  {/* Round Progress Indicators for Quantile Mode */}
                  {roundProgress && reviewMode === 'top_10+score_quantiles' && (
                    <>
                      <Box>
                        <Typography variant="caption" color="text.secondary">Category</Typography>
                        <Box>
                          <Chip
                            label={roundProgress.categoryLabel}
                            size="small"
                            sx={{
                              backgroundColor: roundProgress.category === 'top_10' ? '#d1fae5' : '#dbeafe',
                              color: roundProgress.category === 'top_10' ? '#059669' : '#1e40af',
                              border: roundProgress.category === 'top_10' ? '2px solid #10b981' : '2px solid #3b82f6',
                              fontWeight: 600,
                              height: 24,
                            }}
                          />
                        </Box>
                      </Box>
                      <Box>
                        <Typography variant="caption" color="text.secondary">Round</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {roundProgress.position} / {roundProgress.total}
                        </Typography>
                      </Box>
                    </>
                  )}
                </Stack>
              </Grid>
              <Grid item xs={12} md={4}>
                <Stack direction="row" spacing={1} justifyContent="flex-end" flexWrap="wrap">
                  {/* Generate New Round Button for Quantile Mode */}
                  {reviewMode === 'top_10+score_quantiles' && (
                    <Button
                      variant="contained"
                      size="small"
                      onClick={generateNewRound}
                      disabled={isLoading}
                      sx={{ minWidth: 140 }}
                    >
                      Generate New Round
                    </Button>
                  )}
                  <Tooltip title="Previous Clip (Left Arrow)">
                    <span>
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={previousClip}
                        disabled={clips.length <= 1 || (currentClipIndex === 0 && reviewMode !== 'top_10+score_quantiles' && reviewMode !== 'review_annotated')}
                        startIcon={<PrevIcon />}
                      >
                        Previous
                      </Button>
                    </span>
                  </Tooltip>
                  <Tooltip title="Next Clip (Right Arrow)">
                    <span>
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={nextClip}
                        disabled={clips.length <= 1 || (currentClipIndex >= clips.length - 1 && reviewMode !== 'top_10+score_quantiles' && reviewMode !== 'review_annotated')}
                        endIcon={<NextIcon />}
                      >
                        Next
                      </Button>
                    </span>
                  </Tooltip>
                </Stack>
              </Grid>
            </Grid>
            ) : (
                <Skeleton animation="wave" height={40} />
            )}

            {/* Class Labels */}
            {currentClip && clipLabels && clipLabels.length > 0 && (
              <Box sx={{ mt: 1.5, pt: 1.5, borderTop: '1px solid #e0e0e0' }}>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                  Class Labels
                </Typography>
                <Stack direction="row" spacing={0.5} flexWrap="wrap">
                  {clipLabels.filter(cl => cl.is_current).map((label, idx) => (
                    <Chip
                      key={idx}
                      label={`${selectedClass?.label}: ${label.label_text}`}
                      size="small"
                      sx={{
                        backgroundColor: label.label_text === 'Present' ? '#10b98120' :
                                       label.label_text === 'Not Present' ? '#ef444420' : '#f59e0b20',
                        color: label.label_text === 'Present' ? '#10b981' :
                               label.label_text === 'Not Present' ? '#ef4444' : '#f59e0b',
                        border: '1px solid currentColor',
                      }}
                    />
                  ))}
                  {clipLabels.filter(cl => !cl.is_current && cl.label_text === 'Present').map((label, idx) => (
                    <Chip
                      key={idx}
                      label={`${label.class_name}: Present`}
                      size="small"
                      sx={{
                        backgroundColor: '#10b98120',
                        color: '#10b981',
                        border: '1px solid currentColor',
                      }}
                    />
                  ))}
                </Stack>
              </Box>
            )}
          </CardContent>
        </Card>
      )}

      {/* Main Content: Left Spectrogram/Audio + Right Controls */}
      {isDatasetLoaded && isAnnotating && (
        <Grid container spacing={2} ref={annotationAreaRef} sx={{ minHeight: 600 }}>
          {/* Left Column - Spectrogram & Audio Combined */}
          <Grid item xs={12} md={8}>
            <Card elevation={0} sx={{ border: '1px solid #e0e0e0' }}>
              <CardContent sx={{ p: 2 }}>
                {/* Spectrogram Header */}
                <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1.5 }}>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Spectrogram
                  </Typography>
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={saveSpectrogram}
                    startIcon={<SaveIcon />}
                    disabled={!spectrogram}
                  >
                    SAVE IMAGE
                  </Button>
                </Stack>

                {/* Spectrogram Image with Skeleton */}
                {!isLoading && spectrogram ? (
                  <Box
                    component="img"
                    src={spectrogram}
                    alt="Spectrogram"
                    sx={{
                      width: '100%',
                      height: 'auto',
                      border: '1px solid #e0e0e0',
                      borderRadius: 1,
                      backgroundColor: '#fafafa',
                      mb: 2,
                    }}
                  />
                ) : (
                  <Skeleton variant="rectangular" height={400} sx={{ borderRadius: 1, mb: 2 }} />
                )}

                {/* Custom Audio Player */}
                <Divider sx={{ mb: 1.5 }} />
                <Typography variant="h6" sx={{ mb: 1.5, fontWeight: 600 }}>
                  Audio Player
                </Typography>
                
                {/* Hidden Audio Element */}
                <audio
                  ref={audioRef}
                  onTimeUpdate={handleTimeUpdate}
                  onEnded={handleAudioEnded}
                  style={{ display: 'none' }}
                />
                
                {/* Custom Controls */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                   <IconButton onClick={togglePlay} color="primary" sx={{ border: '1px solid #1976d2' }}>
                      {isPlaying ? <PauseIcon /> : <PlayIcon />}
                   </IconButton>
                   
                   <Typography variant="caption" sx={{ minWidth: 40, textAlign: 'right' }}>
                      {formatTime(currentTime)}
                   </Typography>
                   
                   <Slider
                      size="small"
                      value={currentTime}
                      min={0}
                      max={duration || 100}
                      onChange={handleSliderChange}
                      sx={{ flexGrow: 1 }}
                   />
                   
                   <Typography variant="caption" sx={{ minWidth: 40 }}>
                      {formatTime(duration)}
                   </Typography>
                   
                   <Stack direction="row" spacing={1} alignItems="center" sx={{ width: 100 }}>
                       <IconButton size="small" onClick={toggleMute}>
                           {isMuted || volume === 0 ? <VolumeOffIcon fontSize="small" /> : <VolumeUpIcon fontSize="small" />}
                       </IconButton>
                       <Slider 
                           size="small"
                           value={volume}
                           min={0}
                           max={1}
                           step={0.1}
                           onChange={handleVolumeChange}
                       />
                   </Stack>
                </Box>
                
              </CardContent>
            </Card>
          </Grid>

          {/* Right Column - Annotation & Additional Classes - ENHANCED BUTTONS */}
          <Grid item xs={12} md={4}>
            <Card elevation={0} sx={{ border: '1px solid #e0e0e0' }}>
              <CardContent sx={{ p: 2 }}>
                {/* Annotation Controls */}
                <Typography variant="h6" sx={{ mb: 1.5, fontWeight: 600 }}>
                  Annotation
                </Typography>

                <Stack spacing={1} sx={{ mb: 3 }}>
                    <Button
                        fullWidth
                        variant="outlined"
                        onClick={() => annotate(1)}
                        startIcon={<CheckCircleIcon sx={{ fontSize: 32 }} />}
                        sx={{
                            height: 80,
                            flexDirection: 'row',
                            justifyContent: 'flex-start',
                            px: 3,
                            color: '#10b981',
                            borderColor: '#10b981',
                            '&:hover': { borderColor: '#10b981', backgroundColor: '#10b98110' },
                            gap: 2
                        }}
                    >
                        <Box sx={{ textAlign: 'left', flexGrow: 1 }}>
                            <Typography variant="button" sx={{ fontWeight: 'bold', display: 'block' }}>Present</Typography>
                            <Typography variant="caption">Confirm detection</Typography>
                        </Box>
                        <Typography variant="caption" sx={{ fontWeight: 'bold', opacity: 0.7 }}>(C / Y)</Typography>
                    </Button>

                    <Button
                        fullWidth
                        variant="outlined"
                        onClick={() => annotate(0)}
                        startIcon={<CancelIcon sx={{ fontSize: 32 }} />}
                        sx={{
                            height: 80,
                            flexDirection: 'row',
                            justifyContent: 'flex-start',
                            px: 3,
                            color: '#ef4444',
                            borderColor: '#ef4444',
                            '&:hover': { borderColor: '#ef4444', backgroundColor: '#ef444410' },
                            gap: 2
                        }}
                    >
                        <Box sx={{ textAlign: 'left', flexGrow: 1 }}>
                            <Typography variant="button" sx={{ fontWeight: 'bold', display: 'block' }}>Absent</Typography>
                            <Typography variant="caption">Negative detection</Typography>
                        </Box>
                        <Typography variant="caption" sx={{ fontWeight: 'bold', opacity: 0.7 }}>(R / N)</Typography>
                    </Button>

                    <Button
                        fullWidth
                        variant="outlined"
                        onClick={() => annotate(3)}
                        startIcon={<HelpIcon sx={{ fontSize: 32 }} />}
                        sx={{
                            height: 80,
                            flexDirection: 'row',
                            justifyContent: 'flex-start',
                            px: 3,
                            color: '#f59e0b',
                            borderColor: '#f59e0b',
                            '&:hover': { borderColor: '#f59e0b', backgroundColor: '#f59e0b10' },
                            gap: 2
                        }}
                    >
                        <Box sx={{ textAlign: 'left', flexGrow: 1 }}>
                            <Typography variant="button" sx={{ fontWeight: 'bold', display: 'block' }}>Unsure</Typography>
                            <Typography variant="caption">Needs further review</Typography>
                        </Box>
                        <Typography variant="caption" sx={{ fontWeight: 'bold', opacity: 0.7 }}>(U)</Typography>
                    </Button>
                </Stack>

                {/* Keyboard Shortcuts Help */}
                <Box sx={{ mb: 3, p: 1.5, backgroundColor: '#f8f9fa', borderRadius: 1, border: '1px solid #e0e0e0' }}>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5, fontWeight: 600 }}>
                    Keyboard Shortcuts:
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem', lineHeight: 1.4 }}>
                    <strong>Spacebar:</strong> Play/Pause • <strong>Arrow Keys:</strong> Navigate
                  </Typography>
                </Box>

                {reviewMode === 'review_annotated' && (
                  <Button
                    fullWidth
                    variant="outlined"
                    color="error"
                    onClick={deleteAnnotation}
                    startIcon={<DeleteIcon />}
                    sx={{ mb: 3 }}
                  >
                    Delete Annotation
                  </Button>
                )}

                {/* Additional Classes */}
                {availableClasses.length > 1 && (
                  <>
                    <Divider sx={{ mb: 1.5 }} />
                    <Typography variant="h6" sx={{ mb: 1.5, fontWeight: 600 }}>
                      Additional Classes
                    </Typography>

                    <Stack spacing={1.5}>
                      <Select
                        isMulti
                        options={availableClasses.filter(c => c.value !== currentClassIndex)}
                        value={additionalPositiveClasses}
                        onChange={(selected) => setAdditionalPositiveClasses(selected || [])}
                        placeholder="Select additional present classes..."
                        styles={{
                          control: (base) => ({
                            ...base,
                            minHeight: '36px',
                            fontSize: '0.875rem',
                          }),
                        }}
                      />

                      <Button
                        fullWidth
                        variant="contained"
                        onClick={annotateAdditionalClasses}
                        disabled={additionalPositiveClasses.length === 0}
                        size="small"
                      >
                        MARK AS PRESENT
                      </Button>

                      <Button
                        fullWidth
                        variant="outlined"
                        onClick={markOtherClassesAsAbsent}
                        size="small"
                      >
                        MARK ALL OTHERS AS ABSENT
                      </Button>
                    </Stack>
                  </>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default ActiveLearning;
