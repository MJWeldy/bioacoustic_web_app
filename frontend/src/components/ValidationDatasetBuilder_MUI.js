import React, { useState } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  CardActionArea,
  TextField,
  Button,
  Grid,
  Typography,
  Divider,
  Stack,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  FormControlLabel,
  Checkbox,
} from '@mui/material';
import {
  Storage as StorageIcon,
  Description as FileIcon,
  Folder as FolderIcon,
  Settings as SettingsIcon,
  PlayArrow as RunIcon,
  CheckCircle as SuccessIcon,
  Loop as LoadingIcon,
  AudioFile as AudioIcon,
  TableChart as TableIcon,
} from '@mui/icons-material';

const ValidationDatasetBuilder = ({ isActive = true }) => {
  // Validation workflow type
  const [validationType, setValidationType] = useState('');

  // Project persistence states (common to all workflows)
  const [projectName, setProjectName] = useState('');
  const [saveLocation, setSaveLocation] = useState('');

  // Workflow 1: Unvalidated Clips
  const [audioDirectory, setAudioDirectory] = useState('');
  const [clipWindowLength, setClipWindowLength] = useState(3.0);
  const [targetClasses, setTargetClasses] = useState('');
  const [strataFile, setStrataFile] = useState('');
  const [useFilenameAsStrata, setUseFilenameAsStrata] = useState(false);

  // Workflow 2: Prediction Sets (Standard)
  const [predictionsFile, setPredictionsFile] = useState('');
  const [predictionAudioDirectory, setPredictionAudioDirectory] = useState('');
  const [modelName, setModelName] = useState('');
  const [formatType, setFormatType] = useState('auto');
  const [recursive, setRecursive] = useState(true);

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
          strata_file: strataFile.trim() || null,
          use_filename_as_strata: useFilenameAsStrata,
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
      }

      if (response && response.data.status === 'success') {
        setLoadSummary(response.data);
        const itemCount = response.data.total_predictions || response.data.total_clips || 0;
        toast.success(`Loaded ${itemCount} items successfully`);

        // Show warning about unmapped files if any
        if (response.data.unmapped_files && response.data.unmapped_files > 0) {
          toast.warning(`⚠️ ${response.data.unmapped_files} audio files not found in strata mapping. These will use 'unmapped' as strata.`, { autoClose: 8000 });
        }

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

  const createStrataAutomatic = async () => {
    try {
      const response = await axios.post('/api/validation/create-strata', {});

      if (response.data.status === 'success') {
        setStrataCreated(true);
        setStrataSummary(response.data);
        toast.success(`Automatically created ${response.data.strata_created} validation strata`);

        // Automatically save project after strata creation if save location is provided
        if (saveLocation.trim()) {
          console.log("Auto-saving project to:", saveLocation);
          await saveProjectAutomatic();
        } else {
            console.warn("Skipping auto-save: No save location provided");
        }
      } else {
        toast.error(response.data.message || 'Failed to create strata automatically');
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to create strata automatically';
      toast.error(message);
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
        const errorMsg = response.data.message || 'Failed to save project automatically';
        console.error("Auto-save failed:", errorMsg);
        toast.error(errorMsg);
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to save project automatically';
      console.error("Auto-save exception:", error);
      toast.error(message);
    }
  };

  return (
    <Box sx={{ display: isActive ? 'block' : 'none', pb: 4 }}>
      
      {/* Validation Type Selection */}
      <Card elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
            <StorageIcon color="primary" /> Select Validation Type
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Card 
                variant="outlined" 
                sx={{ 
                  height: '100%', 
                  borderColor: validationType === 'unvalidated_clips' ? 'primary.main' : 'divider',
                  bgcolor: validationType === 'unvalidated_clips' ? '#f0f7ff' : 'background.paper',
                  transition: '0.2s',
                  '&:hover': { borderColor: 'primary.light' }
                }}
              >
                <CardActionArea 
                    onClick={() => setValidationType('unvalidated_clips')}
                    sx={{ height: '100%', p: 2, display: 'flex', flexDirection: 'column', alignItems: 'flex-start', justifyContent: 'flex-start' }}
                >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <AudioIcon color={validationType === 'unvalidated_clips' ? 'primary' : 'action'} />
                        <Typography variant="subtitle1" fontWeight="bold">Unvalidated Clips</Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                        Generate fixed-length clips from audio files for manual validation. Ideal for creating ground-truth datasets from raw audio.
                    </Typography>
                </CardActionArea>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card 
                variant="outlined" 
                sx={{ 
                  height: '100%', 
                  borderColor: validationType === 'prediction_sets' ? 'primary.main' : 'divider',
                  bgcolor: validationType === 'prediction_sets' ? '#f0f7ff' : 'background.paper',
                  transition: '0.2s',
                  '&:hover': { borderColor: 'primary.light' }
                }}
              >
                <CardActionArea 
                    onClick={() => setValidationType('prediction_sets')}
                    sx={{ height: '100%', p: 2, display: 'flex', flexDirection: 'column', alignItems: 'flex-start', justifyContent: 'flex-start' }}
                >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <TableIcon color={validationType === 'prediction_sets' ? 'primary' : 'action'} />
                        <Typography variant="subtitle1" fontWeight="bold">Prediction Sets</Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                        Validate model predictions from CSV files. Supports standard wide/long formats with optional site/time strata.
                    </Typography>
                </CardActionArea>
              </Card>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Project Configuration Section */}
      {validationType && (
        <Card elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2 }}>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
              <SettingsIcon color="primary" /> Project Configuration
            </Typography>

            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth size="small"
                  label="Project Save Location *"
                  placeholder="/path/to/validation/projects"
                  value={saveLocation}
                  onChange={(e) => setSaveLocation(e.target.value)}
                  disabled={isLoading}
                  helperText="Absolute path for saving validation results"
                  InputProps={{ startAdornment: <FolderIcon sx={{ mr: 1, color: 'text.secondary' }} /> }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth size="small"
                  label="Project Name (Optional)"
                  placeholder="my_validation_project"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  disabled={isLoading}
                  helperText="Leave blank for auto-generated name"
                />
              </Grid>
            </Grid>

            {!saveLocation.trim() && (
              <Alert severity="warning" sx={{ mt: 2 }}>
                Please specify a save location to ensure your validation progress is preserved.
              </Alert>
            )}
          </CardContent>
        </Card>
      )}

      {/* Workflow 1: Unvalidated Clips */}
      {validationType === 'unvalidated_clips' && (
        <Card elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2 }}>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Unvalidated Clips Configuration</Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth size="small"
                  label="Audio Files Directory *"
                  placeholder="/path/to/audio/files"
                  value={audioDirectory}
                  onChange={(e) => setAudioDirectory(e.target.value)}
                  disabled={isLoading}
                  InputProps={{ startAdornment: <FolderIcon sx={{ mr: 1, color: 'text.secondary' }} /> }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth size="small"
                  type="number"
                  label="Clip Window Length (s) *"
                  value={clipWindowLength}
                  onChange={(e) => setClipWindowLength(parseFloat(e.target.value))}
                  disabled={isLoading}
                  inputProps={{ step: 0.1, min: 0.1 }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth size="small"
                  label="Target Class Names *"
                  placeholder="Species1, Species2, Species3"
                  value={targetClasses}
                  onChange={(e) => setTargetClasses(e.target.value)}
                  disabled={isLoading}
                  helperText="Comma-separated"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth size="small"
                  label="Strata Metadata File (Optional)"
                  placeholder="/path/to/strata.csv"
                  value={strataFile}
                  onChange={(e) => setStrataFile(e.target.value)}
                  disabled={isLoading || useFilenameAsStrata}
                  helperText="CSV file with 'filename' and 'strata' columns"
                  InputProps={{ startAdornment: <FileIcon sx={{ mr: 1, color: 'text.secondary' }} /> }}
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={useFilenameAsStrata}
                      onChange={(e) => {
                        setUseFilenameAsStrata(e.target.checked);
                        if (e.target.checked) {
                          setStrataFile(''); // Clear strata file when using filename
                        }
                      }}
                      disabled={isLoading}
                    />
                  }
                  label="Use audio filename as strata (keeps clips from each file sequential)"
                />
              </Grid>
            </Grid>

            <Box sx={{ mt: 3, textAlign: 'center' }}>
              <Button
                variant="contained"
                size="large"
                onClick={loadValidationDataset}
                disabled={isLoading || !audioDirectory.trim() || !targetClasses.trim()}
                startIcon={isLoading ? <LoadingIcon /> : <RunIcon />}
                sx={{ minWidth: 250 }}
              >
                {isLoading ? 'Creating Dataset...' : 'Create Validation Dataset'}
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Workflow 2: Standard Prediction Sets */}
      {validationType === 'prediction_sets' && (
        <Card elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2 }}>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Prediction Sets Configuration</Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth size="small"
                  label="Predictions File/Directory *"
                  placeholder="/path/to/predictions.csv"
                  value={predictionsFile}
                  onChange={(e) => setPredictionsFile(e.target.value)}
                  disabled={isLoading}
                  InputProps={{ startAdornment: <FileIcon sx={{ mr: 1, color: 'text.secondary' }} /> }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth size="small"
                  label="Audio Files Directory (Optional)"
                  placeholder="/path/to/audio/files"
                  value={predictionAudioDirectory}
                  onChange={(e) => setPredictionAudioDirectory(e.target.value)}
                  disabled={isLoading}
                  InputProps={{ startAdornment: <FolderIcon sx={{ mr: 1, color: 'text.secondary' }} /> }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth size="small"
                  label="Model Name *"
                  placeholder="BirdNET, PERCH, etc."
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  disabled={isLoading}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth size="small">
                    <InputLabel>Data Format</InputLabel>
                    <Select
                        value={formatType}
                        label="Data Format"
                        onChange={(e) => setFormatType(e.target.value)}
                        disabled={isLoading}
                    >
                        {formatOptions.map(opt => <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>)}
                    </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                 <FormControlLabel
                    control={<Checkbox checked={recursive} onChange={(e) => setRecursive(e.target.checked)} disabled={isLoading} />}
                    label="Recursive Search"
                 />
              </Grid>
            </Grid>

            <Box sx={{ mt: 3, textAlign: 'center' }}>
              <Button
                variant="contained"
                size="large"
                onClick={loadValidationDataset}
                disabled={isLoading || !predictionsFile.trim() || !modelName.trim()}
                startIcon={isLoading ? <LoadingIcon /> : <RunIcon />}
                sx={{ minWidth: 250 }}
              >
                {isLoading ? 'Loading Predictions...' : 'Load Prediction Dataset'}
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Load Summary */}
      {loadSummary && (
        <Card elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2, bgcolor: '#f9fafb' }}>
          <CardHeader 
            title={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <SuccessIcon color="success" /> 
                    <Typography variant="h6">Load Summary</Typography>
                </Box>
            }
          />
          <CardContent>
            <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" gutterBottom>Dataset Overview</Typography>
                    <Stack spacing={1}>
                        <Typography variant="body2"><strong>Total Predictions:</strong> {loadSummary.total_predictions}</Typography>
                        <Typography variant="body2"><strong>Unique Files:</strong> {loadSummary.unique_files}</Typography>
                        <Typography variant="body2"><strong>Format:</strong> {loadSummary.format_detected}</Typography>
                        {strataSummary && <Typography variant="body2"><strong>Validation Strata:</strong> {strataSummary.strata_created}</Typography>}
                    </Stack>
                </Grid>
                <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" gutterBottom>Species Detected ({loadSummary.unique_species})</Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {loadSummary.species_list?.slice(0, 20).map(sp => (
                            <Alert icon={false} severity="info" sx={{ py: 0, px: 1 }} key={sp}>{sp}</Alert>
                        ))}
                        {loadSummary.species_list?.length > 20 && <Typography variant="caption">+{loadSummary.species_list.length - 20} more</Typography>}
                    </Box>
                </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Success Message */}
      {strataSummary && (
        <Alert severity="success" variant="filled" sx={{ mb: 2 }}>
            <Typography variant="subtitle1" fontWeight="bold">Dataset Ready!</Typography>
            Validation dataset created successfully. Switch to the <strong>Validation</strong> tab to begin reviewing predictions.
        </Alert>
      )}

    </Box>
  );
};

export default ValidationDatasetBuilder;