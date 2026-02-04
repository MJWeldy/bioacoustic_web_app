import React, { useState, useRef } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  FormControl,
  FormControlLabel,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  Radio,
  RadioGroup,
  IconButton,
  LinearProgress,
  Alert,
  Grid,
  Paper,
  Typography,
  Chip,
  Divider,
  Tooltip,
  Stack,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  SwapHoriz as SwapIcon,
  CheckCircle as CheckCircleIcon,
  Info as InfoIcon,
  Folder as FolderIcon,
  Save as SaveIcon,
} from '@mui/icons-material';

const DatasetBuilder = () => {
  const [audioFolder, setAudioFolder] = useState('');
  const [savePath, setSavePath] = useState('');
  const [backendModel, setBackendModel] = useState('PERCH');
  const [classMap, setClassMap] = useState([{ name: '', value: 0 }]);
  const [isEvaluationDataset, setIsEvaluationDataset] = useState(false);
  const [classMapMode, setClassMapMode] = useState('manual');
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
      let dictData;

      try {
        dictData = JSON.parse(cleanDict);
      } catch (jsonError) {
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
          key = key.replace(/^['"]|['"]$/g, '');
          const numValue = parseInt(value);
          if (isNaN(numValue)) continue;
          dictData[key] = numValue;
        }
      }

      const parsed = [];
      for (const [key, value] of Object.entries(dictData)) {
        if (typeof value === 'number') {
          parsed.push({ name: key, value: value });
        }
      }

      if (parsed.length > 0) {
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
    if (classMapMode === 'dictionary') {
      if (!classMapDict.trim()) {
        toast.error('Please enter a class map dictionary or switch to manual mode');
        return false;
      }
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
      let finalClassMap = classMap;
      if (classMapMode === 'dictionary' && classMapDict.trim()) {
        parseDictionary();
        finalClassMap = classMap;
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
        pollIntervalRef.current = setInterval(() => {
          pollBuildingProgress();
        }, 2000);
        toast.info('Dataset creation started. This may take several minutes...');
      } else if (response.data.status === 'success') {
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

  React.useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  return (
    <Box>
      {/* Current Dataset Status - Dashboard Style */}
      {datasetStatus && datasetStatus.loaded && (
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} md={3}>
            <Paper elevation={0} sx={{ p: 2, border: '1px solid #e0e0e0' }}>
              <Typography variant="caption" color="text.secondary">Total Clips</Typography>
              <Typography variant="h4" sx={{ fontWeight: 600, color: '#1976d2' }}>
                {datasetStatus.clips_count?.toLocaleString() || 0}
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={3}>
            <Paper elevation={0} sx={{ p: 2, border: '1px solid #e0e0e0' }}>
              <Typography variant="caption" color="text.secondary">Backend Model</Typography>
              <Typography variant="h6" sx={{ fontWeight: 600, mt: 1 }}>
                {datasetStatus.backend_model}
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={3}>
            <Paper elevation={0} sx={{ p: 2, border: '1px solid #e0e0e0' }}>
              <Typography variant="caption" color="text.secondary">Dataset Type</Typography>
              <Typography variant="h6" sx={{ fontWeight: 600, mt: 1 }}>
                {datasetStatus.dataset_type || 'Active Learning'}
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={3}>
            <Paper elevation={0} sx={{ p: 2, border: '1px solid #e0e0e0' }}>
              <Typography variant="caption" color="text.secondary">Classes</Typography>
              <Box sx={{ mt: 1 }}>
                {Object.keys(datasetStatus.class_map || {}).slice(0, 2).map(cls => (
                  <Chip
                    key={cls}
                    label={cls}
                    size="small"
                    sx={{ mr: 0.5, mb: 0.5 }}
                  />
                ))}
                {Object.keys(datasetStatus.class_map || {}).length > 2 && (
                  <Chip label={`+${Object.keys(datasetStatus.class_map || {}).length - 2}`} size="small" />
                )}
              </Box>
            </Paper>
          </Grid>
        </Grid>
      )}

      {/* Building Progress */}
      {buildingProgress && (
        <Alert severity="info" icon={<InfoIcon />} sx={{ mb: 3 }}>
          <Typography variant="body2" sx={{ mb: 1, fontWeight: 600 }}>
            {buildingProgress.message || 'Processing audio files and generating embeddings...'}
          </Typography>
          {buildingProgress.progress && (
            <Box>
              <LinearProgress
                variant="determinate"
                value={buildingProgress.progress}
                sx={{ height: 6, borderRadius: 3 }}
              />
              <Typography variant="caption" sx={{ mt: 0.5, display: 'block' }}>
                {buildingProgress.progress}% complete
              </Typography>
            </Box>
          )}
        </Alert>
      )}

      {/* Main Configuration */}
      <Grid container spacing={3}>
        {/* Left Column - Configuration */}
        <Grid item xs={12} lg={8}>
          <Card elevation={0} sx={{ border: '1px solid #e0e0e0' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                Dataset Configuration
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Audio Folder Path"
                    placeholder="/path/to/audio/files"
                    value={audioFolder}
                    onChange={(e) => setAudioFolder(e.target.value)}
                    helperText="Path to folder containing WAV or MP3 files"
                    InputProps={{
                      startAdornment: <FolderIcon sx={{ mr: 1, color: 'text.secondary' }} />,
                    }}
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Save Location"
                    placeholder="/path/to/save/dataset"
                    value={savePath}
                    onChange={(e) => setSavePath(e.target.value)}
                    helperText="Location to save embeddings and database"
                    InputProps={{
                      startAdornment: <SaveIcon sx={{ mr: 1, color: 'text.secondary' }} />,
                    }}
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Backend Model</InputLabel>
                    <Select
                      value={backendModel}
                      onChange={(e) => setBackendModel(e.target.value)}
                      label="Backend Model"
                    >
                      {backendOptions.map(opt => (
                        <MenuItem key={opt.value} value={opt.value}>
                          {opt.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={isEvaluationDataset}
                        onChange={(e) => setIsEvaluationDataset(e.target.checked)}
                      />
                    }
                    label={
                      <Box>
                        <Typography variant="body2">Evaluation Dataset</Typography>
                        <Typography variant="caption" color="text.secondary">
                          Check if dataset contains labeled files
                        </Typography>
                      </Box>
                    }
                  />
                </Grid>
              </Grid>

              <Divider sx={{ my: 3 }} />

              {/* Class Map Section */}
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Class Map
              </Typography>

              <RadioGroup
                row
                value={classMapMode}
                onChange={(e) => setClassMapMode(e.target.value)}
                sx={{ mb: 2 }}
              >
                <FormControlLabel
                  value="manual"
                  control={<Radio size="small" />}
                  label="Build Manually"
                  disabled={isLoading}
                />
                <FormControlLabel
                  value="dictionary"
                  control={<Radio size="small" />}
                  label="Paste Dictionary"
                  disabled={isLoading}
                />
              </RadioGroup>

              {classMapMode === 'dictionary' ? (
                <Box>
                  <TextField
                    fullWidth
                    multiline
                    rows={5}
                    size="small"
                    placeholder={`{"bird_song": 0, "frog_call": 1, "insect_chirp": 2}`}
                    value={classMapDict}
                    onChange={(e) => setClassMapDict(e.target.value)}
                    disabled={isLoading}
                    sx={{
                      fontFamily: 'monospace',
                      fontSize: '0.875rem',
                      mb: 1,
                    }}
                  />
                  <Stack direction="row" spacing={1}>
                    <Button
                      size="small"
                      variant="contained"
                      onClick={parseDictionary}
                      disabled={isLoading || !classMapDict.trim()}
                      startIcon={<RefreshIcon />}
                    >
                      Parse
                    </Button>
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() => setClassMapMode('manual')}
                      disabled={isLoading}
                      startIcon={<SwapIcon />}
                    >
                      Switch to Manual
                    </Button>
                  </Stack>
                </Box>
              ) : (
                <Box>
                  <Stack spacing={1}>
                    {classMap.map((entry, index) => (
                      <Stack key={index} direction="row" spacing={1} alignItems="center">
                        <TextField
                          fullWidth
                          size="small"
                          placeholder="Class name"
                          value={entry.name}
                          onChange={(e) => updateClassMapEntry(index, 'name', e.target.value)}
                          disabled={isLoading}
                        />
                        <TextField
                          size="small"
                          type="number"
                          placeholder="Value"
                          value={entry.value}
                          onChange={(e) => updateClassMapEntry(index, 'value', e.target.value)}
                          disabled={isLoading}
                          sx={{ width: 100 }}
                        />
                        <Tooltip title="Remove class">
                          <span>
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => removeClassMapEntry(index)}
                              disabled={classMap.length === 1 || isLoading}
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </span>
                        </Tooltip>
                      </Stack>
                    ))}
                  </Stack>
                  <Stack direction="row" spacing={1} sx={{ mt: 2 }}>
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={addClassMapEntry}
                      disabled={isLoading}
                      startIcon={<AddIcon />}
                    >
                      Add Class
                    </Button>
                    <Button
                      size="small"
                      variant="contained"
                      onClick={() => {
                        convertToDict();
                        setClassMapMode('dictionary');
                      }}
                      disabled={isLoading || classMap.some(c => !c.name.trim())}
                      startIcon={<SwapIcon />}
                    >
                      Convert to Dictionary
                    </Button>
                  </Stack>
                </Box>
              )}

              <Box sx={{ mt: 3 }}>
                <Button
                  fullWidth
                  variant="contained"
                  size="large"
                  onClick={createDataset}
                  disabled={isLoading}
                  sx={{ py: 1.5 }}
                >
                  {isLoading ? 'Creating Dataset...' : 'Create Dataset'}
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Right Column - Instructions */}
        <Grid item xs={12} lg={4}>
          <Card elevation={0} sx={{ border: '1px solid #e0e0e0', backgroundColor: '#fafafa' }}>
            <CardContent>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
                <InfoIcon color="primary" />
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Instructions
                </Typography>
              </Stack>

              <Typography variant="body2" component="div" sx={{ lineHeight: 1.7, color: '#666' }}>
                <Box component="ol" sx={{ pl: 2, m: 0 }}>
                  <li>
                    <strong>Audio Folder:</strong> Select a folder containing WAV or MP3 files
                  </li>
                  <li>
                    <strong>Class Map:</strong> Define your classification classes manually or paste a dictionary
                  </li>
                  <li>
                    <strong>Backend Model:</strong> Choose the embedding model (PERCH, BirdNET, etc.)
                  </li>
                  <li>
                    <strong>Save Location:</strong> Choose where to store embeddings and database files
                  </li>
                </Box>
              </Typography>

              <Divider sx={{ my: 2 }} />

              <Alert severity="info" icon={<InfoIcon />} sx={{ fontSize: '0.875rem' }}>
                Dataset creation may take several minutes depending on the number and size of audio files.
              </Alert>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DatasetBuilder;
