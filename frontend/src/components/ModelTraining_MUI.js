import React, { useState } from 'react';
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
  Select,
  MenuItem,
  Grid,
  Typography,
  FormControlLabel,
  Radio,
  RadioGroup,
  Checkbox,
  Divider,
  Paper,
  Stack,
  Collapse,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  Folder as FolderIcon,
  Description as FileIcon,
  Settings as SettingsIcon,
  PlayArrow as StartIcon,
  Stop as StopIcon,
  ExpandMore as ExpandMoreIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Loop as LoadingIcon,
  Save as SaveIcon,
  Science as ScienceIcon,
  Timeline as TimelineIcon,
  Terminal as TerminalIcon,
  Tune as TuneIcon,
} from '@mui/icons-material';

const ModelTraining = ({ isActive = true }) => {
  const [trainingAudioFolder, setTrainingAudioFolder] = useState('');
  const [metadataPath, setMetadataPath] = useState('');
  const [testDataMode, setTestDataMode] = useState('split');
  const [testSplit, setTestSplit] = useState(0.2);
  const [testAudioFolder, setTestAudioFolder] = useState('');
  const [randomState, setRandomState] = useState(42);
  const [modelSavePath, setModelSavePath] = useState('');
  
  // Training parameters
  const [nSteps, setNSteps] = useState(1000);
  const [batchSize, setBatchSize] = useState(128);
  const [learningRate, setLearningRate] = useState(0.01);
  const [modelType, setModelType] = useState(2);
  const [verbose, setVerbose] = useState(true);
  const [weakNegWeight, setWeakNegWeight] = useState(0.05);

  // Early stopping and learning rate reduction
  const [enableEarlyStopping, setEnableEarlyStopping] = useState(true);
  const [enableLrReduction, setEnableLrReduction] = useState(true);
  const [lrRedux, setLrRedux] = useState(0.5);
  const [patience, setPatience] = useState(5000);
  const [lrReducePatience, setLrReducePatience] = useState(1000);
  const [metricForTracking, setMetricForTracking] = useState('loss');

  const [isLoading, setIsLoading] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingResults, setTrainingResults] = useState(null);
  const [trainingLogs, setTrainingLogs] = useState([]);
  
  // Data loading states
  const [datasetLoaded, setDatasetLoaded] = useState(false);
  const [previewData, setPreviewData] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  
  const [configExpanded, setConfigExpanded] = useState(true);

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
    setTrainingStatus('starting');
    
    // Collapse config to focus on logs
    setConfigExpanded(false);
    
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
          verbose: verbose,
          weak_neg_weight: weakNegWeight,
          enable_early_stopping: enableEarlyStopping,
          enable_lr_reduction: enableLrReduction,
          lr_redux: lrRedux,
          patience: patience,
          lr_reduce_patience: lrReducePatience,
          metric_for_tracking: metricForTracking
        }
      };

      const response = await axios.post('/api/model-training/start', config);
      
      if (response.data.status === 'started') {
        toast.success('Model training started!');
        setTrainingStatus('training');
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
        setTimeout(pollTrainingStatus, 3000);
      }
    } catch (error) {
      console.error('Failed to check training status:', error);
      setTimeout(pollTrainingStatus, 5000);
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

  const loadTrainingDataset = async () => {
    if (!trainingAudioFolder.trim() || !metadataPath.trim()) {
      toast.error('Please specify both training audio folder and metadata file path');
      return;
    }

    setPreviewLoading(true);
    setPreviewData(null);
    setDatasetLoaded(false);
    
    try {
      const response = await axios.post('/api/model-training/preview-data', null, {
        params: {
          training_audio_folder: trainingAudioFolder,
          metadata_path: metadataPath
        }
      });
      
      if (response.data.status === 'success') {
        setPreviewData(response.data.data);
        setDatasetLoaded(true);
        toast.success(`Successfully loaded ${response.data.data.total_files} training files`);
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to load training dataset';
      toast.error(message);
      setDatasetLoaded(false);
    } finally {
      setPreviewLoading(false);
    }
  };

  return (
    <Box sx={{ display: isActive ? 'block' : 'none', pb: 4 }}>
      {/* 1. Data Setup Section */}
      <Card elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
            <FolderIcon color="primary" /> Data Setup
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                size="small"
                label="Training Audio Folder"
                placeholder="/path/to/training/audio/files"
                value={trainingAudioFolder}
                onChange={(e) => setTrainingAudioFolder(e.target.value)}
                disabled={isLoading || previewLoading}
                helperText="Folder containing labeled audio files"
                InputProps={{
                  startAdornment: <FolderIcon sx={{ mr: 1, color: 'text.secondary', fontSize: 20 }} />,
                }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                size="small"
                label="Dataset Metadata File"
                placeholder="/path/to/dataset/metadata.json"
                value={metadataPath}
                onChange={(e) => setMetadataPath(e.target.value)}
                disabled={isLoading || previewLoading}
                helperText="Metadata file from Dataset Builder"
                InputProps={{
                  startAdornment: <FileIcon sx={{ mr: 1, color: 'text.secondary', fontSize: 20 }} />,
                }}
              />
            </Grid>
          </Grid>

          <Box sx={{ mt: 2, textAlign: 'center' }}>
            <Button
              variant={datasetLoaded ? "outlined" : "contained"}
              color={datasetLoaded ? "success" : "primary"}
              onClick={loadTrainingDataset}
              disabled={previewLoading || !trainingAudioFolder.trim() || !metadataPath.trim()}
              startIcon={previewLoading ? <LoadingIcon /> : datasetLoaded ? <SuccessIcon /> : <FolderIcon />}
            >
              {previewLoading ? 'Loading...' : datasetLoaded ? 'Dataset Loaded' : 'Load Training Dataset'}
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* 2. Dataset Preview Accordion */}
      {datasetLoaded && previewData && (
        <Accordion elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2, '&:before': { display: 'none' } }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
                <ScienceIcon color="primary" fontSize="small" /> Dataset Preview
              </Typography>
              <Chip 
                label={`${previewData.total_files} files`}
                size="small"
                color="success"
                variant="outlined" 
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={12} md={4}>
                <Typography variant="subtitle2" gutterBottom color="text.secondary">Configuration</Typography>
                <Stack direction="row" spacing={2} sx={{ mb: 1 }}>
                    <Chip size="small" label={`Model: ${previewData.backend_model}`} />
                    <Chip size="small" label={previewData.use_label_strength ? 'Strength Labels: ON' : 'Strength Labels: OFF'} color={previewData.use_label_strength ? "primary" : "default"} />
                </Stack>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="subtitle2" gutterBottom color="text.secondary">Embeddings Shape</Typography>
                <Box sx={{ fontFamily: 'monospace', fontSize: '0.875rem', color: 'primary.main', fontWeight: 600 }}>
                  {previewData.embedding_shape ? `(${previewData.embedding_shape[0]}, ${previewData.embedding_shape[1]})` : 'Loading...'}
                </Box>
                {previewData.embedding_shape && (
                  <Typography variant="caption" color="text.secondary">
                    {previewData.embedding_shape[0]} samples × {previewData.embedding_shape[1]} features
                  </Typography>
                )}
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="subtitle2" gutterBottom color="text.secondary">Classes</Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {Object.keys(previewData.class_map).map(className => (
                    <Chip key={className} label={className} size="small" variant="outlined" />
                  ))}
                </Box>
              </Grid>
            </Grid>

            <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 300 }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>File Name</TableCell>
                    <TableCell>Label Vector</TableCell>
                    {previewData.use_label_strength && <TableCell>Strength</TableCell>}
                    {Object.keys(previewData.class_map).map(cls => (
                      <TableCell key={cls} align="center">{cls}</TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {previewData.files.map((file, idx) => (
                    <TableRow key={idx}>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>{file.file_name}</TableCell>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>[{file.raw_label_vector?.join(', ')}]</TableCell>
                      {previewData.use_label_strength && (
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                          [{file.raw_strength_vector?.map(v => v.toFixed(2)).join(', ')}]
                        </TableCell>
                      )}
                      {Object.keys(previewData.class_map).map(cls => {
                        const label = file.class_labels[cls];
                        const isPresent = label === 'Present';
                        return (
                          <TableCell key={cls} align="center">
                            <Box 
                              sx={{ 
                                width: 12, 
                                height: 12, 
                                borderRadius: '50%', 
                                bgcolor: isPresent ? 'success.main' : 'grey.300',
                                mx: 'auto'
                              }} 
                              title={label}
                            />
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </AccordionDetails>
        </Accordion>
      )}

      {/* 3. Configuration Section */}
      <Accordion 
        expanded={configExpanded} 
        onChange={() => setConfigExpanded(!configExpanded)}
        elevation={0} 
        sx={{ border: '1px solid #e0e0e0', mb: 2, '&:before': { display: 'none' } }}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
            <SettingsIcon color="primary" /> Configuration
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          {/* Test Data Config */}
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>TEST DATA STRATEGY</Typography>
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12}>
                <FormControl component="fieldset">
                    <RadioGroup row value={testDataMode} onChange={(e) => setTestDataMode(e.target.value)}>
                        <FormControlLabel value="split" control={<Radio size="small" />} label="Train/Test Split" />
                        <FormControlLabel value="folder" control={<Radio size="small" />} label="Separate Test Folder" />
                    </RadioGroup>
                </FormControl>
            </Grid>
            {testDataMode === 'split' ? (
                <>
                    <Grid item xs={6} md={3}>
                        <TextField
                            fullWidth size="small" type="number"
                            label="Split Ratio"
                            value={testSplit}
                            onChange={(e) => setTestSplit(parseFloat(e.target.value))}
                            inputProps={{ min: 0, max: 1, step: 0.05 }}
                            helperText="Validation fraction (0-1)"
                        />
                    </Grid>
                    <Grid item xs={6} md={3}>
                        <TextField
                            fullWidth size="small" type="number"
                            label="Random Seed"
                            value={randomState}
                            onChange={(e) => setRandomState(parseInt(e.target.value))}
                        />
                    </Grid>
                </>
            ) : (
                <Grid item xs={12} md={6}>
                    <TextField
                        fullWidth size="small"
                        label="Test Audio Folder"
                        placeholder="/path/to/test/audio"
                        value={testAudioFolder}
                        onChange={(e) => setTestAudioFolder(e.target.value)}
                        InputProps={{ startAdornment: <FolderIcon sx={{ mr: 1, color: 'text.secondary' }} /> }}
                    />
                </Grid>
            )}
            <Grid item xs={12}>
                <TextField
                    fullWidth size="small"
                    label="Model Save Path"
                    placeholder="/path/to/save/model.keras"
                    value={modelSavePath}
                    onChange={(e) => setModelSavePath(e.target.value)}
                    InputProps={{ startAdornment: <SaveIcon sx={{ mr: 1, color: 'text.secondary' }} /> }}
                />
            </Grid>
          </Grid>
          
          <Divider sx={{ my: 2 }} />

          {/* Hyperparameters */}
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>HYPERPARAMETERS</Typography>
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={6} md={3}>
                <TextField
                    fullWidth size="small" type="number"
                    label="Steps"
                    value={nSteps}
                    onChange={(e) => setNSteps(parseInt(e.target.value))}
                />
            </Grid>
            <Grid item xs={6} md={3}>
                <TextField
                    fullWidth size="small" type="number"
                    label="Batch Size"
                    value={batchSize}
                    onChange={(e) => setBatchSize(parseInt(e.target.value))}
                />
            </Grid>
            <Grid item xs={6} md={3}>
                <TextField
                    fullWidth size="small" type="number"
                    label="Learning Rate"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                    inputProps={{ step: 0.0001 }}
                />
            </Grid>
            <Grid item xs={6} md={3}>
                <TextField
                    fullWidth size="small" type="number"
                    label="Weak Neg Weight"
                    value={weakNegWeight}
                    onChange={(e) => setWeakNegWeight(parseFloat(e.target.value))}
                    inputProps={{ step: 0.01, min: 0.01, max: 1.0 }}
                    helperText="Weight for weak negative samples"
                />
            </Grid>
            <Grid item xs={12} md={6}>
                <FormControl fullWidth size="small">
                    <InputLabel>Architecture</InputLabel>
                    <Select
                        value={modelType}
                        label="Architecture"
                        onChange={(e) => setModelType(e.target.value)}
                    >
                        {modelTypeOptions.map(opt => (
                            <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
                        ))}
                    </Select>
                </FormControl>
            </Grid>
          </Grid>

          {/* Advanced Options Accordion nested */}
          <Accordion elevation={0} sx={{ bgcolor: 'transparent', '&:before': { display: 'none' }, border: 'none' }} disableGutters>
            <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ px: 0, minHeight: 48 }}>
               <Typography variant="subtitle2" color="primary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                 <TuneIcon fontSize="small" /> Advanced Options
               </Typography>
            </AccordionSummary>
            <AccordionDetails sx={{ px: 0 }}>
               <Grid container spacing={2}>
                 <Grid item xs={12} md={4}>
                     <Paper variant="outlined" sx={{ p: 2 }}>
                       <FormControlLabel
                          control={<Checkbox checked={enableEarlyStopping} onChange={(e) => setEnableEarlyStopping(e.target.checked)} />}
                          label={<Typography variant="body2" fontWeight="bold">Early Stopping</Typography>}
                       />
                       <Collapse in={enableEarlyStopping}>
                           <TextField
                              fullWidth size="small" type="number"
                              label="Patience (steps)"
                              value={patience}
                              onChange={(e) => setPatience(parseInt(e.target.value))}
                              sx={{ mt: 1 }}
                           />
                       </Collapse>
                     </Paper>
                 </Grid>
                 <Grid item xs={12} md={4}>
                     <Paper variant="outlined" sx={{ p: 2 }}>
                       <FormControlLabel
                          control={<Checkbox checked={enableLrReduction} onChange={(e) => setEnableLrReduction(e.target.checked)} />}
                          label={<Typography variant="body2" fontWeight="bold">LR Reduction</Typography>}
                       />
                       <Collapse in={enableLrReduction}>
                           <Stack spacing={1} sx={{ mt: 1 }}>
                              <TextField
                                  fullWidth size="small" type="number"
                                  label="Patience (steps)"
                                  value={lrReducePatience}
                                  onChange={(e) => setLrReducePatience(parseInt(e.target.value))}
                              />
                              <TextField
                                  fullWidth size="small" type="number"
                                  label="Factor (0-1)"
                                  value={lrRedux}
                                  onChange={(e) => setLrRedux(parseFloat(e.target.value))}
                                  inputProps={{ step: 0.1 }}
                              />
                           </Stack>
                       </Collapse>
                     </Paper>
                 </Grid>
                 <Grid item xs={12} md={4}>
                     <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
                        <Typography variant="caption" color="text.secondary" display="block" mb={1}>METRIC & LOGGING</Typography>
                        <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                            <InputLabel>Tracking Metric</InputLabel>
                            <Select
                                value={metricForTracking}
                                label="Tracking Metric"
                                onChange={(e) => setMetricForTracking(e.target.value)}
                            >
                                <MenuItem value="cmap">Macro cMAP</MenuItem>
                                <MenuItem value="auc">Macro AUC</MenuItem>
                                <MenuItem value="geomean">Geometric Mean</MenuItem>
                                <MenuItem value="loss">Validation Loss</MenuItem>
                            </Select>
                        </FormControl>
                        <FormControlLabel
                            control={<Checkbox checked={verbose} onChange={(e) => setVerbose(e.target.checked)} size="small" />}
                            label={<Typography variant="body2">Verbose Logging</Typography>}
                         />
                     </Paper>
                 </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>

          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center', gap: 2 }}>
             {!trainingStatus || trainingStatus === 'idle' || trainingStatus === 'completed' || trainingStatus === 'error' ? (
                <Button 
                    variant="contained" 
                    size="large" 
                    color="primary" 
                    onClick={startTraining}
                    disabled={isLoading || !datasetLoaded}
                    startIcon={isLoading ? <LoadingIcon /> : <StartIcon />}
                    sx={{ minWidth: 200, borderRadius: 2 }}
                >
                    {isLoading ? 'Starting...' : 'Start Training'}
                </Button>
             ) : (
                <Button 
                    variant="contained" 
                    size="large" 
                    color="error" 
                    onClick={stopTraining}
                    startIcon={<StopIcon />}
                    sx={{ minWidth: 200, borderRadius: 2 }}
                >
                    Stop Training
                </Button>
             )}
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* 4. Training Status & Logs - TERMINAL STYLE */}
      {(trainingStatus && trainingStatus !== 'idle') && (
        <Card elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2 }}>
            <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderBottom: '1px solid #e0e0e0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TerminalIcon /> Training Output
                </Typography>
                <Chip 
                    label={trainingStatus.toUpperCase()} 
                    color={trainingStatus === 'training' ? 'warning' : trainingStatus === 'completed' ? 'success' : 'error'}
                    size="small" 
                    variant="outlined"
                />
            </Box>
            <CardContent sx={{ p: 0 }}>
                <Box 
                    sx={{ 
                        height: 400, 
                        overflowY: 'auto', 
                        p: 2, 
                        bgcolor: '#1e1e1e', 
                        color: '#33ff33',
                        fontFamily: '"Fira Code", monospace', 
                        fontSize: '0.85rem',
                        whiteSpace: 'pre-wrap',
                        borderBottomLeftRadius: 4,
                        borderBottomRightRadius: 4,
                    }}
                >
                    {trainingLogs.length > 0 ? trainingLogs.join('\n') : "> Waiting for training logs..."}
                </Box>
            </CardContent>
        </Card>
      )}

      {/* 5. Results Section */}
      {trainingResults && (
        <Card elevation={0} sx={{ border: '1px solid #b3e5fc', bgcolor: '#e1f5fe' }}>
            <CardContent>
                <Typography variant="h6" sx={{ mb: 2, color: '#01579b', display: 'flex', alignItems: 'center', gap: 1 }}>
                    <SuccessIcon /> Training Results
                </Typography>
                <Grid container spacing={3}>
                    <Grid item xs={12} md={3}>
                        <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'white' }}>
                            <Typography variant="caption" color="text.secondary">FINAL LOSS</Typography>
                            <Typography variant="h5" color="error">{trainingResults.final_loss?.toFixed(5) || 'N/A'}</Typography>
                        </Paper>
                    </Grid>
                    <Grid item xs={12} md={3}>
                        <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'white' }}>
                            <Typography variant="caption" color="text.secondary">BEST CMAP</Typography>
                            <Typography variant="h5" color="success.main">{trainingResults.best_cmap?.toFixed(5) || 'N/A'}</Typography>
                        </Paper>
                    </Grid>
                    <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 2, bgcolor: 'white' }}>
                            <Typography variant="subtitle2" gutterBottom>Saved Model</Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, bgcolor: '#f5f5f5', p: 1, borderRadius: 1 }}>
                                <SaveIcon fontSize="small" color="action" />
                                <Typography variant="caption" sx={{ wordBreak: 'break-all', fontFamily: 'monospace' }}>
                                    {trainingResults.model_path}
                                </Typography>
                            </Box>
                        </Paper>
                    </Grid>
                </Grid>
            </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default ModelTraining;