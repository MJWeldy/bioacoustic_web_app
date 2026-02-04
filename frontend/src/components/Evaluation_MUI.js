import React, { useState } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Folder as FolderIcon,
  Description as FileIcon,
  PlayArrow as RunIcon,
  ExpandMore as ExpandMoreIcon,
  Assessment as AssessmentIcon,
  TableChart as TableChartIcon,
  GetApp as DownloadIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Loop as LoadingIcon,
  Help as HelpIcon,
  Science as ScienceIcon,
} from '@mui/icons-material';

const Evaluation = ({ isActive = true }) => {
  const [evaluationDatasetPath, setEvaluationDatasetPath] = useState('');
  const [classifierPath, setClassifierPath] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [isDatasetLoaded, setIsDatasetLoaded] = useState(false);
  const [isClassifierLoaded, setIsClassifierLoaded] = useState(false);
  const [datasetMetadata, setDatasetMetadata] = useState(null);
  const [threshold, setThreshold] = useState(0);
  const [selectedConfusionClasses, setSelectedConfusionClasses] = useState([]);
  const [selectedPredictionClasses, setSelectedPredictionClasses] = useState([]);
  const [predictionsExpanded, setPredictionsExpanded] = useState(false);

  // --- API Handlers ---
  const loadEvaluationDataset = async () => {
    if (!evaluationDatasetPath.trim()) {
      toast.error('Please specify an evaluation dataset path');
      return;
    }

    setIsLoading(true);
    try {
      const response = await axios.post('/api/evaluation/load-dataset', null, {
        params: { dataset_path: evaluationDatasetPath }
      });
      
      if (response.data.status === 'success') {
        toast.success(response.data.message);
        setIsDatasetLoaded(true);
        setDatasetMetadata(response.data.metadata);
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to load evaluation dataset';
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
      const response = await axios.post('/api/evaluation/load-classifier', null, {
        params: { classifier_path: classifierPath }
      });
      
      if (response.data.status === 'success') {
        toast.success(response.data.message);
        setIsClassifierLoaded(true);
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to load classifier';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const runEvaluation = async () => {
    if (!isDatasetLoaded || !isClassifierLoaded) {
      toast.error('Please load both evaluation dataset and classifier first');
      return;
    }

    setIsLoading(true);
    try {
      const response = await axios.post(`/api/evaluation/run-evaluation?threshold=${threshold}`);
      
      if (response.data.status === 'success') {
        toast.success('Evaluation completed successfully');
        setEvaluationResults(response.data.results);
        // Initialize class selections
        if (response.data.results.class_names && response.data.results.class_names.length > 0) {
          setSelectedConfusionClasses([response.data.results.class_names[0]]);
          setSelectedPredictionClasses([response.data.results.class_names[0]]);
        }
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to run evaluation';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const exportMetricsCSV = async () => {
    const exportPath = prompt('Enter export directory path:');
    if (!exportPath) return;
    
    const fileName = prompt('Enter filename (without extension):', 'evaluation_metrics');
    if (!fileName) return;

    try {
      const response = await axios.post('/api/evaluation/export-metrics-csv', null, {
        params: { export_path: exportPath, filename: fileName }
      });
      if (response.data.status === 'success') toast.success(response.data.message);
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to export metrics CSV');
    }
  };

  const exportPredictionsCSV = async () => {
    const exportPath = prompt('Enter export directory path:');
    if (!exportPath) return;
    
    const fileName = prompt('Enter filename (without extension):', 'evaluation_predictions');
    if (!fileName) return;

    try {
      const response = await axios.post('/api/evaluation/export-predictions-csv', null, {
        params: { export_path: exportPath, filename: fileName }
      });
      if (response.data.status === 'success') toast.success(response.data.message);
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to export predictions CSV');
    }
  };

  // --- Render Helpers ---
  const renderSingleClassResults = (results) => (
    <Grid container spacing={2}>
      <Grid item xs={12} md={6}>
        <Card variant="outlined" sx={{ height: '100%' }}>
            <CardHeader title="Performance Metrics" subheader={`Evaluation Level: ${results.evaluation_level} (${results.num_samples} samples)`} />
            <CardContent>
                <Grid container spacing={2}>
                    <Grid item xs={6}>
                        <Paper sx={{ p: 2, textAlign: 'center', bgcolor: '#f5f5f5' }} elevation={0}>
                            <Typography variant="caption" color="text.secondary">AUC</Typography>
                            <Typography variant="h4" color="primary">{results.auc.toFixed(4)}</Typography>
                        </Paper>
                    </Grid>
                    <Grid item xs={6}>
                        <Paper sx={{ p: 2, textAlign: 'center', bgcolor: '#f5f5f5' }} elevation={0}>
                            <Typography variant="caption" color="text.secondary">Avg Precision</Typography>
                            <Typography variant="h4" color="primary">{results.average_precision.toFixed(4)}</Typography>
                        </Paper>
                    </Grid>
                </Grid>
            </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={6}>
        <Card variant="outlined" sx={{ height: '100%' }}>
            <CardHeader title="Confusion Matrix" />
            <CardContent>
                <TableContainer component={Paper} elevation={0} variant="outlined">
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell></TableCell>
                                <TableCell align="center" sx={{ bgcolor: '#fff3e0' }}>Pred Neg</TableCell>
                                <TableCell align="center" sx={{ bgcolor: '#fff3e0' }}>Pred Pos</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            <TableRow>
                                <TableCell component="th" scope="row" sx={{ bgcolor: '#fff3e0', fontWeight: 'bold' }}>Actual Neg</TableCell>
                                <TableCell align="center" sx={{ bgcolor: '#e8f5e9', fontWeight: 'bold' }}>{results.confusion_matrix[0][0]}</TableCell>
                                <TableCell align="center">{results.confusion_matrix[0][1]}</TableCell>
                            </TableRow>
                            <TableRow>
                                <TableCell component="th" scope="row" sx={{ bgcolor: '#fff3e0', fontWeight: 'bold' }}>Actual Pos</TableCell>
                                <TableCell align="center">{results.confusion_matrix[1][0]}</TableCell>
                                <TableCell align="center" sx={{ bgcolor: '#e8f5e9', fontWeight: 'bold' }}>{results.confusion_matrix[1][1]}</TableCell>
                            </TableRow>
                        </TableBody>
                    </Table>
                </TableContainer>
            </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const renderMultiClassResults = (results) => (
    <Stack spacing={2}>
        {/* Overall Metrics */}
        <Grid container spacing={2}>
            <Grid item xs={12} md={5}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                    <CardHeader title="Overall Performance" subheader={`${results.evaluation_level} level`} />
                    <CardContent>
                         <Grid container spacing={2}>
                            <Grid item xs={6}>
                                <Paper sx={{ p: 2, textAlign: 'center', bgcolor: '#f5f5f5' }} elevation={0}>
                                    <Typography variant="caption" color="text.secondary">Macro AUC</Typography>
                                    <Typography variant="h4" color="primary">{results.macro_auc?.toFixed(4) || 'N/A'}</Typography>
                                </Paper>
                            </Grid>
                            <Grid item xs={6}>
                                <Paper sx={{ p: 2, textAlign: 'center', bgcolor: '#f5f5f5' }} elevation={0}>
                                    <Typography variant="caption" color="text.secondary">Mean AP</Typography>
                                    <Typography variant="h4" color="primary">{results.mean_ap?.toFixed(4) || 'N/A'}</Typography>
                                </Paper>
                            </Grid>
                         </Grid>
                    </CardContent>
                </Card>
            </Grid>
            <Grid item xs={12} md={7}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                    <CardHeader title="Class Metrics" />
                    <TableContainer sx={{ maxHeight: 200 }}>
                        <Table stickyHeader size="small">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Class</TableCell>
                                    <TableCell align="center">AUC</TableCell>
                                    <TableCell align="center">AP</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {results.class_names.map((className, idx) => (
                                    <TableRow key={idx}>
                                        <TableCell component="th" scope="row">{className}</TableCell>
                                        <TableCell align="center"><strong>{results.class_aucs[idx]?.toFixed(4) || 'N/A'}</strong></TableCell>
                                        <TableCell align="center"><strong>{results.class_aps[idx]?.toFixed(4) || 'N/A'}</strong></TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </Card>
            </Grid>
        </Grid>

        {/* Confusion Matrix */}
        <Card variant="outlined">
            <CardHeader 
                title={results.confusion_matrix_type === 'multilabel' ? 'Binary Confusion Matrices' : 'Multiclass Confusion Matrix'} 
                action={
                    results.confusion_matrix_type === 'multilabel' && (
                        <FormControl size="small" sx={{ minWidth: 200 }}>
                            <InputLabel>Select Class</InputLabel>
                            <Select
                                multiple
                                value={selectedConfusionClasses}
                                label="Select Class"
                                onChange={(e) => setSelectedConfusionClasses(typeof e.target.value === 'string' ? e.target.value.split(',') : e.target.value)}
                                renderValue={(selected) => selected.join(', ')}
                            >
                                {results.class_names.map((name) => (
                                    <MenuItem key={name} value={name}>{name}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    )
                }
            />
            <CardContent>
                {results.confusion_matrix_type === 'multilabel' ? (
                    <Grid container spacing={2}>
                        {results.confusion_matrix.map((cm, idx) => {
                            const className = results.class_names[idx];
                            if (!selectedConfusionClasses.includes(className)) return null;
                            return (
                                <Grid item xs={12} md={6} lg={4} key={idx}>
                                    <Paper variant="outlined" sx={{ p: 2 }}>
                                        <Typography variant="subtitle2" align="center" gutterBottom>{className}</Typography>
                                        <Table size="small">
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell></TableCell>
                                                    <TableCell align="center" sx={{ fontSize: '0.75rem', bgcolor: '#fff3e0' }}>Pred Neg</TableCell>
                                                    <TableCell align="center" sx={{ fontSize: '0.75rem', bgcolor: '#fff3e0' }}>Pred Pos</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                <TableRow>
                                                    <TableCell sx={{ fontSize: '0.75rem', fontWeight: 'bold', bgcolor: '#fff3e0' }}>Act Neg</TableCell>
                                                    <TableCell align="center" sx={{ bgcolor: '#e8f5e9' }}>{cm[0][0]}</TableCell>
                                                    <TableCell align="center">{cm[0][1]}</TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell sx={{ fontSize: '0.75rem', fontWeight: 'bold', bgcolor: '#fff3e0' }}>Act Pos</TableCell>
                                                    <TableCell align="center">{cm[1][0]}</TableCell>
                                                    <TableCell align="center" sx={{ bgcolor: '#e8f5e9' }}>{cm[1][1]}</TableCell>
                                                </TableRow>
                                            </TableBody>
                                        </Table>
                                    </Paper>
                                </Grid>
                            );
                        })}
                    </Grid>
                ) : (
                    <TableContainer sx={{ maxHeight: 400 }}>
                        <Table size="small" stickyHeader>
                            <TableHead>
                                <TableRow>
                                    <TableCell sx={{ bgcolor: 'white' }}>Actual \ Pred</TableCell>
                                    {results.class_names.map((name, i) => (
                                        <TableCell key={i} align="center" sx={{ bgcolor: '#fff3e0', fontWeight: 'bold' }}>{name}</TableCell>
                                    ))}
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {results.confusion_matrix.map((row, i) => (
                                    <TableRow key={i}>
                                        <TableCell component="th" scope="row" sx={{ bgcolor: '#fff3e0', fontWeight: 'bold' }}>{results.class_names[i]}</TableCell>
                                        {row.map((val, j) => (
                                            <TableCell key={j} align="center" sx={{ bgcolor: i === j ? '#e8f5e9' : 'inherit' }}>{val}</TableCell>
                                        ))}
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                )}
            </CardContent>
        </Card>
    </Stack>
  );

  return (
    <Box sx={{ display: isActive ? 'block' : 'none', pb: 4 }}>
      {/* 1. Setup Section */}
      <Card elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
            <AssessmentIcon color="primary" /> Evaluation Setup
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={5}>
              <TextField
                fullWidth size="small"
                label="Evaluation Dataset Path"
                placeholder="/path/to/eval/dataset"
                value={evaluationDatasetPath}
                onChange={(e) => setEvaluationDatasetPath(e.target.value)}
                InputProps={{ startAdornment: <FolderIcon sx={{ mr: 1, color: 'text.secondary', fontSize: 20 }} /> }}
              />
            </Grid>
            <Grid item xs={12} md={5}>
              <TextField
                fullWidth size="small"
                label="Classifier Model Path"
                placeholder="/path/to/model.keras"
                value={classifierPath}
                onChange={(e) => setClassifierPath(e.target.value)}
                InputProps={{ startAdornment: <FileIcon sx={{ mr: 1, color: 'text.secondary', fontSize: 20 }} /> }}
              />
            </Grid>
            <Grid item xs={12} md={2}>
              <TextField
                 fullWidth size="small"
                 type="number"
                 label="Threshold"
                 value={threshold}
                 onChange={(e) => setThreshold(parseFloat(e.target.value) || 0)}
                 helperText="Min samples/class"
              />
            </Grid>
          </Grid>
          
          <Stack direction="row" spacing={2} justifyContent="center" sx={{ mt: 3 }}>
              <Button 
                variant={isDatasetLoaded ? "outlined" : "contained"} 
                color={isDatasetLoaded ? "success" : "primary"}
                onClick={loadEvaluationDataset} 
                disabled={isLoading}
                startIcon={isDatasetLoaded ? <SuccessIcon /> : <FolderIcon />}
              >
                {isDatasetLoaded ? "Dataset Loaded" : "Load Dataset"}
              </Button>
              <Button 
                variant={isClassifierLoaded ? "outlined" : "contained"} 
                color={isClassifierLoaded ? "success" : "secondary"}
                onClick={loadClassifier} 
                disabled={isLoading}
                startIcon={isClassifierLoaded ? <SuccessIcon /> : <FileIcon />}
              >
                {isClassifierLoaded ? "Classifier Loaded" : "Load Classifier"}
              </Button>
          </Stack>

          {/* Dataset Info Chip */}
          {isDatasetLoaded && datasetMetadata && (
             <Alert severity="info" sx={{ mt: 2, py: 0 }} icon={<ScienceIcon />}>
                <Typography variant="caption">
                    <strong>Loaded Dataset:</strong> {datasetMetadata.dataset_info?.dataset_type} • 
                    <strong> Model:</strong> {datasetMetadata.dataset_info?.backend_model} • 
                    <strong> Files:</strong> {datasetMetadata.statistics?.total_files}
                </Typography>
             </Alert>
          )}

          <Divider sx={{ my: 2 }} />

          <Box sx={{ textAlign: 'center' }}>
             <Button
                variant="contained"
                size="large"
                color="success"
                onClick={runEvaluation}
                disabled={isLoading || !isDatasetLoaded || !isClassifierLoaded}
                startIcon={isLoading ? <LoadingIcon /> : <RunIcon />}
                sx={{ minWidth: 200, borderRadius: 2 }}
             >
                {isLoading ? "Evaluating..." : "Run Evaluation"}
             </Button>
          </Box>
        </CardContent>
      </Card>

      {/* 2. Results Section */}
      {evaluationResults && (
        <Stack spacing={2}>
            {evaluationResults.is_single_class 
                ? renderSingleClassResults(evaluationResults)
                : renderMultiClassResults(evaluationResults)
            }

            {/* Predictions Accordion */}
            {evaluationResults.detailed_predictions?.length > 0 && (
                <Accordion 
                    expanded={predictionsExpanded} 
                    onChange={() => setPredictionsExpanded(!predictionsExpanded)}
                    elevation={0}
                    sx={{ border: '1px solid #e0e0e0', '&:before': { display: 'none' } }}
                >
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                         <Typography variant="h6" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
                            <TableChartIcon color="primary" /> Predictions Viewer 
                            <Chip size="small" label={`${evaluationResults.detailed_predictions.length} items`} />
                         </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        {!evaluationResults.is_single_class && (
                            <Box sx={{ mb: 2 }}>
                                <FormControl fullWidth size="small">
                                    <InputLabel>Filter Columns by Class</InputLabel>
                                    <Select
                                        multiple
                                        value={selectedPredictionClasses}
                                        label="Filter Columns by Class"
                                        onChange={(e) => setSelectedPredictionClasses(typeof e.target.value === 'string' ? e.target.value.split(',') : e.target.value)}
                                        renderValue={(selected) => selected.join(', ')}
                                    >
                                        {evaluationResults.class_names.map((name) => (
                                            <MenuItem key={name} value={name}>{name}</MenuItem>
                                        ))}
                                    </Select>
                                </FormControl>
                            </Box>
                        )}
                        <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 500 }}>
                            <Table stickyHeader size="small">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Sample Name</TableCell>
                                        {evaluationResults.is_single_class ? (
                                            <>
                                                <TableCell align="center">Pred</TableCell>
                                                <TableCell align="center">Label</TableCell>
                                            </>
                                        ) : (
                                            selectedPredictionClasses.map(cls => (
                                                <React.Fragment key={cls}>
                                                    <TableCell align="center" sx={{ bgcolor: '#e3f2fd' }}>{cls} (P)</TableCell>
                                                    <TableCell align="center" sx={{ bgcolor: '#fbe9e7' }}>{cls} (L)</TableCell>
                                                </React.Fragment>
                                            ))
                                        )}
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {evaluationResults.detailed_predictions.map((item, idx) => (
                                        <TableRow key={idx} hover>
                                            <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>{item.file_name}</TableCell>
                                            {evaluationResults.is_single_class ? (
                                                <>
                                                    <TableCell align="center" sx={{ color: item.predictions[0] > 0.5 ? 'error.main' : 'text.secondary' }}>
                                                        {item.predictions[0].toFixed(3)}
                                                    </TableCell>
                                                    <TableCell align="center" sx={{ fontWeight: 'bold', color: item.labels[0] === 1 ? 'error.main' : 'text.disabled' }}>
                                                        {item.labels[0]}
                                                    </TableCell>
                                                </>
                                            ) : (
                                                selectedPredictionClasses.map(cls => {
                                                    const clsIdx = evaluationResults.class_names.indexOf(cls);
                                                    const pred = clsIdx >= 0 ? item.predictions[clsIdx] : 0;
                                                    const label = clsIdx >= 0 ? item.labels[clsIdx] : 0;
                                                    return (
                                                        <React.Fragment key={cls}>
                                                            <TableCell align="center" sx={{ color: pred > 0.5 ? 'error.main' : 'text.secondary' }}>
                                                                {pred.toFixed(3)}
                                                            </TableCell>
                                                            <TableCell align="center" sx={{ fontWeight: 'bold', color: label === 1 ? 'error.main' : 'text.disabled' }}>
                                                                {label}
                                                            </TableCell>
                                                        </React.Fragment>
                                                    );
                                                })
                                            )}
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </AccordionDetails>
                </Accordion>
            )}

            {/* Export Card */}
            <Card variant="outlined">
                <CardHeader title="Export Results" subheader="Download evaluation data as CSV" />
                <CardContent>
                    <Stack direction="row" spacing={2}>
                        <Button variant="outlined" startIcon={<DownloadIcon />} onClick={exportMetricsCSV}>
                            Export Metrics
                        </Button>
                        <Button variant="outlined" startIcon={<DownloadIcon />} onClick={exportPredictionsCSV}>
                            Export Predictions
                        </Button>
                    </Stack>
                </CardContent>
            </Card>
        </Stack>
      )}

      {/* Instructions */}
      {!evaluationResults && (
          <Alert severity="info" sx={{ mt: 4 }} icon={<HelpIcon />}>
              <Typography variant="subtitle2" gutterBottom>How to Evaluate</Typography>
              <Typography variant="body2" paragraph>
                1. <strong>Load Dataset:</strong> Must be a dataset created with "Evaluation Dataset" checked in Dataset Builder.
              </Typography>
              <Typography variant="body2" paragraph>
                2. <strong>Load Classifier:</strong> Load your trained <code>.keras</code> model.
              </Typography>
              <Typography variant="body2">
                3. <strong>Run:</strong> Click "Run Evaluation" to calculate metrics like AUC, AP, and Confusion Matrices.
              </Typography>
          </Alert>
      )}
    </Box>
  );
};

export default Evaluation;
