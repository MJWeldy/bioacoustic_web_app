import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Button,
  Grid,
  Typography,
  Stack,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Checkbox,
  LinearProgress,
  ToggleButton,
  ToggleButtonGroup,
  Pagination,
} from '@mui/material';
import {
  Dashboard as OverviewIcon,
  ViewList as StrataIcon,
  TableRows as AnnotationsIcon,
  GetApp as ExportIcon,
  Refresh as RefreshIcon,
  CheckCircle as ConfirmIcon,
  Cancel as RejectIcon,
  Help as UncertainIcon,
  SkipNext as SkipIcon,
} from '@mui/icons-material';

const ValidationViewer = ({ isActive = true }) => {
  // Data states
  const [validationSummary, setValidationSummary] = useState(null);
  const [strataProgress, setStrataProgress] = useState([]);
  const [detailedAnnotations, setDetailedAnnotations] = useState([]);

  // View controls
  const [selectedView, setSelectedView] = useState('overview');
  const [selectedStrata, setSelectedStrata] = useState('all');
  const [selectedSpecies, setSelectedSpecies] = useState('all');
  const [showCompleted, setShowCompleted] = useState(true);
  const [showIncomplete, setShowIncomplete] = useState(true);

  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(20);
  const [sortField] = useState('validated_at');
  const [sortDirection] = useState('desc');

  // Loading state
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => { loadValidationData(); }, []);

  const loadValidationData = async () => {
    setIsLoading(true);
    try {
      const summaryResponse = await axios.get('/api/validation/summary');
      setValidationSummary(summaryResponse.data);
    } catch (error) { console.error('Failed to load summary:', error); }

    try {
      const progressResponse = await axios.get('/api/validation/strata-progress');
      setStrataProgress(progressResponse.data.strata_progress || []);
    } catch (error) { toast.error('Failed to load data'); }
    setIsLoading(false);
  };

  const loadDetailedAnnotations = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get('/api/validation/annotations', {
        params: {
          strata_id: selectedStrata !== 'all' ? selectedStrata : null,
          species_name: selectedSpecies !== 'all' ? selectedSpecies : null,
          page: currentPage,
          limit: itemsPerPage,
          sort_field: sortField,
          sort_direction: sortDirection
        }
      });
      setDetailedAnnotations(response.data.annotations || []);
    } catch (error) { toast.error('Failed to load annotations'); } 
    finally { setIsLoading(false); }
  };

  useEffect(() => {
    if (selectedView === 'annotations') loadDetailedAnnotations();
  }, [selectedView, selectedStrata, selectedSpecies, currentPage]);

  const exportResults = async (format = 'csv') => {
    // Export strata summary when on strata view
    if (selectedView === 'strata') {
      exportStrataData(format);
      return;
    }

    // For overview, export strata data as well (summary view)
    if (selectedView === 'overview') {
      exportStrataData(format);
      return;
    }

    // Export detailed annotations for annotations view
    try {
      const response = await axios.get(`/api/validation/export/${format}`, {
        params: {
          strata_id: selectedStrata !== 'all' ? selectedStrata : null,
          species_name: selectedSpecies !== 'all' ? selectedSpecies : null
        },
        responseType: 'blob'
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `validation_results.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      toast.success(`Exported as ${format.toUpperCase()}`);
    } catch (error) { toast.error('Export failed'); }
  };

  const exportStrataData = (format = 'csv') => {
    if (format === 'csv') {
      // Create CSV from strata progress data
      const headers = ['Strata', 'Species', 'Status', 'Confirmed', 'Rejected', 'Uncertain', 'Skipped', 'Validated', 'Total'];
      const rows = filteredStrataProgress.map(row => [
        row.strata_name,
        row.species_name,
        row.completion_status,
        row.confirmed_clips,
        row.rejected_clips,
        row.uncertain_clips || 0,
        row.skipped_clips || 0,
        row.validated_clips,
        row.total_clips
      ]);

      const csvContent = [
        headers.join(','),
        ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
      ].join('\n');

      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'strata_summary.csv');
      document.body.appendChild(link);
      link.click();
      link.remove();
      toast.success('Exported strata summary as CSV');
    } else if (format === 'json') {
      // Export as JSON
      const jsonData = filteredStrataProgress.map(row => ({
        strata_name: row.strata_name,
        species_name: row.species_name,
        completion_status: row.completion_status,
        confirmed_clips: row.confirmed_clips,
        rejected_clips: row.rejected_clips,
        uncertain_clips: row.uncertain_clips || 0,
        skipped_clips: row.skipped_clips || 0,
        validated_clips: row.validated_clips,
        total_clips: row.total_clips
      }));

      const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'strata_summary.json');
      document.body.appendChild(link);
      link.click();
      link.remove();
      toast.success('Exported strata summary as JSON');
    }
  };

  // Helper functions
  const uniqueStrata = [...new Map(strataProgress.map(item => [item.strata_id, { id: item.strata_id, name: item.strata_name }])).values()].sort((a,b) => a.name.localeCompare(b.name));
  const uniqueSpecies = [...new Set(strataProgress.map(item => item.species_name))].sort();

  const filteredStrataProgress = strataProgress.filter(item => {
      if (!showCompleted && item.completion_status === 'completed') return false;
      if (!showIncomplete && item.completion_status !== 'completed') return false;
      if (selectedStrata !== 'all' && item.strata_id !== selectedStrata) return false;
      if (selectedSpecies !== 'all' && item.species_name !== selectedSpecies) return false;
      return true;
  });

  return (
    <Box sx={{ display: isActive ? 'block' : 'none', pb: 4 }}>
      
      {/* Controls Card */}
      <Card elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2 }}>
        <CardContent>
            <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
                <ToggleButtonGroup 
                    value={selectedView} 
                    exclusive 
                    onChange={(_, v) => v && setSelectedView(v)} 
                    size="small"
                    color="primary"
                >
                    <ToggleButton value="overview"><OverviewIcon sx={{ mr: 1 }} /> Overview</ToggleButton>
                    <ToggleButton value="strata"><StrataIcon sx={{ mr: 1 }} /> Strata</ToggleButton>
                    <ToggleButton value="annotations"><AnnotationsIcon sx={{ mr: 1 }} /> Annotations</ToggleButton>
                </ToggleButtonGroup>

                <Stack direction="row" spacing={1}>
                    <Button startIcon={<RefreshIcon />} onClick={loadValidationData} variant="outlined" size="small">Refresh</Button>
                    <Button startIcon={<ExportIcon />} onClick={() => exportResults('csv')} variant="outlined" size="small">CSV</Button>
                    <Button startIcon={<ExportIcon />} onClick={() => exportResults('json')} variant="outlined" size="small">JSON</Button>
                </Stack>
            </Stack>

            <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} md={4}>
                    <FormControl fullWidth size="small">
                        <InputLabel>Strata</InputLabel>
                        <Select value={selectedStrata} label="Strata" onChange={(e) => setSelectedStrata(e.target.value)}>
                            <MenuItem value="all">All Strata</MenuItem>
                            {uniqueStrata.map(s => <MenuItem key={s.id} value={s.id}>{s.name}</MenuItem>)}
                        </Select>
                    </FormControl>
                </Grid>
                <Grid item xs={12} md={4}>
                    <FormControl fullWidth size="small">
                        <InputLabel>Species</InputLabel>
                        <Select value={selectedSpecies} label="Species" onChange={(e) => setSelectedSpecies(e.target.value)}>
                            <MenuItem value="all">All Species</MenuItem>
                            {uniqueSpecies.map(s => <MenuItem key={s} value={s}>{s}</MenuItem>)}
                        </Select>
                    </FormControl>
                </Grid>
                <Grid item xs={12} md={4}>
                    <Stack direction="row" spacing={2}>
                        <FormControlLabel control={<Checkbox checked={showCompleted} onChange={(e) => setShowCompleted(e.target.checked)} size="small" />} label={<Typography variant="body2">Completed</Typography>} />
                        <FormControlLabel control={<Checkbox checked={showIncomplete} onChange={(e) => setShowIncomplete(e.target.checked)} size="small" />} label={<Typography variant="body2">In Progress</Typography>} />
                    </Stack>
                </Grid>
            </Grid>
        </CardContent>
      </Card>

      {/* OVERVIEW VIEW */}
      {selectedView === 'overview' && validationSummary && (
        <Grid container spacing={2}>
            {/* Stats Cards */}
            {[
                { label: 'Total Predictions', value: validationSummary.total_predictions, color: 'primary.main' },
                { label: 'Validated', value: validationSummary.total_annotations, color: 'info.main' },
                { label: 'Confirmed', value: validationSummary.confirmed_count, color: 'success.main' },
                { label: 'Rejected', value: validationSummary.rejected_count, color: 'error.main' },
            ].map((stat, i) => (
                <Grid item xs={6} md={3} key={i}>
                    <Card variant="outlined">
                        <CardContent sx={{ textAlign: 'center', py: 2 }}>
                            <Typography variant="h4" color={stat.color} fontWeight="bold">{stat.value}</Typography>
                            <Typography variant="caption" color="text.secondary">{stat.label}</Typography>
                        </CardContent>
                    </Card>
                </Grid>
            ))}

            <Grid item xs={12}>
                <Card variant="outlined">
                    <CardHeader title="Overall Progress" />
                    <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Box sx={{ width: '100%', mr: 1 }}>
                                <LinearProgress variant="determinate" value={validationSummary.completion_percentage || 0} sx={{ height: 10, borderRadius: 5 }} />
                            </Box>
                            <Box sx={{ minWidth: 35 }}>
                                <Typography variant="body2" color="text.secondary">{`${Math.round(validationSummary.completion_percentage || 0)}%`}</Typography>
                            </Box>
                        </Box>
                    </CardContent>
                </Card>
            </Grid>
        </Grid>
      )}

      {/* STRATA VIEW */}
      {selectedView === 'strata' && (
        <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 600 }}>
            <Table stickyHeader size="small">
                <TableHead>
                    <TableRow>
                        <TableCell>Strata</TableCell>
                        <TableCell>Species</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Progress</TableCell>
                        <TableCell align="right">Confirmed</TableCell>
                        <TableCell align="right">Rejected</TableCell>
                        <TableCell align="right">Total</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {filteredStrataProgress.map((row, idx) => (
                        <TableRow key={idx} hover>
                            <TableCell>{row.strata_name}</TableCell>
                            <TableCell>{row.species_name}</TableCell>
                            <TableCell>
                                <Chip 
                                    label={row.completion_status} 
                                    size="small" 
                                    color={row.completion_status === 'completed' ? 'success' : 'default'} 
                                    variant="outlined" 
                                />
                            </TableCell>
                            <TableCell sx={{ width: 200 }}>
                                <LinearProgress variant="determinate" value={(row.validated_clips / row.total_clips) * 100} sx={{ height: 6, borderRadius: 3 }} />
                            </TableCell>
                            <TableCell align="right" sx={{ color: 'success.main', fontWeight: 'bold' }}>{row.confirmed_clips}</TableCell>
                            <TableCell align="right" sx={{ color: 'error.main', fontWeight: 'bold' }}>{row.rejected_clips}</TableCell>
                            <TableCell align="right">{row.total_clips}</TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
      )}

      {/* ANNOTATIONS VIEW */}
      {selectedView === 'annotations' && (
        <>
            <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 600 }}>
                <Table stickyHeader size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell>Filename</TableCell>
                            <TableCell>Time</TableCell>
                            <TableCell>Species</TableCell>
                            <TableCell>Confidence</TableCell>
                            <TableCell>Validation</TableCell>
                            <TableCell>Date</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {detailedAnnotations.map((ann, idx) => (
                            <TableRow key={idx} hover>
                                <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>{ann.filename}</TableCell>
                                <TableCell>{ann.start_time.toFixed(1)}s - {ann.end_time.toFixed(1)}s</TableCell>
                                <TableCell>{ann.species_name}</TableCell>
                                <TableCell>{(ann.original_confidence * 100).toFixed(1)}%</TableCell>
                                <TableCell>
                                    <Chip 
                                        icon={ann.validation_state === 'confirmed' ? <ConfirmIcon /> : ann.validation_state === 'rejected' ? <RejectIcon /> : ann.validation_state === 'uncertain' ? <UncertainIcon /> : <SkipIcon />}
                                        label={ann.validation_state} 
                                        size="small" 
                                        color={ann.validation_state === 'confirmed' ? 'success' : ann.validation_state === 'rejected' ? 'error' : 'default'} 
                                        variant="outlined"
                                    />
                                </TableCell>
                                <TableCell sx={{ fontSize: '0.75rem' }}>{new Date(ann.validated_at).toLocaleString()}</TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
                <Pagination count={10} page={currentPage} onChange={(_, p) => setCurrentPage(p)} color="primary" />
            </Box>
        </>
      )}
    </Box>
  );
};

export default ValidationViewer;
