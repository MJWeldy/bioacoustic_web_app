import React, { useState, useEffect } from 'react';
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
  Pagination,
  LinearProgress,
} from '@mui/material';
import {
  Storage as DatabaseIcon,
  TableChart as TableIcon,
  FilterList as FilterIcon,
  ExpandMore as ExpandMoreIcon,
  Info as InfoIcon,
  BarChart as StatsIcon,
  BarChart,
  Search as SearchIcon,
  Clear as ClearIcon,
  Refresh as RefreshIcon,
  Help as HelpIcon,
  Schema as SchemaIcon,
} from '@mui/icons-material';

const DatabaseViewer = ({ isActive = true }) => {
  const [databaseInfo, setDatabaseInfo] = useState(null);
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(0); // 0-based for API
  const [pageSize, setPageSize] = useState(50);
  const [totalRows, setTotalRows] = useState(0);
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [availableColumns, setAvailableColumns] = useState([]);
  const [filterColumn, setFilterColumn] = useState('');
  const [filterValue, setFilterValue] = useState('');
  const [columnStats, setColumnStats] = useState(null);
  const [selectedStatsColumn, setSelectedStatsColumn] = useState('');
  const [annotationSummary, setAnnotationSummary] = useState(null);
  const [reviewClips, setReviewClips] = useState([]);
  const [files, setFiles] = useState([]);
  const [selectedTable, setSelectedTable] = useState('clips');
  const [expandedSection, setExpandedSection] = useState('data'); // 'data', 'stats', 'schema'

  // Multiclass score view state
  const [showMulticlassView, setShowMulticlassView] = useState(false);
  const [multiclassData, setMulticlassData] = useState([]);
  const [multiclassLoading, setMulticlassLoading] = useState(false);
  const [selectedClasses, setSelectedClasses] = useState([]);
  const [multiclassPage, setMulticlassPage] = useState(0);
  const [multiclassTotalRows, setMulticlassTotalRows] = useState(0);
  const [scoreFilterMin, setScoreFilterMin] = useState('');
  const [scoreFilterMax, setScoreFilterMax] = useState('');
  const [scoreFilterClass, setScoreFilterClass] = useState('');

  const pageSizeOptions = [25, 50, 100, 200];

  const tableOptions = [
    { value: 'files', label: 'Files Table' },
    { value: 'clips', label: 'Clips Table' },
    { value: 'annotations', label: 'Annotations Table' },
    { value: 'clips_with_files', label: 'Clips with File Info (Joined)' }
  ];

  // --- API Handlers ---
  const loadDatabaseInfo = async () => {
    try {
      const response = await axios.get('/api/database/info');
      if (response.data.status === 'success') {
        setDatabaseInfo(response.data.info);
        await loadTableInfo();
        await loadData();
      }
    } catch (error) {
      if (error.response?.status === 400) {
        setDatabaseInfo(null);
        setData([]);
        setAvailableColumns([]);
      } else {
        toast.error('Failed to load database information');
        console.error('Database info error:', error);
      }
    }
  };

  const loadTableInfo = async () => {
    try {
      const response = await axios.get('/api/database/table-info', {
        params: { table: selectedTable }
      });
      if (response.data.status === 'success') {
        const info = response.data.info;
        const columns = info.columns.map(col => ({
          value: col,
          label: `${col} (${info.schema[col]})`
        }));
        setAvailableColumns(columns);
      }
    } catch (error) {
      console.error('Failed to load table info:', error);
    }
  };

  const loadData = async (page = 0) => {
    if (!databaseInfo) return;
    
    setLoading(true);
    try {
      const offset = page * pageSize;
      const columnsParam = selectedColumns.length > 0 
        ? selectedColumns.join(',') 
        : null;
      
      const params = {
        table: selectedTable,
        limit: pageSize,
        offset: offset,
        ...(columnsParam && { columns: columnsParam }),
        ...(filterColumn && filterValue && { 
          filter_column: filterColumn, 
          filter_value: filterValue 
        })
      };
      
      const response = await axios.get('/api/database/table-data', { params });
      
      if (response.data.status === 'success') {
        setData(response.data.data);
        setTotalRows(response.data.total_rows);
        setCurrentPage(page);
      }
    } catch (error) {
      toast.error('Failed to load database data');
    } finally {
      setLoading(false);
    }
  };

  const loadColumnStats = async (column) => {
    if (!column) return;
    try {
      const response = await axios.get('/api/database/column-stats', {
        params: { column }
      });
      if (response.data.status === 'success') {
        setColumnStats(response.data.statistics);
      }
    } catch (error) {
      toast.error('Failed to load column statistics');
    }
  };

  const loadAnnotationSummary = async () => {
    try {
      const response = await axios.get('/api/database/annotation-summary');
      if (response.data.status === 'success') {
        setAnnotationSummary(response.data.summary);
      }
    } catch (error) {
      console.error('Failed to load annotation summary:', error);
    }
  };

  const loadReviewClips = async () => {
    try {
      const response = await axios.get('/api/database/review-clips');
      if (response.data.status === 'success') {
        setReviewClips(response.data.clips.slice(0, 10));
      }
    } catch (error) {
      console.error('Failed to load review clips:', error);
    }
  };

  const loadFiles = async () => {
    try {
      const response = await axios.get('/api/database/files');
      if (response.data.status === 'success') {
        setFiles(response.data.files);
      }
    } catch (error) {
      console.error('Failed to load files:', error);
    }
  };

  const loadMulticlassScores = async (page = 0) => {
    if (!databaseInfo || selectedClasses.length === 0) {
      toast.error('Please select at least one class');
      return;
    }

    setMulticlassLoading(true);
    try {
      const offset = page * pageSize;
      const classIndices = selectedClasses.join(',');

      const params = {
        class_indices: classIndices,
        limit: pageSize,
        offset: offset,
        ...(scoreFilterMin && { min_score: parseFloat(scoreFilterMin) }),
        ...(scoreFilterMax && { max_score: parseFloat(scoreFilterMax) }),
        ...(scoreFilterClass && { filter_class_index: parseInt(scoreFilterClass) })
      };

      const response = await axios.get('/api/database/multiclass-scores', { params });

      if (response.data.status === 'success') {
        setMulticlassData(response.data.data);
        setMulticlassTotalRows(response.data.total_rows);
        setMulticlassPage(page);
      }
    } catch (error) {
      toast.error('Failed to load multiclass scores');
      console.error('Multiclass scores error:', error);
    } finally {
      setMulticlassLoading(false);
    }
  };

  // --- Event Handlers ---
  const handleFilter = async () => {
    if (filterColumn && filterValue) {
      await loadData(0);
    }
  };

  const clearFilter = async () => {
    setFilterColumn('');
    setFilterValue('');
    await loadData(0);
  };

  const handlePageChange = (event, newPage) => {
    // Pagination component is 1-based, API is 0-based
    loadData(newPage - 1);
  };

  const handlePageSizeChange = (event) => {
    setPageSize(event.target.value);
    loadData(0); // Reset to first page
  };

  const handleMulticlassPageChange = (event, newPage) => {
    loadMulticlassScores(newPage - 1);
  };

  const handleApplyScoreFilter = () => {
    loadMulticlassScores(0);
  };

  const handleClearScoreFilter = () => {
    setScoreFilterMin('');
    setScoreFilterMax('');
    setScoreFilterClass('');
    loadMulticlassScores(0);
  };

  // --- Effects ---
  useEffect(() => {
    loadDatabaseInfo();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (databaseInfo?.new_structure) {
      loadAnnotationSummary();
      loadReviewClips();
      loadFiles();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [databaseInfo]);

  useEffect(() => {
    if (selectedStatsColumn) {
      loadColumnStats(selectedStatsColumn);
    }
  }, [selectedStatsColumn]);

  useEffect(() => {
    if (databaseInfo) {
      loadTableInfo();
      loadData(0);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTable]);

  if (!databaseInfo) {
    return (
      <Box sx={{ display: isActive ? 'flex' : 'none', justifyContent: 'center', alignItems: 'center', height: '50vh', flexDirection: 'column', gap: 2 }}>
        <Typography variant="h5" color="text.secondary">No Active Database</Typography>
        <Typography variant="body1" color="text.secondary">Please load a dataset in the Active Learning tab first.</Typography>
        <Button variant="contained" startIcon={<RefreshIcon />} onClick={loadDatabaseInfo}>
            Check for Dataset
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ display: isActive ? 'block' : 'none', pb: 4 }}>
      {/* 1. Header & DB Info */}
      <Card elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2 }}>
        <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
                    <DatabaseIcon color="primary" /> Database Overview
                </Typography>
                <Chip 
                    label={`${databaseInfo.total_rows} Total Rows`} 
                    color="primary" 
                    variant="outlined" 
                    size="small" 
                />
            </Box>
            
            <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, bgcolor: '#f5f5f5' }} elevation={0}>
                        <Typography variant="caption" color="text.secondary">STRUCTURE</Typography>
                        <Stack spacing={0.5} mt={1}>
                            <Typography variant="body2"><strong>Classes:</strong> {databaseInfo.num_classes}</Typography>
                            <Typography variant="body2"><strong>Columns:</strong> {databaseInfo.columns.length}</Typography>
                        </Stack>
                    </Paper>
                </Grid>
                {databaseInfo.new_structure && (
                    <Grid item xs={12} md={4}>
                        <Paper sx={{ p: 2, bgcolor: '#e3f2fd' }} elevation={0}>
                            <Typography variant="caption" color="primary">ENTITIES</Typography>
                            <Stack spacing={0.5} mt={1}>
                                <Typography variant="body2"><strong>Files:</strong> {databaseInfo.new_structure.files_count}</Typography>
                                <Typography variant="body2"><strong>Clips:</strong> {databaseInfo.new_structure.clips_count}</Typography>
                                <Typography variant="body2"><strong>Annotations:</strong> {databaseInfo.new_structure.annotations_count}</Typography>
                            </Stack>
                        </Paper>
                    </Grid>
                )}
                {databaseInfo.class_map && (
                    <Grid item xs={12} md={4}>
                        <Paper sx={{ p: 2, bgcolor: '#fff3e0' }} elevation={0}>
                            <Typography variant="caption" color="warning.dark">CLASS MAP</Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
                                {Object.entries(databaseInfo.class_map).map(([name, val]) => (
                                    <Chip key={val} label={`${name}: ${val}`} size="small" variant="outlined" sx={{ bgcolor: 'white' }} />
                                ))}
                            </Box>
                        </Paper>
                    </Grid>
                )}
            </Grid>
        </CardContent>
      </Card>

      {/* 2. Main Data Viewer */}
      <Card elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2 }}>
        <CardHeader 
            title={
                <Stack direction="row" alignItems="center" spacing={2}>
                    <TableIcon />
                    <FormControl size="small" sx={{ minWidth: 250 }}>
                        <Select
                            value={selectedTable}
                            onChange={(e) => {
                                setSelectedTable(e.target.value);
                                setSelectedColumns([]);
                                setFilterColumn('');
                                setFilterValue('');
                            }}
                            variant="standard"
                            disableUnderline
                            sx={{ fontSize: '1.25rem', fontWeight: 600 }}
                        >
                            {tableOptions.map(opt => <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>)}
                        </Select>
                    </FormControl>
                </Stack>
            }
            action={
                <Button 
                    startIcon={<RefreshIcon />} 
                    onClick={() => loadData(0)} 
                    disabled={loading}
                >
                    Refresh
                </Button>
            }
        />
        <CardContent>
            {/* Toolbar: Columns & Filter */}
            <Accordion variant="outlined" sx={{ mb: 2 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <FilterIcon fontSize="small" /> Filter & Columns
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Grid container spacing={2} alignItems="center">
                        <Grid item xs={12} md={4}>
                            <FormControl fullWidth size="small">
                                <InputLabel>Visible Columns</InputLabel>
                                <Select
                                    multiple
                                    value={selectedColumns}
                                    label="Visible Columns"
                                    onChange={(e) => setSelectedColumns(typeof e.target.value === 'string' ? e.target.value.split(',') : e.target.value)}
                                    renderValue={(selected) => selected.length + ' selected'}
                                >
                                    {availableColumns.map((col) => (
                                        <MenuItem key={col.value} value={col.value}>{col.label}</MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={12} md={3}>
                            <FormControl fullWidth size="small">
                                <InputLabel>Filter Column</InputLabel>
                                <Select
                                    value={filterColumn}
                                    label="Filter Column"
                                    onChange={(e) => setFilterColumn(e.target.value)}
                                >
                                    <MenuItem value=""><em>None</em></MenuItem>
                                    {availableColumns.map((col) => (
                                        <MenuItem key={col.value} value={col.value}>{col.value}</MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={12} md={3}>
                            <TextField
                                fullWidth
                                size="small"
                                label="Filter Value"
                                value={filterValue}
                                onChange={(e) => setFilterValue(e.target.value)}
                                disabled={!filterColumn}
                            />
                        </Grid>
                        <Grid item xs={12} md={2}>
                            <Stack direction="row" spacing={1}>
                                <Button 
                                    variant="contained" 
                                    onClick={handleFilter} 
                                    disabled={!filterColumn || !filterValue}
                                    size="small"
                                >
                                    Apply
                                </Button>
                                <Button 
                                    variant="outlined" 
                                    onClick={clearFilter}
                                    disabled={!filterColumn && !filterValue}
                                    size="small"
                                >
                                    Clear
                                </Button>
                            </Stack>
                        </Grid>
                    </Grid>
                </AccordionDetails>
            </Accordion>

            {/* Data Table */}
            {loading && <LinearProgress sx={{ mb: 1 }} />}
            <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 600 }}>
                <Table stickyHeader size="small">
                    <TableHead>
                        <TableRow>
                            {data.length > 0 && Object.keys(data[0]).map((col) => (
                                <TableCell key={col} sx={{ bgcolor: '#f5f5f5', fontWeight: 'bold', whiteSpace: 'nowrap' }}>
                                    {col}
                                </TableCell>
                            ))}
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {data.map((row, idx) => (
                            <TableRow key={idx} hover>
                                {Object.values(row).map((val, cellIdx) => (
                                    <TableCell key={cellIdx} sx={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                        {val !== null && val !== undefined ? val.toString() : <Typography variant="caption" color="text.disabled">null</Typography>}
                                    </TableCell>
                                ))}
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>

            {/* Pagination Footer */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mt: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                        Total: {totalRows}
                    </Typography>
                    <FormControl size="small">
                        <Select value={pageSize} onChange={handlePageSizeChange} variant="standard" disableUnderline>
                            {pageSizeOptions.map(size => <MenuItem key={size} value={size}>{size} / page</MenuItem>)}
                        </Select>
                    </FormControl>
                </Box>
                <Pagination 
                    count={Math.ceil(totalRows / pageSize)} 
                    page={currentPage + 1} 
                    onChange={handlePageChange} 
                    color="primary" 
                    showFirstButton 
                    showLastButton 
                />
            </Stack>
        </CardContent>
      </Card>

      {/* 3. Multiclass Score View */}
      <Card elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2 }}>
        <CardHeader
          title={
            <Stack direction="row" alignItems="center" spacing={2}>
              <BarChart />
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                Multiclass Score Viewer
              </Typography>
            </Stack>
          }
          subheader="View and filter scores across multiple target classes"
          action={
            <Button
              variant={showMulticlassView ? "outlined" : "contained"}
              onClick={() => setShowMulticlassView(!showMulticlassView)}
            >
              {showMulticlassView ? "Hide" : "Show"}
            </Button>
          }
        />
        <Collapse in={showMulticlassView}>
          <CardContent>
            {/* Class Selection & Filters */}
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={12} md={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>Select Classes</InputLabel>
                  <Select
                    multiple
                    value={selectedClasses}
                    label="Select Classes"
                    onChange={(e) => setSelectedClasses(typeof e.target.value === 'string' ? e.target.value.split(',') : e.target.value)}
                    renderValue={(selected) => selected.length + ' selected'}
                  >
                    {databaseInfo?.class_map && Object.entries(databaseInfo.class_map).map(([name, idx]) => (
                      <MenuItem key={idx} value={idx}>{name}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={2}>
                <TextField
                  fullWidth
                  size="small"
                  type="number"
                  label="Min Score"
                  value={scoreFilterMin}
                  onChange={(e) => setScoreFilterMin(e.target.value)}
                  inputProps={{ step: 0.01, min: 0, max: 1 }}
                />
              </Grid>
              <Grid item xs={12} md={2}>
                <TextField
                  fullWidth
                  size="small"
                  type="number"
                  label="Max Score"
                  value={scoreFilterMax}
                  onChange={(e) => setScoreFilterMax(e.target.value)}
                  inputProps={{ step: 0.01, min: 0, max: 1 }}
                />
              </Grid>
              <Grid item xs={12} md={2}>
                <FormControl fullWidth size="small">
                  <InputLabel>Filter By Class</InputLabel>
                  <Select
                    value={scoreFilterClass}
                    label="Filter By Class"
                    onChange={(e) => setScoreFilterClass(e.target.value)}
                  >
                    <MenuItem value=""><em>Any Class</em></MenuItem>
                    {selectedClasses.map(classIdx => {
                      const className = databaseInfo?.class_map ? Object.keys(databaseInfo.class_map).find(k => databaseInfo.class_map[k] === classIdx) : `Class ${classIdx}`;
                      return <MenuItem key={classIdx} value={classIdx}>{className}</MenuItem>;
                    })}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={2}>
                <Stack direction="row" spacing={1}>
                  <Button
                    variant="contained"
                    fullWidth
                    onClick={handleApplyScoreFilter}
                    disabled={selectedClasses.length === 0 || multiclassLoading}
                    size="small"
                  >
                    Load
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={handleClearScoreFilter}
                    disabled={multiclassLoading}
                    size="small"
                  >
                    Clear
                  </Button>
                </Stack>
              </Grid>
            </Grid>

            {/* Data Table */}
            {multiclassLoading && <LinearProgress sx={{ mb: 1 }} />}
            <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 600 }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    {multiclassData.length > 0 && Object.keys(multiclassData[0]).map((col) => (
                      <TableCell key={col} sx={{ bgcolor: '#f5f5f5', fontWeight: 'bold', whiteSpace: 'nowrap' }}>
                        {col}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {multiclassData.length > 0 ? (
                    multiclassData.map((row, idx) => (
                      <TableRow key={idx} hover>
                        {Object.entries(row).map(([key, val], cellIdx) => (
                          <TableCell key={cellIdx} sx={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {key.startsWith('score_') ? (
                              <Chip
                                label={typeof val === 'number' ? val.toFixed(3) : val}
                                size="small"
                                color={val >= 0.8 ? 'success' : val >= 0.5 ? 'warning' : 'default'}
                                variant="outlined"
                              />
                            ) : (
                              val !== null && val !== undefined ? val.toString() : <Typography variant="caption" color="text.disabled">null</Typography>
                            )}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))
                  ) : (
                    <TableRow>
                      <TableCell colSpan={100} align="center">
                        <Typography variant="body2" color="text.secondary" sx={{ py: 4 }}>
                          {selectedClasses.length === 0 ? 'Select classes and click Load to view scores' : 'No data available'}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>

            {/* Pagination */}
            {multiclassData.length > 0 && (
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Total: {multiclassTotalRows} clips
                </Typography>
                <Pagination
                  count={Math.ceil(multiclassTotalRows / pageSize)}
                  page={multiclassPage + 1}
                  onChange={handleMulticlassPageChange}
                  color="primary"
                  showFirstButton
                  showLastButton
                />
              </Stack>
            )}
          </CardContent>
        </Collapse>
      </Card>

      {/* 4. Advanced Analysis (Schema & Stats) */}
      <Accordion elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2, '&:before': { display: 'none' } }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
                <SchemaIcon color="primary" /> Structural Analysis
            </Typography>
        </AccordionSummary>
        <AccordionDetails>
            <Grid container spacing={3}>
                {/* Column Stats */}
                <Grid item xs={12} md={6}>
                    <Card variant="outlined" sx={{ height: '100%' }}>
                        <CardHeader 
                            title="Column Statistics" 
                            avatar={<StatsIcon color="action" />}
                            subheader={
                                <FormControl fullWidth size="small" sx={{ mt: 1 }}>
                                    <InputLabel>Select Column</InputLabel>
                                    <Select
                                        value={selectedStatsColumn}
                                        label="Select Column"
                                        onChange={(e) => setSelectedStatsColumn(e.target.value)}
                                    >
                                        {availableColumns.map(col => <MenuItem key={col.value} value={col.value}>{col.value}</MenuItem>)}
                                    </Select>
                                </FormControl>
                            } 
                        />
                        <CardContent>
                            {columnStats ? (
                                <Stack spacing={1}>
                                    <Alert severity="info" sx={{ py: 0 }}>Type: {columnStats.data_type}</Alert>
                                    <Grid container spacing={1}>
                                        <Grid item xs={6}><Typography variant="body2"><strong>Count:</strong> {columnStats.total_count}</Typography></Grid>
                                        <Grid item xs={6}><Typography variant="body2"><strong>Nulls:</strong> {columnStats.null_count}</Typography></Grid>
                                        {columnStats.min !== undefined && (
                                            <>
                                                <Grid item xs={6}><Typography variant="body2"><strong>Min:</strong> {columnStats.min}</Typography></Grid>
                                                <Grid item xs={6}><Typography variant="body2"><strong>Max:</strong> {columnStats.max}</Typography></Grid>
                                            </>
                                        )}
                                    </Grid>
                                    {columnStats.top_values && (
                                        <Box mt={1}>
                                            <Typography variant="caption" fontWeight="bold">TOP VALUES</Typography>
                                            <Stack direction="row" spacing={0.5} flexWrap="wrap">
                                                {columnStats.top_values.slice(0, 5).map((v, i) => (
                                                    <Chip key={i} label={`${v.name} (${v.counts})`} size="small" variant="outlined" />
                                                ))}
                                            </Stack>
                                        </Box>
                                    )}
                                </Stack>
                            ) : (
                                <Typography variant="body2" color="text.secondary" align="center">Select a column to view stats</Typography>
                            )}
                        </CardContent>
                    </Card>
                </Grid>

                {/* Annotation Summary */}
                {annotationSummary && (
                    <Grid item xs={12} md={6}>
                        <Card variant="outlined" sx={{ height: '100%' }}>
                            <CardHeader title="Annotation Distribution" avatar={<InfoIcon color="action" />} />
                            <CardContent>
                                <Stack spacing={2}>
                                    <Box>
                                        <Typography variant="caption" color="text.secondary">PROGRESS</Typography>
                                        <Typography variant="body2">
                                            <strong>Annotated Clips:</strong> {annotationSummary.database_structure.annotated_clips} / {annotationSummary.database_structure.total_clips}
                                        </Typography>
                                    </Box>
                                    {Object.entries(annotationSummary.annotations_by_class).map(([cls, stats]) => (
                                        <Box key={cls}>
                                            <Typography variant="subtitle2" gutterBottom>{cls}</Typography>
                                            <Stack direction="row" spacing={1}>
                                                <Chip label={`Present: ${stats.present}`} size="small" color="success" variant="soft" />
                                                <Chip label={`Absent: ${stats.not_present}`} size="small" color="error" variant="soft" />
                                                <Chip label={`Unsure: ${stats.uncertain}`} size="small" color="warning" variant="soft" />
                                            </Stack>
                                        </Box>
                                    ))}
                                </Stack>
                            </CardContent>
                        </Card>
                    </Grid>
                )}
            </Grid>
        </AccordionDetails>
      </Accordion>

      {/* 4. Help Section */}
      <Accordion elevation={0} sx={{ border: '1px solid #e0e0e0', '&:before': { display: 'none' } }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <HelpIcon fontSize="small" /> Guide: Understanding the Database
            </Typography>
        </AccordionSummary>
        <AccordionDetails>
            <Typography variant="body2" paragraph>
                This viewer allows you to inspect the raw SQLite database generated by the application.
            </Typography>
            <Typography variant="body2" component="div">
                <ul>
                    <li><strong>Files Table:</strong> Metadata about original audio files (path, duration, sample rate).</li>
                    <li><strong>Clips Table:</strong> Individual 5-second segments extracted from files. Contains scores and time offsets.</li>
                    <li><strong>Annotations Table:</strong> User labels linked to clips.</li>
                    <li><strong>Joined View:</strong> Useful for exporting data, combining clip scores with original filenames.</li>
                </ul>
            </Typography>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default DatabaseViewer;
