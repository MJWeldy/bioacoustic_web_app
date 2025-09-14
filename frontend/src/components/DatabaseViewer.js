import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import Select from 'react-select';

const DatabaseViewer = () => {
  const [databaseInfo, setDatabaseInfo] = useState(null);
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(0);
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

  const pageSizeOptions = [
    { value: 25, label: '25 rows' },
    { value: 50, label: '50 rows' },
    { value: 100, label: '100 rows' },
    { value: 200, label: '200 rows' }
  ];

  const tableOptions = [
    { value: 'files', label: 'Files Table' },
    { value: 'clips', label: 'Clips Table' }, 
    { value: 'annotations', label: 'Annotations Table' },
    { value: 'clips_with_files', label: 'Clips with File Info (Joined)' }
  ];

  const loadDatabaseInfo = async () => {
    try {
      // Load overall database info first
      const response = await axios.get('/api/database/info');
      if (response.data.status === 'success') {
        setDatabaseInfo(response.data.info);
        
        // Load initial table info and data
        await loadTableInfo();
        await loadData();
      }
    } catch (error) {
      if (error.response?.status === 400) {
        // Don't show error toast immediately - just set state to show the message in UI
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
        // Set up column options for the selected table
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
        ? selectedColumns.map(col => col.value).join(',') 
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
        setReviewClips(response.data.clips.slice(0, 10)); // Show first 10 for demo
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

  const handleColumnSelection = (selectedOptions) => {
    setSelectedColumns(selectedOptions || []);
  };

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

  const handlePageSizeChange = async (option) => {
    setPageSize(option.value);
    await loadData(0);
  };

  const totalPages = Math.ceil(totalRows / pageSize);

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
      <div className="card">
        <div className="card-header">
          <h3>Database Viewer</h3>
          <p>No dataset loaded. Please load a dataset from the Active Learning tab first.</p>
        </div>
        <div style={{ padding: '20px', textAlign: 'center' }}>
          <p style={{ marginBottom: '15px', color: '#666' }}>
            The Database Viewer shows data from datasets loaded in the Active Learning tab.
          </p>
          <button
            onClick={loadDatabaseInfo}
            disabled={loading}
            className="btn btn-primary"
          >
            {loading ? 'Checking...' : 'Check for Dataset'}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <h3>Database Viewer</h3>
          <p>Explore and analyze the audio database structure and content</p>
        </div>

        {/* Database Information */}
        <div style={{ 
          padding: '1rem', 
          backgroundColor: '#d0eaf1', 
          borderRadius: '8px', 
          marginBottom: '1.5rem',
          border: '1px solid #7bbcd5'
        }}>
          <h4 style={{ margin: '0 0 0.5rem 0', color: '#6e7cb9' }}>Database Information</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            <div>
              <strong>Basic Stats:</strong>
              <ul style={{ margin: '0.25rem 0', paddingLeft: '20px' }}>
                <li>Total rows: {databaseInfo.total_rows}</li>
                <li>Columns: {databaseInfo.columns.length}</li>
                <li>Classes: {databaseInfo.num_classes}</li>
              </ul>
            </div>
            {databaseInfo.class_map && Object.keys(databaseInfo.class_map).length > 0 && (
              <div>
                <strong>Class Map:</strong>
                <ul style={{ margin: '0.25rem 0', paddingLeft: '20px' }}>
                  {Object.entries(databaseInfo.class_map).map(([name, value]) => (
                    <li key={value}>{name}: {value}</li>
                  ))}
                </ul>
              </div>
            )}
            {databaseInfo.new_structure && (
              <div>
                <strong>New Structure:</strong>
                <ul style={{ margin: '0.25rem 0', paddingLeft: '20px' }}>
                  <li>Files: {databaseInfo.new_structure.files_count}</li>
                  <li>Clips: {databaseInfo.new_structure.clips_count}</li>
                  <li>Annotations: {databaseInfo.new_structure.annotations_count}</li>
                </ul>
              </div>
            )}
          </div>
        </div>

        {/* Controls */}
        <div className="grid grid-3" style={{ marginBottom: '1.5rem' }}>
          <div className="form-group">
            <label htmlFor="tableSelect">Table to View</label>
            <Select
              id="tableSelect"
              options={tableOptions}
              value={tableOptions.find(opt => opt.value === selectedTable)}
              onChange={(selected) => {
                setSelectedTable(selected.value);
                // Reset column selection and filters when table changes
                setSelectedColumns([]);
                setFilterColumn('');
                setFilterValue('');
              }}
              isSearchable={false}
              styles={{
                control: (base) => ({
                  ...base,
                  border: '2px solid #6e7cb9',
                  '&:hover': { border: '2px solid #6e7cb9' },
                  '&:focus-within': { 
                    border: '2px solid #6e7cb9',
                    boxShadow: '0 0 0 3px rgba(110, 124, 185, 0.1)'
                  }
                })
              }}
            />
          </div>

          <div className="form-group">
            <label htmlFor="columnSelect">Select Columns (leave empty for all)</label>
            <Select
              id="columnSelect"
              isMulti
              options={availableColumns}
              value={selectedColumns}
              onChange={handleColumnSelection}
              placeholder="Select columns to display..."
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
          </div>

          <div className="form-group">
            <label htmlFor="pageSize">Page Size</label>
            <Select
              id="pageSize"
              options={pageSizeOptions}
              value={pageSizeOptions.find(opt => opt.value === pageSize)}
              onChange={handlePageSizeChange}
              isSearchable={false}
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
          </div>
        </div>

        {/* Filtering */}
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <div className="card-header">
            <h4>Filter Data</h4>
          </div>
          <div className="grid grid-3">
            <div className="form-group">
              <label htmlFor="filterColumn">Filter Column</label>
              <select
                id="filterColumn"
                className="form-control"
                value={filterColumn}
                onChange={(e) => setFilterColumn(e.target.value)}
              >
                <option value="">Select column...</option>
                {availableColumns.map(col => (
                  <option key={col.value} value={col.value}>{col.value}</option>
                ))}
              </select>
            </div>
            
            <div className="form-group">
              <label htmlFor="filterValue">Filter Value</label>
              <input
                type="text"
                id="filterValue"
                className="form-control"
                placeholder="Enter filter value..."
                value={filterValue}
                onChange={(e) => setFilterValue(e.target.value)}
              />
            </div>
            
            <div className="form-group" style={{ display: 'flex', alignItems: 'end', gap: '10px' }}>
              <button
                onClick={handleFilter}
                disabled={!filterColumn || !filterValue}
                className="btn btn-primary"
              >
                Apply Filter
              </button>
              <button
                onClick={clearFilter}
                className="btn btn-secondary"
              >
                Clear
              </button>
            </div>
          </div>
        </div>

        {/* Pagination */}
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          marginBottom: '1rem',
          padding: '0.5rem',
          backgroundColor: '#f8f9fa',
          borderRadius: '4px'
        }}>
          <div>
            Showing {currentPage * pageSize + 1} to {Math.min((currentPage + 1) * pageSize, totalRows)} of {totalRows} rows
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              onClick={() => loadData(Math.max(0, currentPage - 1))}
              disabled={currentPage === 0 || loading}
              className="btn btn-secondary btn-sm"
            >
              Previous
            </button>
            <span style={{ display: 'flex', alignItems: 'center', padding: '0 10px' }}>
              Page {currentPage + 1} of {totalPages}
            </span>
            <button
              onClick={() => loadData(Math.min(totalPages - 1, currentPage + 1))}
              disabled={currentPage >= totalPages - 1 || loading}
              className="btn btn-secondary btn-sm"
            >
              Next
            </button>
          </div>
        </div>

        {/* Data Table */}
        {loading ? (
          <div style={{ textAlign: 'center', padding: '2rem' }}>
            Loading data...
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ 
              width: '100%', 
              borderCollapse: 'collapse',
              fontSize: '0.875rem'
            }}>
              <thead>
                <tr style={{ backgroundColor: '#6e7cb9', color: 'white' }}>
                  {data.length > 0 && Object.keys(data[0]).map(column => (
                    <th key={column} style={{ 
                      padding: '8px', 
                      border: '1px solid #ddd',
                      textAlign: 'left',
                      position: 'sticky',
                      top: 0,
                      backgroundColor: '#6e7cb9'
                    }}>
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.map((row, index) => (
                  <tr key={index} style={{ 
                    backgroundColor: index % 2 === 0 ? '#f8f9fa' : 'white',
                    '&:hover': { backgroundColor: '#e9ecef' }
                  }}>
                    {Object.values(row).map((value, cellIndex) => (
                      <td key={cellIndex} style={{ 
                        padding: '6px 8px', 
                        border: '1px solid #ddd',
                        maxWidth: '200px',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                      }}>
                        {value !== null && value !== undefined ? value.toString() : 'null'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* New Three-Table Structure Queries */}
        {databaseInfo?.new_structure && annotationSummary && (
          <div className="card" style={{ marginTop: '2rem' }}>
            <div className="card-header">
              <h4>Three-Table Structure Query Examples</h4>
              <p>Demonstration of enhanced querying capabilities with the new Files → Clips → Annotations structure</p>
            </div>
            
            {/* Annotation Summary */}
            <div style={{ marginBottom: '1.5rem' }}>
              <h5 style={{ color: '#6e7cb9', marginBottom: '0.5rem' }}>Annotation Summary</h5>
              <div style={{ 
                padding: '1rem', 
                backgroundColor: '#f8f9fa', 
                borderRadius: '4px',
                border: '1px solid #e89c81'
              }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
                  <div>
                    <strong>Database Structure:</strong>
                    <ul style={{ margin: '0.25rem 0', paddingLeft: '20px' }}>
                      <li>Total Files: {annotationSummary.database_structure.total_files}</li>
                      <li>Total Clips: {annotationSummary.database_structure.total_clips}</li>
                      <li>Annotated Clips: {annotationSummary.database_structure.annotated_clips}</li>
                      <li>Unannotated Clips: {annotationSummary.database_structure.unannotated_clips}</li>
                    </ul>
                  </div>
                  {Object.keys(annotationSummary.annotations_by_class).length > 0 && (
                    <div>
                      <strong>Annotations by Class:</strong>
                      {Object.entries(annotationSummary.annotations_by_class).map(([className, stats]) => (
                        <div key={className} style={{ marginLeft: '10px', marginTop: '0.5rem' }}>
                          <strong>{className}:</strong>
                          <ul style={{ margin: '0.25rem 0', paddingLeft: '20px', fontSize: '0.875rem' }}>
                            <li>Present: {stats.present}</li>
                            <li>Not Present: {stats.not_present}</li>
                            <li>Uncertain: {stats.uncertain}</li>
                            <li>Total: {stats.total}</li>
                          </ul>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Files List Sample */}
            {files.length > 0 && (
              <div style={{ marginBottom: '1.5rem' }}>
                <h5 style={{ color: '#6e7cb9', marginBottom: '0.5rem' }}>Files Table Sample (Query: /api/database/files)</h5>
                <div style={{ 
                  padding: '1rem', 
                  backgroundColor: '#f8f9fa', 
                  borderRadius: '4px',
                  border: '1px solid #7bbcd5'
                }}>
                  <p style={{ fontSize: '0.875rem', color: '#666', marginBottom: '0.5rem' }}>
                    Shows {Math.min(5, files.length)} of {files.length} files in the database:
                  </p>
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', fontSize: '0.8rem', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ backgroundColor: '#7bbcd5', color: 'white' }}>
                          <th style={{ padding: '4px 8px', border: '1px solid #ddd' }}>File Name</th>
                          <th style={{ padding: '4px 8px', border: '1px solid #ddd' }}>Duration (s)</th>
                          <th style={{ padding: '4px 8px', border: '1px solid #ddd' }}>Sample Rate</th>
                          <th style={{ padding: '4px 8px', border: '1px solid #ddd' }}>Path</th>
                        </tr>
                      </thead>
                      <tbody>
                        {files.slice(0, 5).map((file, index) => (
                          <tr key={file.file_id} style={{ backgroundColor: index % 2 === 0 ? 'white' : '#f8f9fa' }}>
                            <td style={{ padding: '4px 8px', border: '1px solid #ddd' }}>{file.file_name}</td>
                            <td style={{ padding: '4px 8px', border: '1px solid #ddd' }}>{file.duration_sec?.toFixed(1)}</td>
                            <td style={{ padding: '4px 8px', border: '1px solid #ddd' }}>{file.sampling_rate}</td>
                            <td style={{ 
                              padding: '4px 8px', 
                              border: '1px solid #ddd',
                              maxWidth: '200px',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap'
                            }}>
                              {file.file_path}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* Review Clips Sample */}
            {reviewClips.length > 0 && (
              <div style={{ marginBottom: '1.5rem' }}>
                <h5 style={{ color: '#6e7cb9', marginBottom: '0.5rem' }}>Clips with Annotations Sample (Query: /api/database/review-clips)</h5>
                <div style={{ 
                  padding: '1rem', 
                  backgroundColor: '#f8f9fa', 
                  borderRadius: '4px',
                  border: '1px solid #d0eaf1'
                }}>
                  <p style={{ fontSize: '0.875rem', color: '#666', marginBottom: '0.5rem' }}>
                    Shows first {reviewClips.length} clips that contain at least one annotation:
                  </p>
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', fontSize: '0.8rem', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ backgroundColor: '#6e7cb9', color: 'white' }}>
                          <th style={{ padding: '4px 8px', border: '1px solid #ddd' }}>File</th>
                          <th style={{ padding: '4px 8px', border: '1px solid #ddd' }}>Clip Time</th>
                          <th style={{ padding: '4px 8px', border: '1px solid #ddd' }}>Duration</th>
                          <th style={{ padding: '4px 8px', border: '1px solid #ddd' }}>Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {reviewClips.map((clip, index) => (
                          <tr key={clip.clip_id} style={{ backgroundColor: index % 2 === 0 ? 'white' : '#f8f9fa' }}>
                            <td style={{ padding: '4px 8px', border: '1px solid #ddd' }}>{clip.file_name}</td>
                            <td style={{ padding: '4px 8px', border: '1px solid #ddd' }}>
                              {clip.clip_start?.toFixed(1)}s - {clip.clip_end?.toFixed(1)}s
                            </td>
                            <td style={{ padding: '4px 8px', border: '1px solid #ddd' }}>
                              {((clip.clip_end || 0) - (clip.clip_start || 0)).toFixed(1)}s
                            </td>
                            <td style={{ padding: '4px 8px', border: '1px solid #ddd' }}>
                              {clip.confidence_predictions ? 
                                clip.confidence_predictions[0]?.toFixed(3) || 'N/A' : 
                                'N/A'
                              }
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Column Statistics */}
        <div className="card" style={{ marginTop: '2rem' }}>
          <div className="card-header">
            <h4>Column Statistics</h4>
          </div>
          <div className="form-group">
            <label htmlFor="statsColumn">Select Column for Statistics</label>
            <select
              id="statsColumn"
              className="form-control"
              value={selectedStatsColumn}
              onChange={(e) => setSelectedStatsColumn(e.target.value)}
            >
              <option value="">Select column...</option>
              {availableColumns.map(col => (
                <option key={col.value} value={col.value}>{col.value}</option>
              ))}
            </select>
          </div>

          {columnStats && (
            <div style={{ 
              padding: '1rem', 
              backgroundColor: '#f8f9fa', 
              borderRadius: '4px',
              marginTop: '1rem'
            }}>
              <h5>Statistics for "{selectedStatsColumn}"</h5>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                <div>
                  <strong>Basic Info:</strong>
                  <ul style={{ margin: '0.25rem 0', paddingLeft: '20px' }}>
                    <li>Data Type: {columnStats.data_type}</li>
                    <li>Total Count: {columnStats.total_count}</li>
                    <li>Null Count: {columnStats.null_count}</li>
                  </ul>
                </div>
                
                {columnStats.min !== undefined && (
                  <div>
                    <strong>Numeric Stats:</strong>
                    <ul style={{ margin: '0.25rem 0', paddingLeft: '20px' }}>
                      <li>Min: {columnStats.min}</li>
                      <li>Max: {columnStats.max}</li>
                      <li>Mean: {columnStats.mean?.toFixed(4)}</li>
                      <li>Std Dev: {columnStats.std?.toFixed(4)}</li>
                    </ul>
                  </div>
                )}
                
                {columnStats.unique_count !== undefined && (
                  <div>
                    <strong>String Stats:</strong>
                    <ul style={{ margin: '0.25rem 0', paddingLeft: '20px' }}>
                      <li>Unique Values: {columnStats.unique_count}</li>
                    </ul>
                    {columnStats.top_values && (
                      <div style={{ marginTop: '0.5rem' }}>
                        <strong>Top Values:</strong>
                        <ul style={{ margin: '0.25rem 0', paddingLeft: '20px' }}>
                          {columnStats.top_values.slice(0, 5).map((item, index) => (
                            <li key={index}>{item.name}: {item.counts}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
                
                {columnStats.sample_values && (
                  <div>
                    <strong>Sample Values:</strong>
                    <ul style={{ margin: '0.25rem 0', paddingLeft: '20px' }}>
                      {columnStats.sample_values.map((val, index) => (
                        <li key={index}>{val}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>Instructions</h3>
        </div>
        <div style={{ lineHeight: '1.6' }}>
          <ol>
            <li><strong>Table Selection:</strong> Choose which table to view from the dropdown (Files, Clips, Annotations, or joined view)</li>
            <li><strong>Database Info:</strong> View basic information about the loaded database, including the three-table structure counts</li>
            <li><strong>Three-Table Queries:</strong> See demonstration queries showing Files → Clips → Annotations relationships</li>
            <li><strong>Column Selection:</strong> Choose specific columns to display or leave empty for all (updates based on selected table)</li>
            <li><strong>Filtering:</strong> Filter rows by column values (supports text search and exact matches)</li>
            <li><strong>Pagination:</strong> Navigate through large datasets with customizable page sizes</li>
            <li><strong>Column Statistics:</strong> Get detailed statistics for any column in the selected table</li>
          </ol>
          <p><strong>Three-Table Structure:</strong> The database uses a normalized structure where:</p>
          <ul>
            <li><strong>Files Table:</strong> Stores audio file metadata (filepath, duration, sample rate)</li>
            <li><strong>Clips Table:</strong> Stores clip segments with annotation status and confidence predictions</li>
            <li><strong>Annotations Table:</strong> Stores human labels (present/not_present/uncertain)</li>
            <li><strong>Clips with File Info:</strong> Joined view combining clips and file information</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default DatabaseViewer;