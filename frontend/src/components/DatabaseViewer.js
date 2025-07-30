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

  const pageSizeOptions = [
    { value: 25, label: '25 rows' },
    { value: 50, label: '50 rows' },
    { value: 100, label: '100 rows' },
    { value: 200, label: '200 rows' }
  ];

  const loadDatabaseInfo = async () => {
    try {
      const response = await axios.get('/api/database/info');
      if (response.data.status === 'success') {
        const info = response.data.info;
        setDatabaseInfo(info);
        
        // Set up column options
        const columns = info.columns.map(col => ({
          value: col,
          label: `${col} (${info.schema[col]})`
        }));
        setAvailableColumns(columns);
        
        // Load initial data
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

  const loadData = async (page = 0) => {
    if (!databaseInfo) return;
    
    setLoading(true);
    try {
      const offset = page * pageSize;
      const columnsParam = selectedColumns.length > 0 
        ? selectedColumns.map(col => col.value).join(',') 
        : null;
      
      const params = {
        limit: pageSize,
        offset: offset,
        ...(columnsParam && { columns: columnsParam }),
        ...(filterColumn && filterValue && { 
          filter_column: filterColumn, 
          filter_value: filterValue 
        })
      };
      
      const response = await axios.get('/api/database/data', { params });
      
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
  }, []);

  useEffect(() => {
    if (selectedStatsColumn) {
      loadColumnStats(selectedStatsColumn);
    }
  }, [selectedStatsColumn]);

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
          </div>
        </div>

        {/* Controls */}
        <div className="grid grid-2" style={{ marginBottom: '1.5rem' }}>
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
                {databaseInfo.columns.map(col => (
                  <option key={col} value={col}>{col}</option>
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
              {databaseInfo.columns.map(col => (
                <option key={col} value={col}>{col}</option>
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
            <li><strong>Database Info:</strong> View basic information about the loaded database</li>
            <li><strong>Column Selection:</strong> Choose specific columns to display or leave empty for all</li>
            <li><strong>Filtering:</strong> Filter rows by column values (supports text search and exact matches)</li>
            <li><strong>Pagination:</strong> Navigate through large datasets with customizable page sizes</li>
            <li><strong>Column Statistics:</strong> Get detailed statistics for any column</li>
          </ol>
          <p><strong>Note:</strong> This viewer shows the current state of the database including all annotations and multiclass data. List columns (predictions, annotation_status, label_strength) are displayed as string representations.</p>
        </div>
      </div>
    </div>
  );
};

export default DatabaseViewer;