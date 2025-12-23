import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';

const ValidationViewer = () => {
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

  // Pagination and filtering
  const [currentPage] = useState(1);
  const [itemsPerPage] = useState(20);
  const [sortField] = useState('validated_at');
  const [sortDirection] = useState('desc');

  // Loading state
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadValidationData();
  }, []);

  const loadValidationData = async () => {
    setIsLoading(true);

    // Load overall summary (independent of strata progress)
    try {
      const summaryResponse = await axios.get('/api/validation/summary');
      setValidationSummary(summaryResponse.data);
    } catch (error) {
      console.error('Failed to load validation summary:', error);
      // Don't show error toast here, as strata progress might still work
    }

    // Load strata progress (independent of summary)
    try {
      const progressResponse = await axios.get('/api/validation/strata-progress');
      setStrataProgress(progressResponse.data.strata_progress || []);
    } catch (error) {
      console.error('Failed to load strata progress:', error);
      toast.error('Failed to load validation data');
    }

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
    } catch (error) {
      toast.error('Failed to load detailed annotations');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (selectedView === 'annotations') {
      loadDetailedAnnotations();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedView, selectedStrata, selectedSpecies, currentPage, sortField, sortDirection]);

  const exportResults = async (format = 'csv') => {
    try {
      const response = await axios.get(`/api/validation/export/${format}`, {
        params: {
          strata_id: selectedStrata !== 'all' ? selectedStrata : null,
          species_name: selectedSpecies !== 'all' ? selectedSpecies : null
        },
        responseType: 'blob'
      });

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `validation_results.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();

      toast.success(`Validation results exported as ${format.toUpperCase()}`);
    } catch (error) {
      toast.error('Failed to export validation results');
    }
  };

  const exportStrataProgress = (format = 'csv') => {
    try {
      // Use filtered data for export
      const dataToExport = filteredStrataProgress;

      if (dataToExport.length === 0) {
        toast.error('No data to export');
        return;
      }

      if (format === 'csv') {
        // Create CSV content
        const headers = [
          'Strata',
          'Species',
          'Min Confidence',
          'Status',
          'Total Clips',
          'Validated Clips',
          'Confirmed',
          'Rejected',
          'Uncertain',
          'Skipped',
          'Target Confirmations',
          'Progress %'
        ];

        const csvRows = [headers.join(',')];

        dataToExport.forEach(item => {
          const progressPercentage = item.total_clips > 0
            ? ((item.validated_clips / item.total_clips) * 100).toFixed(2)
            : '0.00';

          const row = [
            `"${item.strata_name}"`,
            `"${item.species_name}"`,
            item.confidence_threshold !== undefined && item.confidence_threshold !== null
              ? item.confidence_threshold.toFixed(2)
              : '0.00',
            `"${item.completion_status || 'incomplete'}"`,
            item.total_clips,
            item.validated_clips,
            item.confirmed_clips,
            item.rejected_clips,
            item.uncertain_clips,
            item.skipped_clips,
            item.target_confirmations,
            progressPercentage
          ];

          csvRows.push(row.join(','));
        });

        const csvContent = csvRows.join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `strata_progress_${new Date().toISOString().split('T')[0]}.csv`);
        document.body.appendChild(link);
        link.click();
        link.remove();

        toast.success('Strata progress exported as CSV');
      } else if (format === 'json') {
        // Export as JSON
        const jsonContent = JSON.stringify(dataToExport, null, 2);
        const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `strata_progress_${new Date().toISOString().split('T')[0]}.json`);
        document.body.appendChild(link);
        link.click();
        link.remove();

        toast.success('Strata progress exported as JSON');
      }
    } catch (error) {
      console.error('Export error:', error);
      toast.error('Failed to export strata progress');
    }
  };

  const getCompletionColor = (percentage) => {
    if (percentage >= 100) return '#059669';
    if (percentage >= 75) return '#0891b2';
    if (percentage >= 50) return '#fbbf24';
    if (percentage >= 25) return '#f97316';
    return '#ef4444';
  };

  const getStatusBadge = (status) => {
    const colors = {
      completed: { bg: '#dcfce7', color: '#166534', text: 'Complete' },
      target_met: { bg: '#dbeafe', color: '#1e40af', text: 'Target Met' },
      incomplete: { bg: '#fef3c7', color: '#92400e', text: 'In Progress' },
      not_started: { bg: '#f3f4f6', color: '#374151', text: 'Not Started' }
    };

    const style = colors[status] || colors.incomplete;
    return (
      <span style={{
        backgroundColor: style.bg,
        color: style.color,
        padding: '0.25rem 0.5rem',
        borderRadius: '12px',
        fontSize: '0.75rem',
        fontWeight: '600'
      }}>
        {style.text}
      </span>
    );
  };

  const getValidationBadge = (validationState) => {
    const colors = {
      confirmed: { bg: '#dcfce7', color: '#166534', text: 'Confirmed' },
      rejected: { bg: '#fee2e2', color: '#991b1b', text: 'Rejected' },
      uncertain: { bg: '#fef3c7', color: '#92400e', text: 'Uncertain' },
      skipped: { bg: '#f3f4f6', color: '#374151', text: 'Skipped' }
    };

    const style = colors[validationState] || { bg: '#f3f4f6', color: '#374151', text: validationState };
    return (
      <span style={{
        backgroundColor: style.bg,
        color: style.color,
        padding: '0.25rem 0.5rem',
        borderRadius: '12px',
        fontSize: '0.75rem',
        fontWeight: '600'
      }}>
        {style.text}
      </span>
    );
  };

  const filteredStrataProgress = strataProgress
    .filter(item => {
      if (!showCompleted && item.completion_status === 'completed') return false;
      if (!showIncomplete && item.completion_status !== 'completed') return false;
      if (selectedStrata !== 'all' && item.strata_id !== selectedStrata) return false;
      if (selectedSpecies !== 'all' && item.species_name !== selectedSpecies) return false;
      return true;
    })
    .sort((a, b) => {
      // Sort by strata name first, then by species name
      const strataCompare = a.strata_name.localeCompare(b.strata_name);
      if (strataCompare !== 0) return strataCompare;
      return a.species_name.localeCompare(b.species_name);
    });

  // Get unique strata (deduplicate by strata_id)
  const strataMap = new Map();
  strataProgress.forEach(item => {
    if (!strataMap.has(item.strata_id)) {
      strataMap.set(item.strata_id, { id: item.strata_id, name: item.strata_name });
    }
  });
  const uniqueStrata = Array.from(strataMap.values()).sort((a, b) => a.name.localeCompare(b.name));

  // Get unique species (sorted alphabetically)
  const uniqueSpecies = [...new Set(strataProgress.map(item => item.species_name))].sort();

  return (
    <div>
      {/* View Selection and Controls */}
      <div className="card">
        <div className="card-header">
          <h3>Validation Results Viewer</h3>
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginTop: '1rem' }}>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                onClick={() => setSelectedView('overview')}
                className={`btn ${selectedView === 'overview' ? 'btn-primary' : 'btn-outline'}`}
              >
                Overview
              </button>
              <button
                onClick={() => setSelectedView('strata')}
                className={`btn ${selectedView === 'strata' ? 'btn-primary' : 'btn-outline'}`}
              >
                Strata Progress
              </button>
              <button
                onClick={() => setSelectedView('annotations')}
                className={`btn ${selectedView === 'annotations' ? 'btn-primary' : 'btn-outline'}`}
              >
                Detailed Annotations
              </button>
            </div>

            <div style={{ marginLeft: 'auto', display: 'flex', gap: '0.5rem' }}>
              <button
                onClick={() => exportResults('csv')}
                className="btn btn-outline btn-sm"
                disabled={isLoading}
              >
                📊 Export CSV
              </button>
              <button
                onClick={() => exportResults('json')}
                className="btn btn-outline btn-sm"
                disabled={isLoading}
              >
                📄 Export JSON
              </button>
              <button
                onClick={loadValidationData}
                className="btn btn-outline btn-sm"
                disabled={isLoading}
              >
                🔄 Refresh
              </button>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="grid grid-4">
          <div className="form-group">
            <label htmlFor="strataFilter">Filter by Strata</label>
            <select
              id="strataFilter"
              className="form-control"
              value={selectedStrata}
              onChange={(e) => setSelectedStrata(e.target.value)}
            >
              <option value="all">All Strata</option>
              {uniqueStrata.map(strata => (
                <option key={strata.id} value={strata.id}>
                  {strata.name}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="speciesFilter">Filter by Species</label>
            <select
              id="speciesFilter"
              className="form-control"
              value={selectedSpecies}
              onChange={(e) => setSelectedSpecies(e.target.value)}
            >
              <option value="all">All Species</option>
              {uniqueSpecies.map(species => (
                <option key={species} value={species}>
                  {species}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Show Status</label>
            <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem' }}>
                <input
                  type="checkbox"
                  checked={showCompleted}
                  onChange={(e) => setShowCompleted(e.target.checked)}
                />
                Completed
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem' }}>
                <input
                  type="checkbox"
                  checked={showIncomplete}
                  onChange={(e) => setShowIncomplete(e.target.checked)}
                />
                In Progress
              </label>
            </div>
          </div>
        </div>
      </div>

      {/* Overview Section */}
      {selectedView === 'overview' && !validationSummary && (
        <div className="card">
          <div className="card-header">
            <h3>Validation Overview</h3>
          </div>
          <div style={{ padding: '2rem', textAlign: 'center', color: '#666' }}>
            <div style={{ fontSize: '1.2rem', marginBottom: '1rem' }}>
              <strong>No validation data loaded</strong>
            </div>
            <p>
              Please go to the <strong>Validation</strong> tab to load a validation dataset or create a new one from the <strong>Dataset Builder</strong> tab.
            </p>
          </div>
        </div>
      )}

      {selectedView === 'overview' && validationSummary && (
        <div className="card">
          <div className="card-header">
            <h3>Validation Overview</h3>
          </div>

          <div className="grid grid-4">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#6e7cb9' }}>
                {validationSummary.total_strata || 0}
              </div>
              <div style={{ fontSize: '0.875rem', color: '#666' }}>Total Strata</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#6e7cb9' }}>
                {validationSummary.total_species || 0}
              </div>
              <div style={{ fontSize: '0.875rem', color: '#666' }}>Species</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#6e7cb9' }}>
                {validationSummary.total_predictions || 0}
              </div>
              <div style={{ fontSize: '0.875rem', color: '#666' }}>Total Predictions</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#6e7cb9' }}>
                {validationSummary.total_annotations || 0}
              </div>
              <div style={{ fontSize: '0.875rem', color: '#666' }}>Validated</div>
            </div>
          </div>

          <div style={{ marginTop: '2rem' }}>
            <h4>Validation Summary by State</h4>
            <div className="grid grid-4">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#059669' }}>
                  {validationSummary.confirmed_count || 0}
                </div>
                <div style={{ fontSize: '0.875rem', color: '#666' }}>Confirmed</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#dc3545' }}>
                  {validationSummary.rejected_count || 0}
                </div>
                <div style={{ fontSize: '0.875rem', color: '#666' }}>Rejected</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#ffc107' }}>
                  {validationSummary.uncertain_count || 0}
                </div>
                <div style={{ fontSize: '0.875rem', color: '#666' }}>Uncertain</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#6c757d' }}>
                  {validationSummary.skipped_count || 0}
                </div>
                <div style={{ fontSize: '0.875rem', color: '#666' }}>Skipped</div>
              </div>
            </div>
          </div>

          {validationSummary.completion_percentage !== undefined && (
            <div style={{ marginTop: '2rem' }}>
              <h4>Overall Progress</h4>
              <div style={{
                backgroundColor: '#e9ecef',
                borderRadius: '4px',
                overflow: 'hidden',
                height: '24px',
                position: 'relative'
              }}>
                <div style={{
                  backgroundColor: getCompletionColor(validationSummary.completion_percentage),
                  height: '100%',
                  width: `${validationSummary.completion_percentage}%`,
                  transition: 'width 0.3s ease'
                }}></div>
                <div style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  fontWeight: '600',
                  color: validationSummary.completion_percentage > 50 ? 'white' : 'black'
                }}>
                  {validationSummary.completion_percentage.toFixed(1)}% Complete
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Strata Progress Section */}
      {selectedView === 'strata' && (
        <div className="card">
          <div className="card-header">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div>
                <h3>Strata Validation Progress</h3>
                <p>Progress tracking for each strata and species combination</p>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  onClick={() => exportStrataProgress('csv')}
                  className="btn btn-primary btn-sm"
                  disabled={filteredStrataProgress.length === 0}
                  title="Export strata progress table as CSV"
                >
                  📊 Export CSV
                </button>
                <button
                  onClick={() => exportStrataProgress('json')}
                  className="btn btn-outline btn-sm"
                  disabled={filteredStrataProgress.length === 0}
                  title="Export strata progress table as JSON"
                >
                  📄 Export JSON
                </button>
              </div>
            </div>
          </div>

          <div style={{
            maxHeight: '600px',
            overflowY: 'auto',
            border: '1px solid #e0e0e0',
            borderRadius: '4px'
          }}>
            <table style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: '0.875rem'
            }}>
              <thead>
                <tr style={{ backgroundColor: '#6e7cb9', color: 'white' }}>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Strata
                  </th>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Species
                  </th>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Min Conf
                  </th>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Status
                  </th>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Progress
                  </th>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Confirmed
                  </th>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Rejected
                  </th>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Uncertain
                  </th>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Skipped
                  </th>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Total
                  </th>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Target
                  </th>
                </tr>
              </thead>
              <tbody>
                {filteredStrataProgress.map((item, index) => {
                  const progressPercentage = item.total_clips > 0 ? (item.validated_clips / item.total_clips) * 100 : 0;

                  return (
                    <tr key={`${item.strata_id}-${item.species_name}`} style={{
                      backgroundColor: index % 2 === 0 ? '#f8f9fa' : 'white'
                    }}>
                      <td style={{ padding: '6px 8px', border: '1px solid #ddd' }}>
                        {item.strata_name}
                      </td>
                      <td style={{ padding: '6px 8px', border: '1px solid #ddd' }}>
                        {item.species_name}
                      </td>
                      <td style={{ padding: '6px 8px', border: '1px solid #ddd', textAlign: 'center' }}>
                        {item.confidence_threshold !== undefined && item.confidence_threshold !== null
                          ? item.confidence_threshold.toFixed(2)
                          : '0.00'}
                      </td>
                      <td style={{ padding: '6px 8px', border: '1px solid #ddd' }}>
                        {getStatusBadge(item.completion_status)}
                      </td>
                      <td style={{ padding: '6px 8px', border: '1px solid #ddd' }}>
                        <div style={{
                          backgroundColor: '#e9ecef',
                          borderRadius: '8px',
                          overflow: 'hidden',
                          height: '16px',
                          minWidth: '100px'
                        }}>
                          <div style={{
                            backgroundColor: getCompletionColor(progressPercentage),
                            height: '100%',
                            width: `${progressPercentage}%`,
                            transition: 'width 0.3s ease'
                          }}></div>
                        </div>
                        <div style={{ fontSize: '0.75rem', marginTop: '2px' }}>
                          {progressPercentage.toFixed(0)}%
                        </div>
                      </td>
                      <td style={{ padding: '6px 8px', border: '1px solid #ddd', textAlign: 'center' }}>
                        <span style={{ color: '#059669', fontWeight: '600' }}>
                          {item.confirmed_clips}
                        </span>
                      </td>
                      <td style={{ padding: '6px 8px', border: '1px solid #ddd', textAlign: 'center' }}>
                        <span style={{ color: '#dc3545', fontWeight: '600' }}>
                          {item.rejected_clips}
                        </span>
                      </td>
                      <td style={{ padding: '6px 8px', border: '1px solid #ddd', textAlign: 'center' }}>
                        <span style={{ color: '#ffc107', fontWeight: '600' }}>
                          {item.uncertain_clips}
                        </span>
                      </td>
                      <td style={{ padding: '6px 8px', border: '1px solid #ddd', textAlign: 'center' }}>
                        <span style={{ color: '#6c757d', fontWeight: '600' }}>
                          {item.skipped_clips}
                        </span>
                      </td>
                      <td style={{ padding: '6px 8px', border: '1px solid #ddd', textAlign: 'center' }}>
                        {item.total_clips}
                      </td>
                      <td style={{ padding: '6px 8px', border: '1px solid #ddd', textAlign: 'center' }}>
                        {item.target_confirmations}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {filteredStrataProgress.length === 0 && (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#666' }}>
              {strataProgress.length === 0 ? (
                <div>
                  <div style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>
                    <strong>No validation data loaded</strong>
                  </div>
                  <p>
                    Please load a validation dataset from the <strong>Validation</strong> tab
                  </p>
                </div>
              ) : (
                'No validation data matches the current filters'
              )}
            </div>
          )}
        </div>
      )}

      {/* Detailed Annotations Section */}
      {selectedView === 'annotations' && (
        <div className="card">
          <div className="card-header">
            <h3>Detailed Validation Annotations</h3>
            <p>Individual validation decisions and timestamps</p>
          </div>

          <div style={{
            maxHeight: '600px',
            overflowY: 'auto',
            border: '1px solid #e0e0e0',
            borderRadius: '4px'
          }}>
            <table style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: '0.875rem'
            }}>
              <thead>
                <tr style={{ backgroundColor: '#6e7cb9', color: 'white' }}>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Filename
                  </th>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Time Range
                  </th>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Strata
                  </th>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Species
                  </th>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Model Conf.
                  </th>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Validation
                  </th>
                  <th style={{ padding: '8px', border: '1px solid #ddd', position: 'sticky', top: 0, backgroundColor: '#6e7cb9' }}>
                    Validated At
                  </th>
                </tr>
              </thead>
              <tbody>
                {detailedAnnotations.map((annotation, index) => (
                  <tr key={annotation.annotation_id} style={{
                    backgroundColor: index % 2 === 0 ? '#f8f9fa' : 'white'
                  }}>
                    <td style={{ padding: '6px 8px', border: '1px solid #ddd', fontFamily: 'monospace', fontSize: '0.75rem' }}>
                      {annotation.filename}
                    </td>
                    <td style={{ padding: '6px 8px', border: '1px solid #ddd' }}>
                      {annotation.start_time?.toFixed(1)}s - {annotation.end_time?.toFixed(1)}s
                    </td>
                    <td style={{ padding: '6px 8px', border: '1px solid #ddd' }}>
                      {annotation.strata_name || 'N/A'}
                    </td>
                    <td style={{ padding: '6px 8px', border: '1px solid #ddd' }}>
                      {annotation.species_name}
                    </td>
                    <td style={{ padding: '6px 8px', border: '1px solid #ddd', textAlign: 'center' }}>
                      {(annotation.original_confidence * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '6px 8px', border: '1px solid #ddd' }}>
                      {getValidationBadge(annotation.validation_state)}
                    </td>
                    <td style={{ padding: '6px 8px', border: '1px solid #ddd' }}>
                      {new Date(annotation.validated_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {detailedAnnotations.length === 0 && !isLoading && (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#666' }}>
              {strataProgress.length === 0 ? (
                <div>
                  <div style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>
                    <strong>No validation data loaded</strong>
                  </div>
                  <p>
                    Please load a validation dataset from the <strong>Validation</strong> tab
                  </p>
                </div>
              ) : (
                'No annotations found for the current filters'
              )}
            </div>
          )}
        </div>
      )}

      {/* Loading Indicator */}
      {isLoading && (
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <div>Loading validation data...</div>
        </div>
      )}
    </div>
  );
};

export default ValidationViewer;