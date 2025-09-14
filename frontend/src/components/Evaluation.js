import React, { useState } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';

const Evaluation = () => {
  const [evaluationDatasetPath, setEvaluationDatasetPath] = useState('');
  const [classifierPath, setClassifierPath] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [isDatasetLoaded, setIsDatasetLoaded] = useState(false);
  const [isClassifierLoaded, setIsClassifierLoaded] = useState(false);
  const [datasetMetadata, setDatasetMetadata] = useState(null);
  const [showPredictions, setShowPredictions] = useState(false);
  const [threshold, setThreshold] = useState(0);
  const [selectedConfusionClasses, setSelectedConfusionClasses] = useState([]);
  const [selectedPredictionClasses, setSelectedPredictionClasses] = useState([]);

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
        // Initialize class selections with just the first class
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
        params: { 
          export_path: exportPath,
          filename: fileName
        }
      });
      
      if (response.data.status === 'success') {
        toast.success(response.data.message);
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to export metrics CSV';
      toast.error(message);
    }
  };

  const exportPredictionsCSV = async () => {
    const exportPath = prompt('Enter export directory path:');
    if (!exportPath) return;
    
    const fileName = prompt('Enter filename (without extension):', 'evaluation_predictions');
    if (!fileName) return;

    try {
      const response = await axios.post('/api/evaluation/export-predictions-csv', null, {
        params: { 
          export_path: exportPath,
          filename: fileName
        }
      });
      
      if (response.data.status === 'success') {
        toast.success(response.data.message);
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to export predictions CSV';
      toast.error(message);
    }
  };

  const renderSingleClassResults = (results) => (
    <div className="grid grid-2">
      <div className="card">
        <div className="card-header">
          <h4>Performance Metrics</h4>
          <small style={{ color: '#666', fontWeight: 'normal' }}>
            Evaluation Level: {results.evaluation_level} ({results.num_samples} {results.evaluation_level}s)
          </small>
        </div>
        <div style={{ fontSize: '1.1rem', lineHeight: '1.6' }}>
          <div style={{ marginBottom: '1rem' }}>
            <strong>AUC:</strong> <span style={{ color: '#6e7cb9', fontWeight: '600' }}>{results.auc.toFixed(4)}</span>
          </div>
          <div>
            <strong>Average Precision:</strong> <span style={{ color: '#6e7cb9', fontWeight: '600' }}>{results.average_precision.toFixed(4)}</span>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h4>Confusion Matrix</h4>
        </div>
        <div style={{ textAlign: 'center' }}>
          <table style={{ margin: '0 auto', borderCollapse: 'collapse', fontSize: '1.1rem' }}>
            <thead>
              <tr>
                <th style={{ padding: '8px', border: '1px solid #e89c81' }}></th>
                <th style={{ padding: '8px', border: '1px solid #e89c81', backgroundColor: '#f5db99' }}>Predicted Negative</th>
                <th style={{ padding: '8px', border: '1px solid #e89c81', backgroundColor: '#f5db99' }}>Predicted Positive</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th style={{ padding: '8px', border: '1px solid #e89c81', backgroundColor: '#f5db99' }}>Actual Negative</th>
                <td style={{ padding: '8px', border: '1px solid #e89c81', fontWeight: '600' }}>{results.confusion_matrix[0][0]}</td>
                <td style={{ padding: '8px', border: '1px solid #e89c81', fontWeight: '600' }}>{results.confusion_matrix[0][1]}</td>
              </tr>
              <tr>
                <th style={{ padding: '8px', border: '1px solid #e89c81', backgroundColor: '#f5db99' }}>Actual Positive</th>
                <td style={{ padding: '8px', border: '1px solid #e89c81', fontWeight: '600' }}>{results.confusion_matrix[1][0]}</td>
                <td style={{ padding: '8px', border: '1px solid #e89c81', fontWeight: '600' }}>{results.confusion_matrix[1][1]}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  const renderMultiClassResults = (results) => (
    <div>
      <div className="grid grid-2">
        <div className="card">
          <div className="card-header">
            <h4>Overall Performance Metrics</h4>
            <small style={{ color: '#666', fontWeight: 'normal' }}>
              Evaluation Level: {results.evaluation_level} ({results.num_samples} {results.evaluation_level}s)
            </small>
          </div>
          <div style={{ fontSize: '1.1rem', lineHeight: '1.6' }}>
            <div style={{ marginBottom: '1rem' }}>
              <strong>Macro AUC:</strong> <span style={{ color: '#6e7cb9', fontWeight: '600' }}>
                {results.macro_auc !== null && results.macro_auc !== undefined ? results.macro_auc.toFixed(4) : 'N/A'}
              </span>
            </div>
            <div>
              <strong>Mean Average Precision:</strong> <span style={{ color: '#6e7cb9', fontWeight: '600' }}>
                {results.mean_ap !== null && results.mean_ap !== undefined ? results.mean_ap.toFixed(4) : 'N/A'}
              </span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h4>Class-Specific Metrics</h4>
          </div>
          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
              <thead>
                <tr style={{ backgroundColor: '#f5db99' }}>
                  <th style={{ padding: '8px', border: '1px solid #e89c81', textAlign: 'left' }}>Class</th>
                  <th style={{ padding: '8px', border: '1px solid #e89c81' }}>AUC</th>
                  <th style={{ padding: '8px', border: '1px solid #e89c81' }}>AP</th>
                </tr>
              </thead>
              <tbody>
                {results.class_names.map((className, index) => (
                  <tr key={index}>
                    <td style={{ padding: '8px', border: '1px solid #e89c81' }}>{className}</td>
                    <td style={{ padding: '8px', border: '1px solid #e89c81', textAlign: 'center', fontWeight: '600' }}>
                      {results.class_aucs[index] !== null && results.class_aucs[index] !== undefined ? results.class_aucs[index].toFixed(4) : 'N/A'}
                    </td>
                    <td style={{ padding: '8px', border: '1px solid #e89c81', textAlign: 'center', fontWeight: '600' }}>
                      {results.class_aps[index] !== null && results.class_aps[index] !== undefined ? results.class_aps[index].toFixed(4) : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h4>{results.confusion_matrix_type === 'multilabel' ? 'Binary Confusion Matrices (Multi-label)' : 'Multiclass Confusion Matrix'}</h4>
          <small style={{ color: '#666', fontWeight: 'normal', display: 'block', marginTop: '0.25rem' }}>
            {results.confusion_matrix_type === 'multilabel' 
              ? 'Separate binary confusion matrix for each class (Rows = Actual, Columns = Predicted)'
              : 'Rows = Actual Labels, Columns = Predicted Labels'
            }
          </small>
        </div>
        
        {results.confusion_matrix_type === 'multilabel' ? (
          <div>
            <div style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
              <label htmlFor="confusionClassSelect" style={{ fontWeight: '600', minWidth: '120px' }}>
                Select Classes:
              </label>
              <select
                id="confusionClassSelect"
                multiple
                value={selectedConfusionClasses}
                onChange={(e) => {
                  const selected = Array.from(e.target.selectedOptions, option => option.value);
                  setSelectedConfusionClasses(selected);
                }}
                style={{ 
                  padding: '8px', 
                  border: '1px solid #ccc', 
                  borderRadius: '4px',
                  minHeight: '100px',
                  minWidth: '200px',
                  fontSize: '0.875rem'
                }}
              >
                {results.class_names.map((className, index) => (
                  <option key={index} value={className}>
                    {className}
                  </option>
                ))}
              </select>
              <div style={{ fontSize: '0.875rem', color: '#666', maxWidth: '300px' }}>
                Hold Ctrl/Cmd to select multiple classes. Shows confusion matrices for selected classes only.
              </div>
            </div>
            <div style={{ overflowY: 'auto', maxHeight: '400px' }}>
              {results.confusion_matrix.map((cm, classIndex) => {
                const className = results.class_names[classIndex];
                if (!selectedConfusionClasses.includes(className)) return null;
                return (
                  <div key={classIndex} style={{ marginBottom: '2rem', padding: '1rem' }}>
                    <h5 style={{ textAlign: 'center', marginBottom: '1rem', color: '#6e7cb9' }}>
                      {results.class_names[classIndex]}
                    </h5>
                    <div style={{ display: 'flex', justifyContent: 'center' }}>
                  <table style={{ borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                    <thead>
                      <tr>
                        <th style={{ padding: '8px', border: '1px solid #e89c81' }}></th>
                        <th style={{ padding: '8px', border: '1px solid #e89c81', backgroundColor: '#f5db99' }}>Predicted Absent</th>
                        <th style={{ padding: '8px', border: '1px solid #e89c81', backgroundColor: '#f5db99' }}>Predicted Present</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <th style={{ padding: '8px', border: '1px solid #e89c81', backgroundColor: '#f5db99' }}>Actual Absent</th>
                        <td style={{ padding: '8px', border: '1px solid #e89c81', textAlign: 'center', fontWeight: '600', backgroundColor: '#d0eaf1' }}>{cm[0][0]}</td>
                        <td style={{ padding: '8px', border: '1px solid #e89c81', textAlign: 'center', fontWeight: '600' }}>{cm[0][1]}</td>
                      </tr>
                      <tr>
                        <th style={{ padding: '8px', border: '1px solid #e89c81', backgroundColor: '#f5db99' }}>Actual Present</th>
                        <td style={{ padding: '8px', border: '1px solid #e89c81', textAlign: 'center', fontWeight: '600' }}>{cm[1][0]}</td>
                        <td style={{ padding: '8px', border: '1px solid #e89c81', textAlign: 'center', fontWeight: '600', backgroundColor: '#d0eaf1' }}>{cm[1][1]}</td>
                      </tr>
                      </tbody>
                    </table>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '2rem', marginBottom: '1rem' }}>
              <div style={{ 
                fontSize: '0.875rem', 
                color: '#666',
                transform: 'rotate(-90deg)',
                transformOrigin: 'center',
                width: '0px',
                textAlign: 'center'
              }}>
                <strong>↑ Actual Labels</strong>
              </div>
              <div>
                <div style={{ textAlign: 'center', marginBottom: '0.5rem', fontSize: '0.875rem', color: '#666' }}>
                  <strong>Predicted Labels →</strong>
                </div>
                <table style={{ margin: '0 auto', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                  <thead>
                    <tr>
                      <th style={{ padding: '8px', border: '1px solid #e89c81' }}></th>
                      {results.class_names.map((className, index) => (
                        <th key={index} style={{ 
                          padding: '8px', 
                          border: '1px solid #e89c81', 
                          backgroundColor: '#f5db99',
                          transform: 'rotate(-45deg)',
                          minWidth: '60px'
                        }}>
                          {className}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {results.confusion_matrix.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        <th style={{ 
                          padding: '8px', 
                          border: '1px solid #e89c81', 
                          backgroundColor: '#f5db99',
                          textAlign: 'left'
                        }}>
                          {results.class_names[rowIndex]}
                        </th>
                        {row.map((value, colIndex) => (
                          <td key={colIndex} style={{ 
                            padding: '8px', 
                            border: '1px solid #e89c81', 
                            textAlign: 'center',
                            fontWeight: '600',
                            backgroundColor: rowIndex === colIndex ? '#d0eaf1' : 'white'
                          }}>
                            {value}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <h3>Model Evaluation</h3>
          <p>Load an evaluation dataset and classifier to calculate performance metrics</p>
        </div>

        <div className="grid grid-2">
          <div className="form-group">
            <label htmlFor="evaluationDatasetPath">Evaluation Dataset Path</label>
            <div style={{ display: 'flex', gap: '10px' }}>
              <input
                type="text"
                id="evaluationDatasetPath"
                className="form-control"
                placeholder="/path/to/evaluation/dataset"
                value={evaluationDatasetPath}
                onChange={(e) => setEvaluationDatasetPath(e.target.value)}
                style={{ flex: 1 }}
              />
              <button
                onClick={loadEvaluationDataset}
                disabled={isLoading}
                className="btn btn-primary"
              >
                Load Dataset
              </button>
            </div>
            <small style={{ color: '#666', fontSize: '0.875rem' }}>
              Path to evaluation dataset with embeddings and labels
            </small>
          </div>

          <div className="form-group">
            <label htmlFor="classifierPath">Classifier Model</label>
            <div style={{ display: 'flex', gap: '10px' }}>
              <input
                type="text"
                id="classifierPath"
                className="form-control"
                placeholder="/path/to/classifier.keras"
                value={classifierPath}
                onChange={(e) => setClassifierPath(e.target.value)}
                style={{ flex: 1 }}
              />
              <button
                onClick={loadClassifier}
                disabled={isLoading || !isDatasetLoaded}
                className="btn btn-secondary"
              >
                Load Classifier
              </button>
            </div>
            <small style={{ color: '#666', fontSize: '0.875rem' }}>
              Path to trained Keras classifier model
            </small>
          </div>
        </div>

        <div className="form-group" style={{ marginTop: '1.5rem' }}>
          <label htmlFor="threshold">Class Threshold</label>
          <input
            type="number"
            id="threshold"
            className="form-control"
            placeholder="0"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value) || 0)}
            min="0"
            step="1"
            style={{ maxWidth: '200px', margin: '0 auto' }}
          />
          <small style={{ color: '#666', fontSize: '0.875rem', display: 'block', marginTop: '0.25rem' }}>
            Minimum number of positive labels required per class for inclusion in metrics (0 = include all classes)
          </small>
        </div>

        <div style={{ textAlign: 'center', marginTop: '2rem' }}>
          <button
            onClick={runEvaluation}
            disabled={isLoading || !isDatasetLoaded || !isClassifierLoaded}
            className="btn btn-success btn-lg"
          >
            {isLoading ? 'Running Evaluation...' : 'Run Evaluation'}
          </button>
        </div>

        {(isDatasetLoaded || isClassifierLoaded) && (
          <div style={{ marginTop: '1.5rem', padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '6px' }}>
            <div style={{ display: 'flex', gap: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div className={`status-indicator ${isDatasetLoaded ? 'status-success' : 'status-error'}`}></div>
                <span>Evaluation Dataset</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div className={`status-indicator ${isClassifierLoaded ? 'status-success' : 'status-error'}`}></div>
                <span>Classifier Model</span>
              </div>
            </div>
          </div>
        )}

        {isDatasetLoaded && datasetMetadata && (
          <div style={{ 
            padding: '1rem', 
            backgroundColor: '#f8f9fa', 
            borderRadius: '6px',
            marginTop: '1rem',
            border: '1px solid #e89c81'
          }}>
            <strong style={{ color: '#6e7cb9' }}>Dataset Information:</strong>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginTop: '0.5rem', fontSize: '0.9rem' }}>
              <div><strong>Type:</strong> {datasetMetadata.dataset_info?.dataset_type || 'Unknown'}</div>
              <div><strong>Model:</strong> {datasetMetadata.dataset_info?.backend_model || 'Unknown'}</div>
              <div><strong>Created:</strong> {datasetMetadata.dataset_info?.creation_date ? new Date(datasetMetadata.dataset_info.creation_date).toLocaleDateString() : 'Unknown'}</div>
              <div><strong>Classes:</strong> {Object.keys(datasetMetadata.class_map || {}).join(', ')}</div>
              <div><strong>Samples:</strong> {datasetMetadata.statistics?.total_files || 'Unknown'} files</div>
              <div><strong>Clips:</strong> {datasetMetadata.statistics?.total_clips || 'Unknown'} clips</div>
            </div>
          </div>
        )}
      </div>

      {evaluationResults && (
        <div>
          <div className="card">
            <div className="card-header">
              <h3>Evaluation Results</h3>
              <p>Performance metrics for {evaluationResults.is_single_class ? 'single class' : 'multiclass'} classification</p>
            </div>
          </div>
          
          {evaluationResults.is_single_class 
            ? renderSingleClassResults(evaluationResults)
            : renderMultiClassResults(evaluationResults)
          }

          {evaluationResults.detailed_predictions && evaluationResults.detailed_predictions.length > 0 && (
            <div className="card">
              <div className="card-header">
                <h3 
                  style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
                  onClick={() => setShowPredictions(!showPredictions)}
                >
                  <span>Predictions Viewer ({evaluationResults.detailed_predictions.length} {evaluationResults.evaluation_level}s)</span>
                  <span>{showPredictions ? '▼' : '▶'}</span>
                </h3>
                <p>Detailed predictions for each {evaluationResults.evaluation_level}</p>
              </div>
              
              {showPredictions && (
                <div>
                  {!evaluationResults.is_single_class && (
                    <div style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap', padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '6px' }}>
                      <label htmlFor="predictionClassSelect" style={{ fontWeight: '600', minWidth: '120px' }}>
                        Select Classes:
                      </label>
                      <select
                        id="predictionClassSelect"
                        multiple
                        value={selectedPredictionClasses}
                        onChange={(e) => {
                          const selected = Array.from(e.target.selectedOptions, option => option.value);
                          setSelectedPredictionClasses(selected);
                        }}
                        style={{ 
                          padding: '8px', 
                          border: '1px solid #ccc', 
                          borderRadius: '4px',
                          minHeight: '100px',
                          minWidth: '200px',
                          fontSize: '0.875rem'
                        }}
                      >
                        {evaluationResults.class_names.map((className, index) => (
                          <option key={index} value={className}>
                            {className}
                          </option>
                        ))}
                      </select>
                      <div style={{ fontSize: '0.875rem', color: '#666', maxWidth: '300px' }}>
                        Hold Ctrl/Cmd to select multiple classes. Shows prediction and label columns for selected classes only.
                      </div>
                    </div>
                  )}
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
                    <thead style={{ position: 'sticky', top: 0, backgroundColor: '#f5db99' }}>
                      <tr>
                        <th style={{ padding: '12px 8px', border: '1px solid #e89c81', textAlign: 'left', fontWeight: '600' }}>
                          {evaluationResults.evaluation_level === 'file' ? 'File Name' : 'Sample Name'}
                        </th>
                        {evaluationResults.is_single_class ? (
                          <>
                            <th style={{ padding: '12px 8px', border: '1px solid #e89c81', textAlign: 'center', fontWeight: '600' }}>Prediction</th>
                            <th style={{ padding: '12px 8px', border: '1px solid #e89c81', textAlign: 'center', fontWeight: '600' }}>Label</th>
                          </>
                        ) : (
                          selectedPredictionClasses.map((className) => (
                            <React.Fragment key={className}>
                              <th style={{ padding: '12px 8px', border: '1px solid #e89c81', textAlign: 'center', fontWeight: '600', backgroundColor: '#e8f4fd' }}>
                                {className} (Pred)
                              </th>
                              <th style={{ padding: '12px 8px', border: '1px solid #e89c81', textAlign: 'center', fontWeight: '600', backgroundColor: '#fff2e8' }}>
                                {className} (Label)
                              </th>
                            </React.Fragment>
                          ))
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {evaluationResults.detailed_predictions.map((item, index) => (
                        <tr key={index} style={{ backgroundColor: index % 2 === 0 ? '#f9f9f9' : 'white' }}>
                          <td style={{ padding: '8px', border: '1px solid #e89c81', fontFamily: 'monospace', fontSize: '0.8rem' }}>
                            {item.file_name}
                          </td>
                          {evaluationResults.is_single_class ? (
                            <>
                              <td style={{ 
                                padding: '8px', 
                                border: '1px solid #e89c81', 
                                textAlign: 'center',
                                fontFamily: 'monospace',
                                color: item.predictions[0] > 0.5 ? '#d73527' : '#6c757d'
                              }}>
                                {item.predictions[0].toFixed(4)}
                              </td>
                              <td style={{ 
                                padding: '8px', 
                                border: '1px solid #e89c81', 
                                textAlign: 'center',
                                fontFamily: 'monospace',
                                color: item.labels[0] === 1 ? '#d73527' : '#6c757d',
                                fontWeight: '600'
                              }}>
                                {item.labels[0]}
                              </td>
                            </>
                          ) : (
                            selectedPredictionClasses.map((className) => {
                              const classIndex = evaluationResults.class_names.indexOf(className);
                              const prediction = classIndex >= 0 ? item.predictions[classIndex] : 0;
                              const label = classIndex >= 0 ? item.labels[classIndex] : 0;
                              
                              return (
                                <React.Fragment key={className}>
                                  <td style={{ 
                                    padding: '8px', 
                                    border: '1px solid #e89c81', 
                                    textAlign: 'center',
                                    fontFamily: 'monospace',
                                    fontSize: '0.85rem',
                                    backgroundColor: '#f0f8ff',
                                    color: prediction > 0.5 ? '#d73527' : '#6c757d'
                                  }}>
                                    {prediction.toFixed(4)}
                                  </td>
                                  <td style={{ 
                                    padding: '8px', 
                                    border: '1px solid #e89c81', 
                                    textAlign: 'center',
                                    fontFamily: 'monospace',
                                    fontSize: '0.85rem',
                                    backgroundColor: '#fffaf0',
                                    color: label === 1 ? '#d73527' : '#6c757d',
                                    fontWeight: '600'
                                  }}>
                                    {label}
                                  </td>
                                </React.Fragment>
                              );
                            })
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="card">
            <div className="card-header">
              <h3>Export Results</h3>
              <p>Download evaluation results in CSV format</p>
            </div>
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
              <button
                onClick={exportMetricsCSV}
                className="btn btn-outline-primary"
                disabled={isLoading}
              >
                📊 Export Metrics Summary CSV
              </button>
              <button
                onClick={exportPredictionsCSV}
                className="btn btn-outline-secondary"
                disabled={isLoading}
              >
                📋 Export Predictions CSV
              </button>
            </div>
            <div style={{ fontSize: '0.875rem', color: '#666', marginTop: '1rem', lineHeight: '1.5' }}>
              <div><strong>Metrics CSV:</strong> Summary table with AUC and AP metrics (macro and per-class)</div>
              <div><strong>Predictions CSV:</strong> Detailed predictions for each {evaluationResults?.evaluation_level || 'sample'} with class scores</div>
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <h3>Instructions</h3>
        </div>
        <div style={{ lineHeight: '1.6' }}>
          <ol>
            <li><strong>Load Evaluation Dataset:</strong> Select a dataset created with the "Evaluation Dataset" option checked</li>
            <li><strong>Load Classifier:</strong> Load a trained Keras model file (.keras)</li>
            <li><strong>Run Evaluation:</strong> Calculate performance metrics and confusion matrices</li>
          </ol>
          <p><strong>Metrics Displayed:</strong></p>
          <ul style={{ marginTop: '8px' }}>
            <li><strong>Single Class:</strong> AUC, Average Precision, and 2x2 confusion matrix</li>
            <li><strong>Multiple Classes:</strong> Macro AUC, Mean Average Precision, class-specific metrics, and full confusion matrix</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Evaluation;