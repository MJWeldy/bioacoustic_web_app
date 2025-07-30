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
      const response = await axios.post('/api/evaluation/run-evaluation');
      
      if (response.data.status === 'success') {
        toast.success('Evaluation completed successfully');
        setEvaluationResults(response.data.results);
      }
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to run evaluation';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const renderSingleClassResults = (results) => (
    <div className="grid grid-2">
      <div className="card">
        <div className="card-header">
          <h4>Performance Metrics</h4>
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
          </div>
          <div style={{ fontSize: '1.1rem', lineHeight: '1.6' }}>
            <div style={{ marginBottom: '1rem' }}>
              <strong>Macro AUC:</strong> <span style={{ color: '#6e7cb9', fontWeight: '600' }}>{results.macro_auc.toFixed(4)}</span>
            </div>
            <div>
              <strong>Mean Average Precision:</strong> <span style={{ color: '#6e7cb9', fontWeight: '600' }}>{results.mean_ap.toFixed(4)}</span>
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
                      {results.class_aucs[index].toFixed(4)}
                    </td>
                    <td style={{ padding: '8px', border: '1px solid #e89c81', textAlign: 'center', fontWeight: '600' }}>
                      {results.class_aps[index].toFixed(4)}
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
          <h4>Multiclass Confusion Matrix</h4>
        </div>
        <div style={{ overflowX: 'auto' }}>
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