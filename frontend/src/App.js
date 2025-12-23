import React, { useState } from 'react';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import FunctionSidebar from './components/FunctionSidebar';
import DatasetBuilder from './components/DatasetBuilder';
import ActiveLearning from './components/ActiveLearning';
import Evaluation from './components/Evaluation';
import ModelTraining from './components/ModelTraining';
import DatabaseViewer from './components/DatabaseViewer';
import ValidationDatasetBuilder from './components/ValidationDatasetBuilder';
import ValidationInterface from './components/ValidationInterface';
import ValidationViewer from './components/ValidationViewer';
import './App.css';

function App() {
  const [selectedFunction, setSelectedFunction] = useState('active_learning');
  const [tabIndexes, setTabIndexes] = useState({
    active_learning: 0,
    validation: 0,
    call_density: 0
  });
  
  const handleTabSelect = (index) => {
    setTabIndexes(prev => ({
      ...prev,
      [selectedFunction]: index
    }));
  };

  const renderFunctionContent = () => {
    const currentTabIndex = tabIndexes[selectedFunction];

    switch (selectedFunction) {
      case 'active_learning':
        return (
          <Tabs selectedIndex={currentTabIndex} onSelect={handleTabSelect} forceRenderTabPanel={true}>
            <TabList>
              <Tab>Dataset Builder</Tab>
              <Tab>Active Learning</Tab>
              <Tab>Model Training</Tab>
              <Tab>Evaluation</Tab>
              <Tab>Database Viewer</Tab>
            </TabList>

            <TabPanel>
              <div style={{ display: currentTabIndex === 0 ? 'block' : 'none' }}>
                <DatasetBuilder />
              </div>
            </TabPanel>

            <TabPanel>
              <ActiveLearning isActive={currentTabIndex === 1} />
            </TabPanel>

            <TabPanel>
              <div style={{ display: currentTabIndex === 2 ? 'block' : 'none' }}>
                <ModelTraining />
              </div>
            </TabPanel>

            <TabPanel>
              <div style={{ display: currentTabIndex === 3 ? 'block' : 'none' }}>
                <Evaluation />
              </div>
            </TabPanel>

            <TabPanel>
              <div style={{ display: currentTabIndex === 4 ? 'block' : 'none' }}>
                <DatabaseViewer />
              </div>
            </TabPanel>
          </Tabs>
        );

      case 'validation':
        return (
          <Tabs selectedIndex={currentTabIndex} onSelect={handleTabSelect} forceRenderTabPanel={true}>
            <TabList>
              <Tab>Dataset Builder</Tab>
              <Tab>Validation</Tab>
              <Tab>Validation Viewer</Tab>
            </TabList>

            <TabPanel>
              <div style={{ display: currentTabIndex === 0 ? 'block' : 'none' }}>
                <ValidationDatasetBuilder />
              </div>
            </TabPanel>

            <TabPanel>
              <div style={{ display: currentTabIndex === 1 ? 'block' : 'none' }}>
                <ValidationInterface />
              </div>
            </TabPanel>

            <TabPanel>
              <div style={{ display: currentTabIndex === 2 ? 'block' : 'none' }}>
                <ValidationViewer />
              </div>
            </TabPanel>
          </Tabs>
        );

      case 'call_density':
        return (
          <Tabs selectedIndex={currentTabIndex} onSelect={handleTabSelect} forceRenderTabPanel={true}>
            <TabList>
              <Tab>Dataset Builder</Tab>
              <Tab>Validation/Review</Tab>
              <Tab>Call Density Results</Tab>
            </TabList>

            <TabPanel>
              <div style={{ display: currentTabIndex === 0 ? 'block' : 'none' }}>
                <div className="placeholder-content">
                  <h2>Call Density Dataset Builder</h2>
                  <p>Prepare audio files for call density analysis.</p>
                  <p><em>Component under development</em></p>
                </div>
              </div>
            </TabPanel>

            <TabPanel>
              <div style={{ display: currentTabIndex === 1 ? 'block' : 'none' }}>
                <div className="placeholder-content">
                  <h2>Call Density Validation/Review</h2>
                  <p>Review and validate call density estimations.</p>
                  <p><em>Component under development</em></p>
                </div>
              </div>
            </TabPanel>

            <TabPanel>
              <div style={{ display: currentTabIndex === 2 ? 'block' : 'none' }}>
                <div className="placeholder-content">
                  <h2>Call Density Results</h2>
                  <p>View call density analysis results and temporal patterns.</p>
                  <p><em>Component under development</em></p>
                </div>
              </div>
            </TabPanel>
          </Tabs>
        );

      default:
        return null;
    }
  };

  return (
    <div className="App">
      <FunctionSidebar
        selectedFunction={selectedFunction}
        onFunctionSelect={setSelectedFunction}
      />

      <div className="main-content">
        <header className="App-header">
          <h1>Bioacoustics Analysis Platform</h1>
          <p>
            {selectedFunction === 'active_learning' && 'Build datasets and perform active learning for bioacoustic classification'}
            {selectedFunction === 'validation' && 'Test and validate model performance with comprehensive metrics'}
            {selectedFunction === 'call_density' && 'Analyze call patterns and estimate temporal density distributions'}
          </p>
        </header>

        <div className="container">
          {renderFunctionContent()}
        </div>
      </div>
      
      <ToastContainer
        position="top-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
      />
    </div>
  );
}

export default App;