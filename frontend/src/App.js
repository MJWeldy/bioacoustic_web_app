import React, { useState } from 'react';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import DatasetBuilder from './components/DatasetBuilder';
import ActiveLearning from './components/ActiveLearning';
import Evaluation from './components/Evaluation';
import ModelTraining from './components/ModelTraining';
import DatabaseViewer from './components/DatabaseViewer';
import './App.css';

function App() {
  const [selectedTabIndex, setSelectedTabIndex] = useState(0);
  
  return (
    <div className="App">
      <header className="App-header">
        <h1>Bioacoustics Active Learning</h1>
        <p>Build datasets and perform active learning for bioacoustic classification</p>
      </header>
      
      <div className="container">
        <Tabs selectedIndex={selectedTabIndex} onSelect={setSelectedTabIndex} forceRenderTabPanel={true}>
          <TabList>
            <Tab>Dataset Builder</Tab>
            <Tab>Active Learning</Tab>
            <Tab>Model Training</Tab>
            <Tab>Evaluation</Tab>
            <Tab>Database Viewer</Tab>
          </TabList>

          <TabPanel>
            <div style={{ display: selectedTabIndex === 0 ? 'block' : 'none' }}>
              <DatasetBuilder />
            </div>
          </TabPanel>
          
          <TabPanel>
            <ActiveLearning isActive={selectedTabIndex === 1} />
          </TabPanel>
          
          <TabPanel>
            <div style={{ display: selectedTabIndex === 2 ? 'block' : 'none' }}>
              <ModelTraining />
            </div>
          </TabPanel>
          
          <TabPanel>
            <div style={{ display: selectedTabIndex === 3 ? 'block' : 'none' }}>
              <Evaluation />
            </div>
          </TabPanel>
          
          <TabPanel>
            <div style={{ display: selectedTabIndex === 4 ? 'block' : 'none' }}>
              <DatabaseViewer />
            </div>
          </TabPanel>
        </Tabs>
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