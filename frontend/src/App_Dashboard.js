import React, { useState } from 'react';
import {
  Box,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  FormControl,
  Select,
  MenuItem,
  Divider,
  IconButton,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Storage as StorageIcon,
  Psychology as PsychologyIcon,
  ModelTraining as ModelTrainingIcon,
  Assessment as AssessmentIcon,
  TableChart as TableChartIcon,
  CheckCircle as CheckCircleIcon,
  Visibility as VisibilityIcon,
  Build as BuildIcon,
  GitHub as GitHubIcon,
} from '@mui/icons-material';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

// Import components (you'll update these paths as needed)
import DatasetBuilder from './components/DatasetBuilder_MUI';
import ActiveLearning from './components/ActiveLearning_MUI';
import Evaluation from './components/Evaluation_MUI';
import ModelTraining from './components/ModelTraining_MUI';
import DatabaseViewer from './components/DatabaseViewer_MUI';
import ValidationDatasetBuilder from './components/ValidationDatasetBuilder_MUI';
import ValidationInterface from './components/ValidationInterface_MUI';
import ValidationViewer from './components/ValidationViewer_MUI';

const drawerWidth = 260;

const functionConfigs = {
  active_learning: {
    label: 'Active Learning',
    icon: <PsychologyIcon />,
    tabs: [
      { id: 'dataset', label: 'Dataset Builder', icon: <BuildIcon />, component: DatasetBuilder },
      { id: 'learning', label: 'Active Learning', icon: <PsychologyIcon />, component: ActiveLearning },
      { id: 'training', label: 'Model Training', icon: <ModelTrainingIcon />, component: ModelTraining },
      { id: 'evaluation', label: 'Evaluation', icon: <AssessmentIcon />, component: Evaluation },
      { id: 'database', label: 'Database Viewer', icon: <TableChartIcon />, component: DatabaseViewer },
    ],
  },
  validation: {
    label: 'Validation',
    icon: <CheckCircleIcon />,
    tabs: [
      { id: 'dataset', label: 'Dataset Builder', icon: <StorageIcon />, component: ValidationDatasetBuilder },
      { id: 'interface', label: 'Validation', icon: <CheckCircleIcon />, component: ValidationInterface },
      { id: 'viewer', label: 'Validation Viewer', icon: <VisibilityIcon />, component: ValidationViewer },
    ],
  },
};

function App() {
  const [selectedFunction, setSelectedFunction] = useState('active_learning');
  const [selectedTab, setSelectedTab] = useState('dataset');
  const [mobileOpen, setMobileOpen] = useState(false);

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleFunctionChange = (event) => {
    setSelectedFunction(event.target.value);
    setSelectedTab(functionConfigs[event.target.value].tabs[0].id);
  };

  const handleTabChange = (tabId) => {
    setSelectedTab(tabId);
    if (isMobile) {
      setMobileOpen(false);
    }
  };

  const currentConfig = functionConfigs[selectedFunction];
  const currentTab = currentConfig.tabs.find(tab => tab.id === selectedTab);
  const CurrentComponent = currentTab?.component;

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', backgroundColor: '#f5f5f5' }}>
      <Toolbar
        sx={{
          minHeight: '64px !important',
          backgroundColor: '#f5f5f5',
        }}
      />
      <Divider />

      {/* Function Selector */}
      <Box sx={{ p: 2 }}>
        <FormControl fullWidth size="small">
          <Select
            value={selectedFunction}
            onChange={handleFunctionChange}
            sx={{
              backgroundColor: 'white',
              '& .MuiSelect-select': {
                display: 'flex',
                alignItems: 'center',
                gap: 1,
              },
            }}
          >
            {Object.entries(functionConfigs).map(([key, config]) => (
              <MenuItem key={key} value={key}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {config.icon}
                  {config.label}
                </Box>
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      <Divider />

      {/* Navigation Tabs */}
      <List sx={{ flex: 1, pt: 1 }}>
        {currentConfig.tabs.map((tab) => (
          <ListItem key={tab.id} disablePadding>
            <ListItemButton
              selected={selectedTab === tab.id}
              onClick={() => handleTabChange(tab.id)}
              sx={{
                mx: 1,
                mb: 0.5,
                borderRadius: 1,
                '&.Mui-selected': {
                  backgroundColor: '#1976d2',
                  color: 'white',
                  '&:hover': {
                    backgroundColor: '#1565c0',
                  },
                  '& .MuiListItemIcon-root': {
                    color: 'white',
                  },
                },
                '&:hover': {
                  backgroundColor: 'rgba(0, 0, 0, 0.04)',
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 40, color: selectedTab === tab.id ? 'white' : '#666' }}>
                {tab.icon}
              </ListItemIcon>
              <ListItemText
                primary={tab.label}
                primaryTypographyProps={{
                  fontSize: '0.875rem',
                  fontWeight: selectedTab === tab.id ? 600 : 400,
                  color: selectedTab === tab.id ? 'white' : '#333',
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider />

      {/* Footer */}
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary" align="center" display="block" sx={{ mb: 0.5 }}>
          Bioacoustics v1.0
        </Typography>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 0.5 }}>
          <GitHubIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
          <Typography
            variant="caption"
            component="a"
            href="https://github.com/MJWeldy/bioacoustic_web_app"
            target="_blank"
            rel="noopener noreferrer"
            sx={{
              color: 'text.secondary',
              textDecoration: 'none',
              '&:hover': {
                color: 'primary.main',
                textDecoration: 'underline',
              },
            }}
          >
            GitHub
          </Typography>
        </Box>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', backgroundColor: 'white' }}>
      <CssBaseline />

      {/* App Bar */}
      <AppBar
        position="fixed"
        elevation={0}
        sx={{
          width: { md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
          backgroundColor: 'white',
          borderBottom: '1px solid #e0e0e0',
        }}
      >
        <Toolbar>
          <IconButton
            color="default"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1, color: '#333', fontWeight: 600 }}>
            {currentTab?.label}
          </Typography>
          <Typography variant="body2" sx={{ color: '#666' }}>
            {currentConfig.label}
          </Typography>
        </Toolbar>
      </AppBar>

      {/* Drawer */}
      <Box
        component="nav"
        sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
      >
        {/* Mobile drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better mobile performance
          }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              backgroundColor: '#f5f5f5',
            },
          }}
        >
          {drawer}
        </Drawer>

        {/* Desktop drawer */}
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              borderRight: '1px solid #e0e0e0',
              backgroundColor: '#f5f5f5',
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { md: `calc(100% - ${drawerWidth}px)` },
          minHeight: '100vh',
          backgroundColor: 'white',
        }}
      >
        <Toolbar /> {/* Spacer for AppBar */}
        <Box sx={{ p: 3 }}>
          {Object.entries(functionConfigs).map(([funcKey, funcConfig]) => (
            funcConfig.tabs.map((tab) => {
              const Component = tab.component;
              const isActive = selectedFunction === funcKey && selectedTab === tab.id;
              
              // Key needs to be unique across all tabs
              return (
                <Box key={`${funcKey}-${tab.id}`} sx={{ display: isActive ? 'block' : 'none' }}>
                  <Component isActive={isActive} />
                </Box>
              );
            })
          ))}
        </Box>
      </Box>

      {/* Toast notifications */}
      <ToastContainer
        position="bottom-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="light"
      />
    </Box>
  );
}

export default App;
