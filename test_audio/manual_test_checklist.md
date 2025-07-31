# Manual Test Checklist for Bioacoustics Web Application

This document provides a comprehensive manual testing checklist to validate the web application's user interface and functionality.

## Pre-Test Setup

### Prerequisites
- [ ] Application is running (`./run_dev.sh` completed successfully)
- [ ] Frontend accessible at http://localhost:3000
- [ ] Backend accessible at http://localhost:8000
- [ ] API docs accessible at http://localhost:8000/docs
- [ ] Test audio files present in `test_audio/audio/` directory

### Browser Setup
- [ ] Test in Chrome/Chromium (primary)
- [ ] Test in Firefox (secondary)
- [ ] Test in Safari (if on macOS)
- [ ] Developer tools open for console monitoring
- [ ] Network tab monitoring for API calls

## Tab 1: Dataset Builder

### Audio Folder Selection
- [ ] **Input Field Display**: Audio folder path input field is visible and editable
- [ ] **Path Validation**: Enter valid path `/path/to/test_audio/audio` - no errors
- [ ] **Invalid Path Handling**: Enter invalid path - appropriate error message displayed
- [ ] **Path Auto-completion**: Browser auto-complete works for file paths

### Class Map Configuration
- [ ] **Default State**: Class map section is empty/has default values
- [ ] **Add Class**: Click "Add Class" button - new class entry appears
- [ ] **Class Naming**: Enter class name "bird_song" - accepts text input
- [ ] **Numerical Assignment**: Assign numerical value 0 - accepts numbers
- [ ] **Multiple Classes**: Add second class "background" with value 1
- [ ] **Remove Class**: Remove class functionality works correctly
- [ ] **Validation**: Prevent duplicate class names/numbers

### Backend Model Selection
- [ ] **Dropdown Display**: Model selection dropdown is visible
- [ ] **Available Models**: Dropdown contains BirdNET_2.4, PERCH, PNWCnet options
- [ ] **Model Selection**: Can select different models from dropdown
- [ ] **Default Selection**: Default model is pre-selected

### Save Path Configuration
- [ ] **Save Path Input**: Save path input field is visible and editable
- [ ] **Path Suggestion**: Suggests reasonable default path
- [ ] **Custom Path**: Can enter custom save path

### Pretrained Classifier (Optional)
- [ ] **Optional Field**: Field is clearly marked as optional
- [ ] **File Selection**: Can browse and select classifier files
- [ ] **Empty State**: Works correctly when left empty

### Dataset Creation Process
- [ ] **Create Button**: "Create Dataset" button is prominently displayed
- [ ] **Click Response**: Button click initiates dataset creation
- [ ] **Progress Indicator**: Progress bar or spinner appears during processing
- [ ] **Status Updates**: Real-time status messages show progress
- [ ] **Success Message**: Clear success message when completed
- [ ] **Error Handling**: Appropriate error messages for failures
- [ ] **Completion Time**: Process completes within reasonable time (~2-5 minutes)

### Output Validation
- [ ] **File Creation**: Expected output files are created at save path
- [ ] **Status Display**: Success status clearly indicated in UI
- [ ] **Next Steps**: Clear indication of how to proceed to Active Learning

## Tab 2: Active Learning

### Dataset Loading
- [ ] **Dataset Path Input**: Clear input field for dataset path
- [ ] **Load Button**: "Load Dataset" button visible and functional
- [ ] **Loading Feedback**: Progress indicator during dataset loading
- [ ] **Success Confirmation**: Clear message when dataset loaded successfully
- [ ] **Model Detection**: Backend model automatically detected and displayed

### Classifier Loading (Optional)
- [ ] **Classifier Path Input**: Optional classifier path input
- [ ] **Load Classifier Button**: Button to load pretrained classifier
- [ ] **Loading Progress**: Progress indicator for classifier loading
- [ ] **Success/Error Messages**: Clear feedback on classifier loading

### Filter Controls
- [ ] **Score Range Slider**: Dual-handle slider for score range visible
- [ ] **Min/Max Values**: Can adjust minimum and maximum score values
- [ ] **Real-time Update**: Clip list updates as slider values change
- [ ] **Range Display**: Current range values clearly displayed
- [ ] **Reset Function**: Way to reset filters to default values

### Clip Navigation
- [ ] **Clip List Display**: List of clips with scores and metadata
- [ ] **Current Clip Highlight**: Currently selected clip is clearly highlighted
- [ ] **Previous Button**: "Previous" button works and is enabled/disabled appropriately
- [ ] **Next Button**: "Next" button works and is enabled/disabled appropriately
- [ ] **Jump to Clip**: Can jump to specific clip number
- [ ] **Clip Counter**: Current clip number and total count displayed

### Spectrogram Visualization
- [ ] **Spectrogram Display**: Spectrogram image loads and displays correctly
- [ ] **Color Scheme Selection**: Dropdown for colormap selection (viridis, plasma, etc.)
- [ ] **Color Scheme Change**: Changing colormap updates spectrogram in real-time
- [ ] **Image Quality**: Spectrogram is clear and readable
- [ ] **Loading States**: Loading indicator while spectrogram generates

### Audio Playback
- [ ] **Audio Player**: HTML5 audio player is visible and functional
- [ ] **Play/Pause**: Play and pause buttons work correctly
- [ ] **Volume Control**: Volume slider functions properly
- [ ] **Progress Bar**: Seek bar allows jumping to different parts of audio
- [ ] **Auto-load**: Audio automatically loads when navigating to new clip
- [ ] **Audio Quality**: Audio plays clearly without distortion

### Annotation Interface
- [ ] **Annotation Buttons**: Present/Not Present/Uncertain buttons clearly visible
- [ ] **Button States**: Buttons show current annotation state (highlighted/pressed)
- [ ] **Click Response**: Clicking annotation buttons provides immediate feedback
- [ ] **Multi-class Support**: For multi-class, all classes have annotation buttons
- [ ] **Undo Function**: Can change annotation after initial selection

### Progress Tracking
- [ ] **Statistics Display**: Current annotation progress shown (e.g., "5/50 annotated")
- [ ] **Class Breakdown**: Shows annotations per class
- [ ] **Progress Bar**: Visual progress indicator
- [ ] **Real-time Updates**: Statistics update immediately after annotations

### Data Management
- [ ] **Save Database Button**: "Save Database" button visible and functional
- [ ] **Save Confirmation**: Clear confirmation when database saved
- [ ] **Export Clips Button**: "Export Clips" button for exporting annotated clips
- [ ] **Export Configuration**: Options for export (include uncertain, output folder)
- [ ] **Export Progress**: Progress indicator during export process
- [ ] **Export Completion**: Clear message when export completes

## Tab 3: Model Training

### Training Data Loading
- [ ] **Folder Selection**: Input field for selecting training data folder
- [ ] **Load Button**: "Load Training Data" button functional
- [ ] **File Detection**: Automatically detects and displays found audio files
- [ ] **Label Extraction**: Correctly extracts labels from filenames or metadata
- [ ] **Data Summary**: Shows summary of loaded data (file count, class distribution)

### Training Configuration
- [ ] **Backend Model**: Can select embedding model for training
- [ ] **Random State**: Input field for reproducible random state
- [ ] **Test Size**: Slider or input for train/test split ratio
- [ ] **Advanced Options**: Access to additional training parameters
- [ ] **Parameter Validation**: Validates input parameters before training

### Training Process
- [ ] **Start Training Button**: Prominent button to begin training
- [ ] **Progress Bar**: Real-time training progress indicator
- [ ] **Loss Curves**: Live plot of training and validation loss
- [ ] **Metric Display**: Current epoch, loss values, and other metrics
- [ ] **Stop Training**: Ability to stop training early if needed

### Results Display
- [ ] **Training Completion**: Clear indication when training finishes
- [ ] **Final Metrics**: Display of final training and validation metrics
- [ ] **Model Saving**: Confirmation that trained model was saved
- [ ] **Next Steps**: Clear guidance on how to use trained model

## Tab 4: Evaluation

### Evaluation Data Loading
- [ ] **Test Data Selection**: Input field for evaluation dataset folder
- [ ] **Load Test Data**: Button to load evaluation dataset
- [ ] **Data Validation**: Verification that test data is properly formatted
- [ ] **Ground Truth**: Correctly extracts ground truth labels

### Model Selection
- [ ] **Model Path Input**: Field to select trained model for evaluation
- [ ] **Load Model**: Button to load the trained classifier
- [ ] **Model Validation**: Confirmation that model loaded successfully

### Evaluation Execution
- [ ] **Run Evaluation Button**: Button to start evaluation process
- [ ] **Progress Indicator**: Shows evaluation progress
- [ ] **Real-time Updates**: Updates as evaluation proceeds

### Results Visualization
- [ ] **Metrics Display**: AUC, Average Precision, and other metrics clearly shown
- [ ] **Confusion Matrix**: Visual confusion matrix for classification results
- [ ] **Class-specific Metrics**: Individual class performance metrics
- [ ] **Charts and Graphs**: Visual representations of performance data

## Tab 5: Database Viewer

### Database Loading
- [ ] **Database Selection**: Input field for selecting database file
- [ ] **Load Database**: Button to load and examine database
- [ ] **Database Summary**: Overview of database contents and statistics

### Data Exploration
- [ ] **Clip Browser**: Interface to browse through database clips
- [ ] **Filtering Options**: Filters for annotation status, scores, etc.
- [ ] **Search Functionality**: Search for specific clips or metadata
- [ ] **Sorting Options**: Sort clips by various criteria

### Bulk Operations
- [ ] **Bulk Selection**: Select multiple clips for batch operations
- [ ] **Bulk Annotation**: Update annotations for selected clips
- [ ] **Export Options**: Export filtered subsets of data

## Cross-Tab Functionality

### Navigation
- [ ] **Tab Switching**: Can switch between all tabs without issues
- [ ] **State Persistence**: Tab states are maintained when switching
- [ ] **URL Updates**: Browser URL updates to reflect current tab (if applicable)

### Data Flow
- [ ] **Dataset Builder → Active Learning**: Can load dataset created in Dataset Builder
- [ ] **Active Learning → Model Training**: Can use exported clips for training
- [ ] **Model Training → Evaluation**: Can evaluate trained models
- [ ] **Any → Database Viewer**: Can examine databases from any workflow

## Error Handling and Edge Cases

### Input Validation
- [ ] **Empty Fields**: Appropriate handling of empty required fields
- [ ] **Invalid Paths**: Clear error messages for invalid file paths
- [ ] **Invalid File Formats**: Proper handling of unsupported file types
- [ ] **Network Errors**: Graceful handling of backend connection issues

### Resource Management
- [ ] **Large Files**: Proper handling of large audio files
- [ ] **Memory Usage**: Application remains responsive with large datasets
- [ ] **Concurrent Operations**: Multiple simultaneous operations don't break UI

### User Experience
- [ ] **Loading States**: Clear loading indicators for all async operations
- [ ] **Error Recovery**: Can recover from errors without page refresh
- [ ] **Confirmation Dialogs**: Important actions have confirmation prompts
- [ ] **Help Text**: Tooltips or help text for complex features

## Performance Testing

### Response Times
- [ ] **Page Load**: Initial page loads within 3 seconds
- [ ] **Tab Switching**: Tab switches are instantaneous
- [ ] **API Calls**: API responses within 2 seconds for normal operations
- [ ] **File Operations**: File loading/saving provides progress feedback

### Resource Usage
- [ ] **Browser Memory**: Browser memory usage remains reasonable
- [ ] **CPU Usage**: No excessive CPU usage during normal operation
- [ ] **Network Efficiency**: Minimal unnecessary network requests

## Accessibility and Usability

### Interface Design
- [ ] **Visual Hierarchy**: Clear visual hierarchy and organization
- [ ] **Button Sizes**: Buttons are adequately sized for clicking
- [ ] **Color Contrast**: Sufficient contrast for readability
- [ ] **Responsive Design**: Interface adapts to different window sizes

### Keyboard Navigation
- [ ] **Tab Navigation**: Can navigate interface using Tab key
- [ ] **Keyboard Shortcuts**: Essential functions have keyboard shortcuts
- [ ] **Focus Indicators**: Clear focus indicators for keyboard navigation

## Browser Compatibility

### Chrome/Chromium
- [ ] **Full Functionality**: All features work correctly
- [ ] **Performance**: Good performance and responsiveness
- [ ] **Audio Playback**: Audio plays correctly
- [ ] **File Operations**: File selection and operations work

### Firefox
- [ ] **Core Features**: All major features functional
- [ ] **Audio Compatibility**: Audio playback works properly  
- [ ] **UI Rendering**: Interface renders correctly

### Safari (macOS)
- [ ] **Basic Functionality**: Core features work
- [ ] **Audio Support**: Audio formats supported
- [ ] **WebKit Compatibility**: No WebKit-specific issues

## Final Validation

### End-to-End Workflow
- [ ] **Complete Workflow**: Can complete entire workflow from dataset creation to evaluation
- [ ] **Data Integrity**: Data is preserved correctly throughout workflow
- [ ] **File Outputs**: All expected output files are created and valid
- [ ] **Reproducibility**: Can repeat workflows with consistent results

### Documentation Alignment
- [ ] **Feature Parity**: All documented features are present and working
- [ ] **API Compatibility**: Frontend matches documented API endpoints
- [ ] **User Guide Accuracy**: Manual matches actual interface behavior

---

## Test Results

**Date**: _______________  
**Tester**: _______________  
**Browser**: _______________  
**OS**: _______________  

**Overall Status**: ☐ Pass ☐ Fail ☐ Partial  

**Major Issues Found**:
- 
- 
- 

**Minor Issues Found**:
- 
- 
- 

**Additional Notes**:
