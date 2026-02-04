# Component Modernization Guide

## What's Been Completed

✅ **Dashboard Shell** (`App_Dashboard.js`)
- Light grey sidebar, white content area
- Clean, minimal design
- Removed gradients and bright colors
- Added GitHub link in footer

✅ **DatasetBuilder** (`DatasetBuilder_MUI.js`)
- Dashboard-style metric cards
- Two-column layout (configuration + instructions)
- Compact form fields with icons
- Clean borders, no shadows

## Remaining Components to Modernize

| Component | Size | Complexity | Priority |
|-----------|------|------------|----------|
| ActiveLearning.js | 50KB | High | 1 (Most used) |
| ModelTraining.js | 38KB | Medium | 2 |
| Evaluation.js | 33KB | Medium | 3 |
| DatabaseViewer.js | 30KB | Medium | 4 |
| ValidationInterface.js | 43KB | High | 5 |
| ValidationDatasetBuilder.js | 44KB | Medium | 6 |
| ValidationViewer.js | 31KB | Medium | 7 |

## Modernization Pattern

### Step-by-Step Process

For each component, follow this pattern:

#### 1. **Replace Containers**
```javascript
// Before:
<div className="card">
  <div className="card-header">
    <h3>Title</h3>
  </div>
  <div>Content</div>
</div>

// After:
<Card elevation={0} sx={{ border: '1px solid #e0e0e0' }}>
  <CardContent>
    <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
      Title
    </Typography>
    <Box>Content</Box>
  </CardContent>
</Card>
```

#### 2. **Replace Inputs**
```javascript
// Before:
<input
  type="text"
  className="form-control"
  value={value}
  onChange={onChange}
/>

// After:
<TextField
  fullWidth
  size="small"
  value={value}
  onChange={onChange}
  label="Label Text"
/>
```

#### 3. **Replace Buttons**
```javascript
// Before:
<button className="btn btn-primary">Submit</button>
<button className="btn btn-secondary">Cancel</button>

// After:
<Button variant="contained">Submit</Button>
<Button variant="outlined">Cancel</Button>
```

#### 4. **Replace Grids**
```javascript
// Before:
<div className="grid grid-2">
  <div>Column 1</div>
  <div>Column 2</div>
</div>

// After:
<Grid container spacing={2}>
  <Grid item xs={12} md={6}>Column 1</Grid>
  <Grid item xs={12} md={6}>Column 2</Grid>
</Grid>
```

#### 5. **Replace Alerts/Status Messages**
```javascript
// Before:
<div style={{ backgroundColor: '#d0eaf1', padding: '1rem' }}>
  Status message
</div>

// After:
<Alert severity="info">
  Status message
</Alert>
```

## Component-Specific Modernization Notes

### ActiveLearning.js (Priority 1)

**Key Features:**
- Audio player with spectrogram display
- Annotation controls (Present/Absent/Unsure)
- Class selection dropdown
- Score range slider
- Navigation controls (Prev/Next)
- Label statistics display

**Modernization Focus:**
1. **Top Section**: Add dashboard-style cards for statistics
   - Total clips
   - Annotated clips
   - Current class
   - Progress percentage

2. **Main Layout**: Split into left panel (controls) + right panel (spectrogram)
   ```javascript
   <Grid container spacing={3}>
     <Grid item xs={12} lg={4}>
       {/* Controls: class selector, review mode, filters */}
     </Grid>
     <Grid item xs={12} lg={8}>
       {/* Spectrogram and audio player */}
     </Grid>
   </Grid>
   ```

3. **Annotation Buttons**: Use Button Group
   ```javascript
   <ButtonGroup fullWidth variant="outlined">
     <Button onClick={() => annotate(0)}>Absent</Button>
     <Button onClick={() => annotate(0.5)}>Unsure</Button>
     <Button onClick={() => annotate(1)}>Present</Button>
   </ButtonGroup>
   ```

4. **Score Range**: Replace react-range with MUI Slider
   ```javascript
   <Slider
     value={scoreRange}
     onChange={(e, newValue) => setScoreRange(newValue)}
     valueLabelDisplay="auto"
     min={0}
     max={1}
     step={0.01}
   />
   ```

### ModelTraining.js (Priority 2)

**Key Features:**
- Training configuration (epochs, batch size, learning rate)
- Train/stop controls
- Real-time training progress
- Loss curves / metrics visualization

**Modernization Focus:**
1. **Configuration Section**: Use compact form layout
2. **Progress Display**: LinearProgress with status text
3. **Metrics Cards**: Dashboard-style cards showing:
   - Current epoch
   - Training loss
   - Validation accuracy
   - Estimated time remaining

### Evaluation.js (Priority 3)

**Key Features:**
- Model selection
- Test dataset configuration
- Evaluation metrics (precision, recall, F1)
- Confusion matrix
- ROC curves

**Modernization Focus:**
1. **Metrics Dashboard**: Grid of Paper cards showing key metrics
2. **Visualization Area**: Full-width section for charts
3. **Results Table**: Use DataGrid or TableContainer

### DatabaseViewer.js (Priority 4)

**Key Features:**
- Table display of clips/annotations
- Filtering controls
- Export functionality

**Modernization Focus:**
1. **Filters Section**: Compact form with multiple filters
2. **Data Table**: Use MUI TableContainer with sortable columns
3. **Actions**: IconButtons for row actions (view, edit, delete)

## Quick Start Template

Here's a template to speed up modernization:

```javascript
import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Grid,
  Paper,
  Typography,
  Alert,
  Divider,
  Stack,
} from '@mui/material';
import { toast } from 'react-toastify';

const YourComponent = ({ isActive }) => {
  // ... existing state and logic ...

  return (
    <Box>
      {/* Status/Metrics Dashboard */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Paper elevation={0} sx={{ p: 2, border: '1px solid #e0e0e0' }}>
            <Typography variant="caption" color="text.secondary">
              Metric Label
            </Typography>
            <Typography variant="h4" sx={{ fontWeight: 600, color: '#1976d2' }}>
              Value
            </Typography>
          </Paper>
        </Grid>
        {/* Repeat for other metrics */}
      </Grid>

      {/* Main Content */}
      <Grid container spacing={3}>
        {/* Left Panel: Controls */}
        <Grid item xs={12} lg={4}>
          <Card elevation={0} sx={{ border: '1px solid #e0e0e0' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Configuration
              </Typography>

              <Stack spacing={2}>
                <TextField
                  fullWidth
                  size="small"
                  label="Input Label"
                  // ... props
                />

                <Button variant="contained" fullWidth>
                  Submit
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Right Panel: Visualization */}
        <Grid item xs={12} lg={8}>
          <Card elevation={0} sx={{ border: '1px solid #e0e0e0' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Display Area
              </Typography>
              {/* Content here */}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default YourComponent;
```

## Color Palette (Consistent with Dashboard)

```javascript
const theme = {
  grey: {
    light: '#fafafa',      // Background for info cards
    border: '#e0e0e0',     // Card borders
    text: '#666',          // Secondary text
  },
  primary: '#1976d2',      // Blue for selected items, metrics
  success: '#10b981',      // Green for positive states
  error: '#ef4444',        // Red for errors
  warning: '#f59e0b',      // Orange for warnings
  info: '#3b82f6',         // Blue for info alerts
};
```

## Progressive Migration Strategy

### Phase 1: Layout & Structure (Week 1)
1. ✅ Dashboard shell (DONE)
2. ✅ DatasetBuilder (DONE)
3. Active Learning - Convert layout to Grid
4. Model Training - Convert layout to Grid

### Phase 2: Form Components (Week 2)
1. Replace all `<input>` with `<TextField>`
2. Replace all `<button>` with `<Button>`
3. Replace `<select>` with `<Select>` or `<Autocomplete>`

### Phase 3: Status & Feedback (Week 3)
1. Add dashboard metric cards
2. Replace custom alerts with MUI `<Alert>`
3. Add progress indicators with `<LinearProgress>`

### Phase 4: Polish (Week 4)
1. Add icons to buttons and inputs
2. Refine spacing and sizing
3. Test responsive behavior
4. Performance optimization

## Testing Checklist

For each modernized component:

- [ ] Layout looks good on desktop (1920x1080)
- [ ] Layout is responsive on tablet (768px)
- [ ] Layout works on mobile (375px)
- [ ] All functionality preserved
- [ ] No console errors
- [ ] Loading states work correctly
- [ ] Error handling works
- [ ] API calls still function
- [ ] Toast notifications appear correctly

## Common Pitfalls

1. **Forgetting to import MUI components**
   ```javascript
   import { Box, TextField } from '@mui/material';
   ```

2. **Not using `sx` prop for styling**
   ```javascript
   // Prefer this:
   <Box sx={{ p: 2, mb: 3 }}>

   // Over this:
   <div style={{ padding: '16px', marginBottom: '24px' }}>
   ```

3. **Inconsistent sizing**
   - Use `size="small"` for form fields to keep UI compact
   - Use consistent spacing (2, 3, etc.) instead of px values

4. **Forgetting elevation={0}**
   - Cards should have `elevation={0}` and `border: '1px solid #e0e0e0'`

## Need Help?

- MUI Documentation: https://mui.com/material-ui/
- Component API: https://mui.com/material-ui/api/
- Icons: https://mui.com/material-ui/material-icons/

## Next Steps

1. Start with **ActiveLearning.js** (most used component)
2. Create `ActiveLearning_MUI.js` (don't replace original yet)
3. Test thoroughly
4. Once satisfied, replace original
5. Repeat for remaining components

Remember: **Preserve all functionality**. The goal is to improve the visual design while keeping everything working exactly as before.
