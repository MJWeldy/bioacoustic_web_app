# Dashboard Layout Implementation Guide

## Overview

The new dashboard layout transforms your app from a tabbed interface into a professional data dashboard with:
- **Left sidebar navigation** (like the MUI template example)
- **Function selector dropdown** (Active Learning / Validation)
- **Full-screen space utilization**
- **Responsive design** (mobile hamburger menu)
- **Modern Material-UI design**

---

## Installation & Setup

### Step 1: Install Material-UI Icons (if not already installed)

```bash
cd frontend
npm install @mui/icons-material
```

### Step 2: Test the Dashboard Layout

Edit `frontend/src/index.js` and change line 4:

```javascript
// Before:
import App from './App';

// After:
import App from './App_Dashboard';
```

### Step 3: Start the Development Server

```bash
npm start
```

---

## What's Changed?

### Layout Transformation

**Before (Tabbed Layout):**
```
┌─────────────────────────────────────────┐
│ [Tab1] [Tab2] [Tab3] [Tab4] [Tab5]     │
├─────────────────────────────────────────┤
│                                         │
│         Content Area                    │
│                                         │
└─────────────────────────────────────────┘
```

**After (Dashboard Layout):**
```
┌────────┬──────────────────────────────┐
│ Logo   │  Current Page Name           │
├────────┼──────────────────────────────┤
│ [Drop  │                              │
│  down] │                              │
├────────┤     Content Area             │
│ ⚙ Tab1 │     (Full Width)             │
│ 🧠 Tab2│                              │
│ 📊 Tab3│                              │
│ 🎯 Tab4│                              │
│ 📋 Tab5│                              │
└────────┴──────────────────────────────┘
```

---

## Key Features

### 1. **Sidebar Navigation (Desktop)**
- **260px fixed sidebar** on the left
- Function selector dropdown at top
- Icon + text navigation items
- Active state highlighting (purple background)
- Version footer at bottom

### 2. **Responsive Mobile Design**
- **Hamburger menu** on mobile/tablet
- Sidebar slides in from left
- Auto-closes after selection
- Breakpoint: `md` (900px)

### 3. **Top AppBar**
- Shows current page name
- Shows current function (Active Learning / Validation)
- Hamburger icon on mobile
- Gradient purple background

### 4. **Content Area**
- **Full width** (minus sidebar)
- Light gray background (#f5f5f5)
- 24px padding around content
- Automatic scrolling

---

## Navigation Structure

The dashboard organizes your app into two main functions:

### Active Learning Function
1. Dataset Builder
2. Active Learning
3. Model Training
4. Evaluation
5. Database Viewer

### Validation Function
1. Dataset Builder
2. Validation
3. Validation Viewer

Users select the function in the dropdown, then navigate between pages in the sidebar.

---

## Customization Options

### Change Sidebar Width

In `App_Dashboard.js`, line 34:
```javascript
const drawerWidth = 260; // Change to 240, 280, etc.
```

### Change Color Scheme

Replace the gradient colors:
```javascript
// Current: Purple gradient
background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'

// Example alternatives:
// Blue: 'linear-gradient(135deg, #2196F3 0%, #1976D2 100%)'
// Green: 'linear-gradient(135deg, #4CAF50 0%, #388E3C 100%)'
// Teal: 'linear-gradient(135deg, #00BCD4 0%, #0097A7 100%)'
```

### Change Icons

Import different icons from `@mui/icons-material`:
```javascript
import { Home, Settings, Person } from '@mui/icons-material';
```

Browse all icons: https://mui.com/material-ui/material-icons/

### Add New Functions

In the `functionConfigs` object:
```javascript
call_density: {
  label: 'Call Density',
  icon: <YourIcon />,
  tabs: [
    { id: 'analysis', label: 'Analysis', icon: <AnalysisIcon />, component: YourComponent },
    // ... more tabs
  ],
},
```

---

## Component Compatibility

### Current Components Work As-Is

The dashboard imports your existing components:
- `DatasetBuilder_MUI` (modernized version)
- `ActiveLearning` (original)
- `ModelTraining` (original)
- `Evaluation` (original)
- `DatabaseViewer` (original)
- `ValidationDatasetBuilder` (original)
- `ValidationInterface` (original)
- `ValidationViewer` (original)

### Props Passed to Components

Each component receives:
```javascript
<YourComponent isActive={true} />
```

The `isActive` prop can be used for conditional logic (e.g., start/stop polling when the tab is visible).

---

## Mobile Behavior

### Breakpoints

- **Desktop** (≥900px): Permanent sidebar
- **Mobile/Tablet** (<900px): Hamburger menu

### Mobile Drawer Behavior

1. Tap hamburger icon → Drawer slides in
2. Tap a navigation item → Page loads, drawer auto-closes
3. Tap outside drawer → Drawer closes
4. Swipe from left → Drawer opens (native gesture)

---

## State Management

The dashboard maintains two pieces of state:

1. **selectedFunction**: 'active_learning' or 'validation'
2. **selectedTab**: Current page ID within the function

When the function changes, the tab automatically resets to the first tab in that function.

---

## Next Steps

### Option 1: Keep Both Layouts
- Keep `App.js` (original tabbed layout)
- Keep `App_Dashboard.js` (new dashboard layout)
- Switch between them in `index.js` as needed

### Option 2: Fully Migrate
```bash
# Backup original
mv frontend/src/App.js frontend/src/App_Tabbed.js

# Use dashboard as main
mv frontend/src/App_Dashboard.js frontend/src/App.js

# Revert index.js to import './App'
```

### Modernize Other Components

Now that you have the dashboard shell, modernize the remaining components with Material-UI:

1. **ActiveLearning.js** → ActiveLearning_MUI.js
2. **ModelTraining.js** → ModelTraining_MUI.js
3. **Evaluation.js** → Evaluation_MUI.js
4. **DatabaseViewer.js** → DatabaseViewer_MUI.js

Follow the same pattern as `DatasetBuilder_MUI.js`:
- Use Material-UI components (TextField, Button, Card, etc.)
- Use Grid for layout
- Use Alert for status messages
- Add icons for visual enhancement

---

## Troubleshooting

### Sidebar not showing on desktop
- Check browser width is ≥900px
- Inspect console for errors

### Component not rendering
- Verify import path in `App_Dashboard.js`
- Check component exports default

### Icons not showing
- Run: `npm install @mui/icons-material`
- Restart dev server

### Mobile menu not closing after selection
- This is handled automatically in `handleTabChange()`
- Check `isMobile` logic

---

## Space Utilization Improvements

The dashboard makes better use of screen space:

1. **Vertical space**: Sidebar uses full height
2. **Horizontal space**: Content area is full width (minus 260px sidebar)
3. **No wasted tabs**: Navigation is compact in sidebar
4. **Scalable**: Add unlimited navigation items without crowding

For even better space usage in components:

### Use Grid Layouts
```javascript
<Grid container spacing={3}>
  <Grid item xs={12} md={6} lg={4}>
    <Card>...</Card>
  </Grid>
  <Grid item xs={12} md={6} lg={4}>
    <Card>...</Card>
  </Grid>
</Grid>
```

### Use Data Grid for Tables
```bash
npm install @mui/x-data-grid
```

```javascript
import { DataGrid } from '@mui/x-data-grid';

<DataGrid
  rows={rows}
  columns={columns}
  pageSize={10}
  autoHeight
/>
```

---

## Summary

✅ Dashboard layout created as `App_Dashboard.js`
✅ Sidebar navigation with function selector
✅ Responsive mobile design
✅ Full-screen space utilization
✅ All existing components compatible
✅ Cross-platform (no OS-specific code)
✅ FastAPI backend unchanged

**Test it now:** Change `index.js` and run `npm start`!
