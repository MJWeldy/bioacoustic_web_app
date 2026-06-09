import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Stack,
  Slider,
  FormControlLabel,
  Switch,
  Collapse,
  IconButton,
  Tooltip,
  Chip,
  alpha,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Tune as TuneIcon,
  RestartAlt as ResetIcon,
  Palette as PaletteIcon,
  GraphicEq as FreqIcon,
  Settings as SettingsIcon,
  FilterAlt as FilterIcon,
} from '@mui/icons-material';

const SpectrogramOptions = ({
  options,
  onChange,
  onReset,
  modelDefaults = {}
}) => {
  const [expanded, setExpanded] = useState(false);
  const [bandpassEnabled, setBandpassEnabled] = useState(false);

  const handleOptionChange = (field, value) => {
    onChange({ ...options, [field]: value });
  };

  const handleBandpassToggle = (event) => {
    const enabled = event.target.checked;
    setBandpassEnabled(enabled);
    if (!enabled) {
      onChange({
        ...options,
        bandpass_min: null,
        bandpass_max: null
      });
    }
  };

  const getDefaultValue = (field, defaultValue) => {
    return modelDefaults[field] !== undefined ? modelDefaults[field] : defaultValue;
  };

  // Compact section wrapper
  const OptionSection = ({ icon, title, children, bgcolor }) => (
    <Box
      sx={{
        p: 1.5,
        borderRadius: 1,
        bgcolor: bgcolor || alpha('#000', 0.02),
        border: '1px solid',
        borderColor: alpha('#000', 0.06),
      }}
    >
      <Stack direction="row" spacing={0.75} alignItems="center" sx={{ mb: 1.25 }}>
        {icon}
        <Typography
          variant="overline"
          sx={{
            fontWeight: 700,
            letterSpacing: '0.08em',
            fontSize: '0.65rem',
            color: 'text.secondary',
            lineHeight: 1
          }}
        >
          {title}
        </Typography>
      </Stack>
      {children}
    </Box>
  );

  // Compact parameter slider
  const ParameterSlider = ({ label, value, min, max, step, marks, onChange }) => (
    <Box>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 0.5 }}>
        <Typography variant="caption" sx={{ fontWeight: 500, color: 'text.secondary', fontSize: '0.7rem' }}>
          {label}
        </Typography>
        <Chip
          label={value}
          size="small"
          sx={{
            fontFamily: 'monospace',
            fontWeight: 600,
            fontSize: '0.7rem',
            height: 18,
            '& .MuiChip-label': { px: 0.75, py: 0 },
            bgcolor: alpha('#1976d2', 0.08),
            color: '#1976d2',
          }}
        />
      </Stack>
      <Slider
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e, val) => onChange(val)}
        marks={marks}
        size="small"
        sx={{
          mt: 0.5,
          '& .MuiSlider-thumb': { width: 14, height: 14 },
          '& .MuiSlider-track': { height: 2 },
          '& .MuiSlider-rail': { height: 2, opacity: 0.3 },
          '& .MuiSlider-mark': { width: 1, height: 6, opacity: 0.3 },
          '& .MuiSlider-markLabel': { fontSize: '0.65rem', fontFamily: 'monospace', mt: 0.5 }
        }}
      />
    </Box>
  );

  return (
    <Card
      elevation={0}
      sx={{
        mb: 1.5,
        border: '1px solid',
        borderColor: 'divider',
        overflow: 'hidden',
      }}
    >
      <CardContent sx={{ p: 0 }}>
        {/* Compact Header */}
        <Box
          sx={{
            px: 2,
            py: 1,
            bgcolor: alpha('#000', 0.02),
            borderBottom: '1px solid',
            borderColor: 'divider',
            cursor: 'pointer',
            '&:hover': { bgcolor: alpha('#000', 0.04) }
          }}
          onClick={() => setExpanded(!expanded)}
        >
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Stack direction="row" spacing={1} alignItems="center">
              <TuneIcon sx={{ color: 'text.secondary', fontSize: 18 }} />
              <Typography variant="subtitle2" sx={{ fontWeight: 600, fontSize: '0.85rem' }}>
                Spectrogram & Audio Options
              </Typography>
              {!expanded && (
                <Chip
                  label={options.freq_scale === 'mel' ? 'Mel' : 'Linear'}
                  size="small"
                  sx={{ height: 18, fontSize: '0.65rem', '& .MuiChip-label': { px: 0.75 } }}
                />
              )}
            </Stack>
            <Stack direction="row" spacing={0.25}>
              {onReset && (
                <Tooltip title="Reset">
                  <IconButton
                    size="small"
                    onClick={(e) => { e.stopPropagation(); onReset(); }}
                    sx={{ p: 0.5 }}
                  >
                    <ResetIcon sx={{ fontSize: 18 }} />
                  </IconButton>
                </Tooltip>
              )}
              <IconButton
                size="small"
                sx={{
                  p: 0.5,
                  transition: 'transform 0.3s ease',
                  transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)'
                }}
              >
                <ExpandMoreIcon sx={{ fontSize: 18 }} />
              </IconButton>
            </Stack>
          </Stack>
        </Box>

        {/* Compact Collapsible Content */}
        <Collapse in={expanded}>
          <Box sx={{ p: 1.5 }}>
            <Stack spacing={1.25}>

              {/* Visual - Horizontal layout */}
              <OptionSection icon={<PaletteIcon sx={{ fontSize: 16, color: '#e91e63' }} />} title="Visual">
                <Stack direction="row" spacing={1.5}>
                  <FormControl fullWidth size="small">
                    <InputLabel sx={{ fontSize: '0.8rem' }}>Color</InputLabel>
                    <Select
                      value={options.color_mode || 'viridis'}
                      label="Color"
                      onChange={(e) => handleOptionChange('color_mode', e.target.value)}
                      sx={{ fontSize: '0.8rem' }}
                    >
                      <MenuItem value="viridis">Viridis</MenuItem>
                      <MenuItem value="plasma">Plasma</MenuItem>
                      <MenuItem value="inferno">Inferno</MenuItem>
                      <MenuItem value="gray_r">Grayscale</MenuItem>
                      <MenuItem value="magma">Magma</MenuItem>
                      <MenuItem value="cividis">Cividis</MenuItem>
                    </Select>
                  </FormControl>

                  <FormControl fullWidth size="small">
                    <InputLabel sx={{ fontSize: '0.8rem' }}>Scale</InputLabel>
                    <Select
                      value={options.freq_scale || 'mel'}
                      label="Scale"
                      onChange={(e) => handleOptionChange('freq_scale', e.target.value)}
                      sx={{ fontSize: '0.8rem' }}
                    >
                      <MenuItem value="mel">Mel (Perceptual)</MenuItem>
                      <MenuItem value="linear">Linear</MenuItem>
                    </Select>
                  </FormControl>
                </Stack>
              </OptionSection>

              {/* Analysis Parameters */}
              <OptionSection icon={<SettingsIcon sx={{ fontSize: 16, color: '#2196f3' }} />} title="Analysis">
                <Stack spacing={1.5}>
                  {options.freq_scale === 'mel' && (
                    <ParameterSlider
                      label="Mel Bins"
                      value={options.n_mels || 256}
                      min={64}
                      max={512}
                      step={16}
                      onChange={(val) => handleOptionChange('n_mels', val)}
                      marks={[
                        { value: 64, label: '64' },
                        { value: 256, label: '256' },
                        { value: 512, label: '512' },
                      ]}
                    />
                  )}

                  <ParameterSlider
                    label="FFT Window"
                    value={options.n_fft || 2048}
                    min={512}
                    max={4096}
                    step={256}
                    onChange={(val) => handleOptionChange('n_fft', val)}
                    marks={[
                      { value: 512, label: '512' },
                      { value: 2048, label: '2k' },
                      { value: 4096, label: '4k' },
                    ]}
                  />

                  <ParameterSlider
                    label="Hop Length"
                    value={options.hop_length || 128}
                    min={64}
                    max={512}
                    step={32}
                    onChange={(val) => handleOptionChange('hop_length', val)}
                    marks={[
                      { value: 64, label: '64' },
                      { value: 256, label: '256' },
                      { value: 512, label: '512' },
                    ]}
                  />
                </Stack>
              </OptionSection>

              {/* Frequency Range - Horizontal */}
              <OptionSection icon={<FreqIcon sx={{ fontSize: 16, color: '#ff9800' }} />} title="Frequency Range">
                <Stack direction="row" spacing={1.5}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Min (Hz)"
                    type="number"
                    value={options.fmin !== undefined && options.fmin !== null ? options.fmin : ''}
                    onChange={(e) => handleOptionChange('fmin', e.target.value ? parseFloat(e.target.value) : null)}
                    placeholder={`${getDefaultValue('MIN_FREQ', 60)}`}
                    InputProps={{ sx: { fontFamily: 'monospace', fontSize: '0.8rem' } }}
                    InputLabelProps={{ sx: { fontSize: '0.8rem' } }}
                    FormHelperTextProps={{ sx: { fontSize: '0.65rem', mt: 0.25 } }}
                    helperText="Default if empty"
                  />

                  <TextField
                    fullWidth
                    size="small"
                    label="Max (Hz)"
                    type="number"
                    value={options.fmax !== undefined && options.fmax !== null ? options.fmax : ''}
                    onChange={(e) => handleOptionChange('fmax', e.target.value ? parseFloat(e.target.value) : null)}
                    placeholder={`${getDefaultValue('MAX_FREQ', 10000)}`}
                    InputProps={{ sx: { fontFamily: 'monospace', fontSize: '0.8rem' } }}
                    InputLabelProps={{ sx: { fontSize: '0.8rem' } }}
                    FormHelperTextProps={{ sx: { fontSize: '0.65rem', mt: 0.25 } }}
                    helperText="Default if empty"
                  />
                </Stack>
              </OptionSection>

              {/* Bandpass Filter */}
              <OptionSection
                icon={<FilterIcon sx={{ fontSize: 16, color: '#9c27b0' }} />}
                title="Bandpass"
                bgcolor={bandpassEnabled ? alpha('#9c27b0', 0.04) : alpha('#000', 0.02)}
              >
                <Stack spacing={1}>
                  <FormControlLabel
                    sx={{ m: 0 }}
                    control={
                      <Switch
                        checked={bandpassEnabled}
                        onChange={handleBandpassToggle}
                        size="small"
                      />
                    }
                    label={
                      <Typography variant="caption" sx={{ fontWeight: 500, fontSize: '0.75rem' }}>
                        Enable filtering for noisy audio
                      </Typography>
                    }
                  />

                  <Collapse in={bandpassEnabled}>
                    <Stack direction="row" spacing={1.5} sx={{ pt: 0.5 }}>
                      <TextField
                        fullWidth
                        size="small"
                        label="Low (Hz)"
                        type="number"
                        value={options.bandpass_min || ''}
                        onChange={(e) => handleOptionChange('bandpass_min', parseFloat(e.target.value))}
                        InputProps={{ sx: { fontFamily: 'monospace', fontSize: '0.8rem' } }}
                        InputLabelProps={{ sx: { fontSize: '0.8rem' } }}
                      />

                      <TextField
                        fullWidth
                        size="small"
                        label="High (Hz)"
                        type="number"
                        value={options.bandpass_max || ''}
                        onChange={(e) => handleOptionChange('bandpass_max', parseFloat(e.target.value))}
                        InputProps={{ sx: { fontFamily: 'monospace', fontSize: '0.8rem' } }}
                        InputLabelProps={{ sx: { fontSize: '0.8rem' } }}
                      />
                    </Stack>
                  </Collapse>
                </Stack>
              </OptionSection>

              {/* Context Buffer */}
              <OptionSection
                icon={<SettingsIcon sx={{ fontSize: 16, color: '#4caf50' }} />}
                title="Context Buffer"
                bgcolor={options.buffer_enabled === false ? alpha('#4caf50', 0.04) : alpha('#000', 0.02)}
              >
                <FormControlLabel
                  sx={{ m: 0 }}
                  control={
                    <Switch
                      checked={options.buffer_enabled !== false}
                      onChange={(e) => handleOptionChange('buffer_enabled', e.target.checked)}
                      size="small"
                    />
                  }
                  label={
                    <Typography variant="caption" sx={{ fontWeight: 500, fontSize: '0.75rem' }}>
                      Add 1-second buffer around clip for context
                    </Typography>
                  }
                />
                <Typography variant="caption" sx={{ display: 'block', mt: 0.5, fontSize: '0.65rem', color: 'text.secondary', fontStyle: 'italic' }}>
                  Disable for cleaner exports without surrounding audio
                </Typography>
              </OptionSection>

            </Stack>
          </Box>
        </Collapse>
      </CardContent>
    </Card>
  );
};

export default SpectrogramOptions;
