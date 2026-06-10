import React from 'react';
import './SpectrogramViewer.css';

/**
 * Utility functions for formatting time and frequency
 */
export const formatTime = (seconds) => {
  if (seconds === undefined || seconds === null) return '0:00.00';
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(2);
  return `${mins}:${secs.padStart(5, '0')}`;
};

export const formatFreq = (hz) => {
  if (hz === undefined || hz === null) return '0Hz';
  if (hz >= 1000) {
    return `${(hz / 1000).toFixed(1)}k`;
  }
  return `${Math.round(hz)}`;
};

/**
 * Renders spectrogram with axes to a canvas and returns a downloadable blob
 * @param {string} spectrogramUrl - URL of the spectrogram image
 * @param {object} metadata - Spectrogram metadata with time/freq ranges
 * @returns {Promise<Blob>} - PNG blob of the rendered spectrogram with axes
 */
export const renderSpectrogramWithAxes = async (spectrogramUrl, metadata) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      // Canvas dimensions with margins for axes
      const marginLeft = 80;
      const marginRight = 20;
      const marginTop = 20;
      const marginBottom = 80;

      const canvasWidth = img.width + marginLeft + marginRight;
      const canvasHeight = img.height + marginTop + marginBottom;

      const canvas = document.createElement('canvas');
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;
      const ctx = canvas.getContext('2d');

      // White background
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvasWidth, canvasHeight);

      // Draw spectrogram image
      ctx.drawImage(img, marginLeft, marginTop);

      // Configure text style
      ctx.fillStyle = '#000000';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';

      // Mel scale conversion functions
      const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
      const melToHz = (mel) => 700 * (Math.pow(10, mel / 2595) - 1);

      // Draw frequency axis (left side)
      if (metadata) {
        const { freq_min, freq_max, freq_scale, time_start, time_end } = metadata;
        const isLinear = freq_scale === 'linear';
        const tickCount = 5;

        // Frequency ticks
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';

        for (let i = 0; i <= tickCount; i++) {
          let freq;
          if (isLinear) {
            freq = freq_max - (i / tickCount) * (freq_max - freq_min);
          } else {
            const melMin = hzToMel(freq_min);
            const melMax = hzToMel(freq_max);
            const mel = melMax - (i / tickCount) * (melMax - melMin);
            freq = melToHz(mel);
          }

          const y = marginTop + (i / tickCount) * img.height;

          // Tick mark
          ctx.beginPath();
          ctx.moveTo(marginLeft - 5, y);
          ctx.lineTo(marginLeft, y);
          ctx.strokeStyle = '#000000';
          ctx.lineWidth = 1;
          ctx.stroke();

          // Tick label
          const label = formatFreq(freq);
          ctx.fillText(label, marginLeft - 10, y);
        }

        // Frequency axis label
        ctx.save();
        ctx.translate(15, marginTop + img.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.font = 'bold 14px Arial';
        ctx.fillText('Frequency (Hz)', 0, 0);
        ctx.restore();

        // Time axis (bottom)
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        const totalDuration = time_end - time_start;
        const timeTickCount = 6;

        for (let i = 0; i <= timeTickCount; i++) {
          const time = time_start + (i / timeTickCount) * totalDuration;
          const x = marginLeft + (i / timeTickCount) * img.width;

          // Tick mark
          ctx.beginPath();
          ctx.moveTo(x, marginTop + img.height);
          ctx.lineTo(x, marginTop + img.height + 5);
          ctx.stroke();

          // Tick label
          ctx.fillText(formatTime(time), x, marginTop + img.height + 10);
        }

        // Time axis label
        ctx.font = 'bold 14px Arial';
        ctx.fillText('Time (mm:ss)', marginLeft + img.width / 2, marginTop + img.height + 50);
      }

      // Convert canvas to blob
      canvas.toBlob((blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Failed to create image blob'));
        }
      }, 'image/png');
    };

    img.onerror = () => {
      reject(new Error('Failed to load spectrogram image'));
    };

    img.src = spectrogramUrl;
  });
};

/**
 * Modern spectrogram viewer with clean design and abstracted UI elements
 * Aesthetic: Scientific Precision - dark theme, calibrated colors, professional appearance
 */
const SpectrogramViewer = ({
  spectrogramUrl,
  metadata,
  audioCurrentTime,
  clipDuration,
  isLoading = false,
  error = null,
  showMetadata = true,
  className = ''
}) => {
  if (error) {
    return (
      <div className={`spectrogram-viewer loading ${className}`}>
        <div className="loading-content">
          <div className="loading-text error-text">⚠️ {error}</div>
        </div>
      </div>
    );
  }

  if (isLoading || !spectrogramUrl) {
    return (
      <div className={`spectrogram-viewer loading ${className}`}>
        <div className="loading-content">
          <div className="loading-spinner"></div>
          <div className="loading-text">Analyzing Audio...</div>
        </div>
      </div>
    );
  }

  // Calculate playhead position (0-100% of the entire spectrogram context)
  const getPlayheadPosition = () => {
    if (!metadata || clipDuration <= 0) return 0;
    const { time_start, time_end, clip_start } = metadata;
    const totalDuration = time_end - time_start;
    if (totalDuration <= 0) return 0;
    
    // actualTime is the absolute time in the file
    const actualTime = clip_start + audioCurrentTime;
    // position is relative to the spectrogram context [time_start, time_end]
    return Math.min(Math.max(((actualTime - time_start) / totalDuration) * 100, 0), 100);
  };

  const playheadPosition = getPlayheadPosition();

  // Mel scale conversion (HTK version used by librosa by default)
  const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
  const melToHz = (mel) => 700 * (Math.pow(10, mel / 2595) - 1);

  // Generate time ticks (5-7 ticks across timeline)
  const generateTimeTicks = () => {
    if (!metadata) return [];
    const { time_start, time_end, clip_start, clip_end } = metadata;
    const totalDuration = time_end - time_start;
    const tickCount = 6;
    const ticks = [];

    for (let i = 0; i <= tickCount; i++) {
      const time = time_start + (i / tickCount) * totalDuration;
      const position = (i / tickCount) * 100;
      // Use a small epsilon for boundary detection
      const isClipBoundary = Math.abs(time - clip_start) < 0.05 || Math.abs(time - clip_end) < 0.05;

      ticks.push({
        position,
        time,
        label: formatTime(time),
        isBoundary: isClipBoundary
      });
    }

    return ticks;
  };

  // Generate frequency ticks (5 ticks on Y-axis) - Adjusted for Mel or Linear scale
  const generateFreqTicks = () => {
    if (!metadata) return [];
    const { freq_min, freq_max, freq_scale } = metadata;

    const tickCount = 5;
    const ticks = [];

    // Check if we're using mel scale or linear scale
    const isLinear = freq_scale === 'linear';

    if (isLinear) {
      // Linear scale: evenly spaced in Hz
      for (let i = 0; i <= tickCount; i++) {
        const freq = freq_max - (i / tickCount) * (freq_max - freq_min);
        const position = (i / tickCount) * 100;

        ticks.push({
          position,
          freq,
          label: formatFreq(freq)
        });
      }
    } else {
      // Mel scale: evenly spaced in Mel space
      const melMin = hzToMel(freq_min);
      const melMax = hzToMel(freq_max);

      for (let i = 0; i <= tickCount; i++) {
        const mel = melMax - (i / tickCount) * (melMax - melMin);
        const freq = melToHz(mel);
        const position = (i / tickCount) * 100;

        ticks.push({
          position,
          freq,
          label: formatFreq(freq)
        });
      }
    }

    return ticks;
  };

  const timeTicks = generateTimeTicks();
  const freqTicks = generateFreqTicks();

  return (
    <div className={`spectrogram-viewer ${className}`}>
      {/* Header Info Area */}
      {showMetadata && metadata && (
        <div className="spectrogram-header">
          <div className="header-left">
            <div className="header-item">
              <span className="item-label">CLIP RANGE</span>
              <span className="item-value">{formatTime(metadata.clip_start)} — {formatTime(metadata.clip_end)}</span>
            </div>
            <div className="header-item">
              <span className="item-label">DURATION</span>
              <span className="item-value">{(metadata.clip_end - metadata.clip_start).toFixed(2)}s</span>
            </div>
          </div>
          <div className="header-right">
            <div className="header-item">
              <span className="item-label">FREQ RANGE</span>
              <span className="item-value">{formatFreq(metadata.freq_min)} - {formatFreq(metadata.freq_max)}Hz</span>
            </div>
          </div>
        </div>
      )}

      {/* Main Spectrogram Container */}
      <div className="spectrogram-container">
        {/* Frequency Scale (Left) */}
        <div className="frequency-scale">
          <div className="scale-label">Hz</div>
          <div className="scale-ticks">
            {freqTicks.map((tick, idx) => (
              <div
                key={idx}
                className="tick"
                style={{ top: `${tick.position}%` }}
              >
                <span className="tick-mark"></span>
                <span className="tick-label">{tick.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Spectrogram Image Area */}
        <div className="spectrogram-image-wrapper">
          <img
            src={spectrogramUrl}
            alt="Audio Spectrogram"
            className="spectrogram-image"
          />

          {/* Playhead Overlay */}
          {audioCurrentTime >= 0 && clipDuration > 0 && (
            <div
              className="playhead"
              style={{ left: `${playheadPosition}%` }}
            >
              <div className="playhead-line"></div>
              <div className="playhead-head"></div>
            </div>
          )}
        </div>
      </div>

      {/* Time Scale (Bottom) */}
      <div className="time-scale">
        <div className="scale-ticks">
          {timeTicks.map((tick, idx) => (
            <div
              key={idx}
              className={`tick ${tick.isBoundary ? 'boundary' : ''}`}
              style={{ left: `${tick.position}%` }}
            >
              <span className="tick-mark"></span>
              <span className="tick-label">{tick.label}</span>
            </div>
          ))}
        </div>
        <div className="scale-label">Time (mm:ss)</div>
      </div>
    </div>
  );
};

export default SpectrogramViewer;
