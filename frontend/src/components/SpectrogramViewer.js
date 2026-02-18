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
 * Modern spectrogram viewer with clean design and abstracted UI elements
 * Aesthetic: Scientific Precision - dark theme, calibrated colors, professional appearance
 */
const SpectrogramViewer = ({
  spectrogramUrl,
  metadata,
  audioCurrentTime,
  clipDuration,
  isLoading = false,
  showMetadata = true,
  className = ''
}) => {
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

  // Generate frequency ticks (5 ticks on Y-axis) - Adjusted for Mel scale
  const generateFreqTicks = () => {
    if (!metadata) return [];
    const { freq_min, freq_max } = metadata;
    
    const melMin = hzToMel(freq_min);
    const melMax = hzToMel(freq_max);
    
    const tickCount = 5;
    const ticks = [];

    for (let i = 0; i <= tickCount; i++) {
      // Evenly spaced in Mel space because that's how the spectrogram is rendered
      const mel = melMax - (i / tickCount) * (melMax - melMin);
      const freq = melToHz(mel);
      const position = (i / tickCount) * 100;

      ticks.push({
        position,
        freq,
        label: formatFreq(freq)
      });
    }

    return ticks;
  };

  // Generate amplitude/dB gradient stops
  const generateAmplitudeGradient = () => {
    if (!metadata) return [];
    const { db_min, db_max } = metadata;
    const stops = [];
    const stepCount = 8;

    for (let i = 0; i <= stepCount; i++) {
      const db = db_max - (i / stepCount) * (db_max - db_min);
      const position = (i / stepCount) * 100;

      stops.push({
        position,
        db,
        label: `${db.toFixed(0)}`
      });
    }

    return stops;
  };

  const timeTicks = generateTimeTicks();
  const freqTicks = generateFreqTicks();
  const amplitudeStops = generateAmplitudeGradient();

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

        {/* Amplitude Scale (Right) */}
        <div className="amplitude-scale">
          <div className="scale-label">dB</div>
          <div className="amplitude-gradient">
            {amplitudeStops.map((stop, idx) => (
              <div
                key={idx}
                className="amplitude-tick"
                style={{ top: `${stop.position}%` }}
              >
                <span className="tick-label">{stop.label}</span>
                <span className="tick-mark"></span>
              </div>
            ))}
          </div>
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
