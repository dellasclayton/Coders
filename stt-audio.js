// ============================================
// STT AUDIO - Microphone Capture Module
// ============================================

// ========== State Variables ==========
let sttAudioContext = null;
let stream = null;
let isRecording = false;

// STT (microphone capture)
let sttProcessor = null;
let sourceNode = null;

// Callbacks
let onAudioData = null;  // Called with PCM16 data for STT
let onError = null;      // Called with error messages

// ========== Audio Context Management ==========
function initializeAudioContext() {
  if (!sttAudioContext || sttAudioContext.state === 'closed') {
    sttAudioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 48000
    });
  }
  return sttAudioContext;
}

// ========== Microphone Capture ==========
/**
 * Start microphone capture for STT
 * Outputs PCM16 @ 16kHz via callback
 */
async function startMicrophone(callback) {
  if (isRecording) {
    console.log('STT Audio: Already recording');
    return;
  }

  // Store callback
  if (typeof callback === 'function') {
    onAudioData = callback;
  }

  try {
    // Initialize audio context
    const ctx = initializeAudioContext();
    await ctx.resume();

    // Get microphone stream
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });

    // Load STT processor
    await ctx.audioWorklet.addModule('./stt-processor.js');

    // Create audio nodes
    sourceNode = ctx.createMediaStreamSource(stream);
    sttProcessor = new AudioWorkletNode(ctx, 'stt-processor', {
      processorOptions: {
        targetSampleRate: 16000
      }
    });

    // Handle STT data
    sttProcessor.port.onmessage = (event) => {
      if (onAudioData) {
        onAudioData(event.data);
      }
    };

    // Connect audio graph
    sourceNode.connect(sttProcessor);

    isRecording = true;
    console.log('STT Audio: Recording started');

  } catch (error) {
    console.error('STT Audio: Failed to start recording:', error);

    // Cleanup on error
    stopMicrophone();

    if (onError) {
      onError('Failed to access microphone: ' + error.message);
    } else {
      throw error;
    }
  }
}

/**
 * Stop microphone capture
 */
function stopMicrophone() {
  if (!isRecording) {
    console.log('STT Audio: Not recording');
    return;
  }

  // Cleanup audio nodes
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }

  if (sttProcessor) {
    sttProcessor.disconnect();
    sttProcessor = null;
  }

  // Stop microphone stream
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  isRecording = false;
  console.log('STT Audio: Recording stopped');
}

// ========== State Getters ==========
function getMicrophoneState() {
  return isRecording;
}

function getStatus() {
  return {
    isRecording,
    hasAudioContext: !!sttAudioContext,
    audioContextState: sttAudioContext ? sttAudioContext.state : null,
    contextSampleRate: sttAudioContext ? sttAudioContext.sampleRate : null
  };
}

// ========== Callback Management ==========
function setAudioDataCallback(callback) {
  if (typeof callback === 'function') {
    onAudioData = callback;
  }
}

function setErrorCallback(callback) {
  if (typeof callback === 'function') {
    onError = callback;
  }
}

// ========== Cleanup ==========
function cleanup() {
  stopMicrophone();

  if (sttAudioContext && sttAudioContext.state !== 'closed') {
    sttAudioContext.close();
  }

  sttAudioContext = null;
  onAudioData = null;
  onError = null;

  console.log('STT Audio: Cleanup complete');
}

// ========== Public API Export ==========
window.STTAudio = {
  // Microphone control
  startMicrophone,
  stopMicrophone,

  // State
  getMicrophoneState,
  getStatus,

  // Callbacks
  setAudioDataCallback,
  setErrorCallback,

  // Cleanup
  cleanup
};
