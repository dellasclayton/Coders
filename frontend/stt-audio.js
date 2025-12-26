/**
 * stt-audio.js - Speech-to-Text Audio Capture Module
 * Captures microphone audio and streams it to the server via WebSocket
 */

import * as websocket from './websocket.js';

// ============================================
// STATE
// ============================================
let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let isRecording = false;
let isInitialized = false;

// Callbacks
let stateChangeCallback = null;
let currentState = 'idle'; // 'idle' | 'listening'

// ============================================
// STATE MANAGEMENT
// ============================================

/**
 * Update state and notify listeners
 * @param {string} newState
 */
function setState(newState) {
  if (newState !== currentState) {
    const oldState = currentState;
    currentState = newState;
    if (stateChangeCallback) {
      stateChangeCallback(newState, oldState);
    }
  }
}

/**
 * Register callback for state changes
 * @param {Function} callback - Called with (newState, oldState)
 */
export function onStateChange(callback) {
  if (typeof callback === 'function') {
    stateChangeCallback = callback;
  }
}

// ============================================
// AUDIO CONTEXT MANAGEMENT
// ============================================

/**
 * Initialize audio context (must be called from user gesture)
 * @returns {Promise<boolean>} Success status
 */
export async function initAudioCapture() {
  if (isInitialized) {
    return true;
  }

  try {
    // Create audio context at 48kHz (will be downsampled to 16kHz by processor)
    audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 48000
    });

    // Resume context (required after user gesture)
    await audioContext.resume();

    // Pre-load the AudioWorklet processor
    await audioContext.audioWorklet.addModule('./stt-processor.js');

    isInitialized = true;
    console.log('[STT Audio] Initialized');
    return true;

  } catch (error) {
    console.error('[STT Audio] Failed to initialize:', error);
    return false;
  }
}

// ============================================
// RECORDING CONTROL
// ============================================

/**
 * Start recording from microphone
 * Streams PCM16 audio to server via WebSocket
 */
export async function startRecording() {
  if (isRecording) {
    console.log('[STT Audio] Already recording');
    return;
  }

  // Ensure initialized
  if (!isInitialized) {
    const success = await initAudioCapture();
    if (!success) {
      console.error('[STT Audio] Cannot start - initialization failed');
      return;
    }
  }

  try {
    // Resume audio context if suspended
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }

    // Get microphone access
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });

    // Create audio nodes
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    processorNode = new AudioWorkletNode(audioContext, 'stt-processor', {
      processorOptions: {
        targetSampleRate: 16000
      }
    });

    // Handle audio data from processor - send to server
    processorNode.port.onmessage = (event) => {
      if (isRecording && websocket.isConnected()) {
        websocket.sendAudio(event.data);
      }
    };

    // Connect: microphone -> processor
    sourceNode.connect(processorNode);

    // Notify server that we're starting to listen
    websocket.startListening();

    isRecording = true;
    setState('listening');
    console.log('[STT Audio] Recording started');

  } catch (error) {
    console.error('[STT Audio] Failed to start recording:', error);
    cleanup();
  }
}

/**
 * Stop recording
 */
export function stopRecording() {
  if (!isRecording) {
    return;
  }

  // Notify server that we're stopping
  websocket.stopListening();

  cleanup();

  isRecording = false;
  setState('idle');
  console.log('[STT Audio] Recording stopped');
}

/**
 * Clean up audio resources (but keep context for reuse)
 */
function cleanup() {
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }

  if (processorNode) {
    processorNode.disconnect();
    processorNode = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }
}

// ============================================
// STATE GETTERS
// ============================================

/**
 * Check if currently recording
 * @returns {boolean}
 */
export function getRecordingState() {
  return isRecording;
}

/**
 * Get detailed status
 * @returns {object}
 */
export function getStatus() {
  return {
    isRecording,
    isInitialized,
    audioContextState: audioContext?.state || null,
    currentState
  };
}

// ============================================
// FULL CLEANUP
// ============================================

/**
 * Full cleanup including audio context
 * Call when leaving the page
 */
export function destroy() {
  stopRecording();

  if (audioContext && audioContext.state !== 'closed') {
    audioContext.close();
  }

  audioContext = null;
  isInitialized = false;
  stateChangeCallback = null;

  console.log('[STT Audio] Destroyed');
}
