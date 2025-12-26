/**
 * stt-audio.js - Audio Input Capture for STT
 * Handles microphone capture and audio streaming to server
 *
 * Note: VAD is handled by backend RealtimeSTT (Silero/WebRTC)
 * This module continuously streams audio when active
 */

import * as websocket from './websocket.js'

// ============================================
// STATE
// ============================================
const state = {
  audioContext: null,
  mediaStream: null,
  workletNode: null,
  sourceNode: null,
  status: 'idle', // 'idle' | 'recording' | 'paused'
  isTTSPlaying: false,
  stateListeners: new Set(),
}

// ============================================
// CONFIGURATION
// ============================================
const config = {
  // Use browser's native sample rate (typically 48kHz)
  // Processor handles downsampling to 16kHz for STT
  sampleRate: 48000,
  channelCount: 1,
}

// ============================================
// INITIALIZATION
// ============================================

/**
 * Initialize audio capture system
 * @returns {Promise<boolean>} - True if initialized successfully
 */
export async function initAudioCapture() {
  try {
    // Request microphone access
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: config.channelCount,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      }
    })

    // Create audio context at native sample rate
    state.audioContext = new AudioContext({
      sampleRate: config.sampleRate,
    })

    // Load AudioWorklet processor
    const processorUrl = new URL('./stt-processor.js', import.meta.url)
    await state.audioContext.audioWorklet.addModule(processorUrl)

    // Create worklet node
    state.workletNode = new AudioWorkletNode(
      state.audioContext,
      'stt-processor'
    )

    // Connect audio pipeline
    state.sourceNode = state.audioContext.createMediaStreamSource(state.mediaStream)
    state.sourceNode.connect(state.workletNode)

    // Handle audio data from worklet
    state.workletNode.port.onmessage = handleWorkletMessage

    console.log('[STT] Audio capture initialized')
    return true

  } catch (error) {
    console.error('[STT] Failed to initialize audio capture:', error)
    return false
  }
}

/**
 * Check if audio is initialized
 * @returns {boolean}
 */
export function isInitialized() {
  return state.audioContext !== null && state.workletNode !== null
}

// ============================================
// RECORDING CONTROL
// ============================================

/**
 * Start recording - begins streaming audio to server
 */
export function startRecording() {
  if (!isInitialized()) {
    console.warn('[STT] Not initialized')
    return
  }

  if (state.status === 'recording') {
    return
  }

  // Resume audio context if suspended
  if (state.audioContext.state === 'suspended') {
    state.audioContext.resume()
  }

  // If TTS is playing, go to paused state and wait
  if (state.isTTSPlaying) {
    setStatus('paused')
    console.log('[STT] TTS playing - will start when finished')
    return
  }

  // Enable mic track (in case it was muted)
  setMicEnabled(true)

  // Start the processor
  state.workletNode.port.postMessage({ command: 'start' })

  // Tell server we're starting
  websocket.startListening()

  setStatus('recording')
  console.log('[STT] Recording started')
}

/**
 * Stop recording completely
 */
export function stopRecording() {
  if (!isInitialized()) return

  // Stop the processor
  state.workletNode.port.postMessage({ command: 'stop' })

  // Tell server we're stopping
  websocket.stopListening()

  setStatus('idle')
  console.log('[STT] Recording stopped')
}

/**
 * Check if currently recording
 * @returns {boolean}
 */
export function isRecording() {
  return state.status === 'recording'
}

/**
 * Check if active (recording or paused waiting for TTS)
 * @returns {boolean}
 */
export function isActive() {
  return state.status === 'recording' || state.status === 'paused'
}

/**
 * Get current status
 * @returns {string}
 */
export function getStatus() {
  return state.status
}

// ============================================
// TTS COORDINATION (Echo Prevention)
// ============================================

/**
 * Set TTS playing state - handles pause/resume and echo prevention
 * Called by tts-audio.js when playback starts/stops
 * @param {boolean} isPlaying
 */
export function setTTSPlaying(isPlaying) {
  state.isTTSPlaying = isPlaying

  if (isPlaying) {
    // TTS starting - pause recording and mute mic to prevent echo
    if (state.status === 'recording') {
      // Pause the processor
      state.workletNode?.port.postMessage({ command: 'pause' })

      // Mute mic track to prevent TTS audio feedback
      setMicEnabled(false)

      setStatus('paused')
      console.log('[STT] Paused for TTS (mic muted)')
    }
  } else {
    // TTS finished - resume recording if we were paused
    if (state.status === 'paused') {
      // Re-enable mic track
      setMicEnabled(true)

      // Resume the processor
      state.workletNode?.port.postMessage({ command: 'resume' })

      setStatus('recording')
      console.log('[STT] Resumed after TTS (mic enabled)')
    }
  }
}

/**
 * Enable/disable microphone track (for echo prevention)
 * @param {boolean} enabled
 */
function setMicEnabled(enabled) {
  if (state.mediaStream) {
    state.mediaStream.getAudioTracks().forEach(track => {
      track.enabled = enabled
    })
  }
}

// ============================================
// WORKLET MESSAGE HANDLING
// ============================================

/**
 * Handle messages from AudioWorklet processor
 * @param {MessageEvent} event
 */
function handleWorkletMessage(event) {
  const { type, data } = event.data

  if (type === 'audio') {
    // data is Int16Array - send to server as binary
    websocket.sendAudio(data.buffer)
  }
}

// ============================================
// STATE MANAGEMENT
// ============================================

/**
 * Set status and notify listeners
 * @param {string} newStatus
 */
function setStatus(newStatus) {
  const oldStatus = state.status
  state.status = newStatus

  if (oldStatus !== newStatus) {
    notifyStateListeners(newStatus, oldStatus)
  }
}

/**
 * Register state change listener
 * @param {Function} handler - Callback(newStatus, oldStatus)
 * @returns {Function} - Unsubscribe function
 */
export function onStateChange(handler) {
  state.stateListeners.add(handler)
  return () => state.stateListeners.delete(handler)
}

/**
 * Notify all state listeners
 * @param {string} newStatus
 * @param {string} oldStatus
 */
function notifyStateListeners(newStatus, oldStatus) {
  state.stateListeners.forEach(handler => {
    try {
      handler(newStatus, oldStatus)
    } catch (error) {
      console.error('[STT] State listener error:', error)
    }
  })
}

// ============================================
// CLEANUP
// ============================================

/**
 * Cleanup audio resources
 */
export function cleanup() {
  stopRecording()

  if (state.sourceNode) {
    state.sourceNode.disconnect()
    state.sourceNode = null
  }

  if (state.workletNode) {
    state.workletNode.disconnect()
    state.workletNode = null
  }

  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach(track => track.stop())
    state.mediaStream = null
  }

  if (state.audioContext) {
    state.audioContext.close()
    state.audioContext = null
  }

  console.log('[STT] Cleaned up')
}
