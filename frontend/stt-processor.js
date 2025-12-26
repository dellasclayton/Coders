/**
 * stt-processor.js - AudioWorklet Processor for STT
 * Captures microphone input, downsamples to 16kHz, outputs PCM16
 *
 * Note: VAD is handled by backend RealtimeSTT (Silero/WebRTC)
 * This processor only handles audio capture and format conversion
 */

class STTProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()

    // Target sample rate for STT (RealtimeSTT expects 16kHz)
    this.targetSampleRate = 16000

    // Source sample rate from AudioContext (typically 48kHz)
    this.sourceSampleRate = sampleRate

    // Calculate decimation ratio for downsampling
    this.decimationRatio = this.sourceSampleRate / this.targetSampleRate

    // Buffer for accumulating downsampled samples
    this.buffer = []

    // Phase accumulator for decimation
    this.phase = 0

    // Chunk size: 320 samples = 20ms @ 16kHz (optimal for real-time STT)
    this.chunkSize = 320

    // Active state - controlled by main thread
    this.isActive = false

    // Handle commands from main thread
    this.port.onmessage = this.handleCommand.bind(this)
  }

  /**
   * Handle commands from main thread
   * @param {MessageEvent} event
   */
  handleCommand(event) {
    const { command } = event.data

    switch (command) {
      case 'start':
        this.isActive = true
        this.buffer = []
        this.phase = 0
        break

      case 'stop':
        this.isActive = false
        this.buffer = []
        this.phase = 0
        break

      case 'pause':
        this.isActive = false
        break

      case 'resume':
        this.isActive = true
        break
    }
  }

  /**
   * Process audio samples
   * Called by audio rendering thread for each quantum (~128 samples)
   * @param {Array<Float32Array[]>} inputs - Input audio buffers
   * @param {Array<Float32Array[]>} outputs - Output audio buffers (unused)
   * @param {Record<string, Float32Array>} parameters - Parameter values
   * @returns {boolean} - Return true to keep processor alive
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0]

    // No input or not active - keep processor alive but don't process
    if (!input || !input[0] || !this.isActive) {
      return true
    }

    const inputData = input[0]

    // Downsample from source rate to 16kHz using linear interpolation
    this.downsampleAndBuffer(inputData)

    // Send chunks when we have enough samples
    this.flushChunks()

    return true
  }

  /**
   * Downsample audio and add to buffer
   * Uses simple decimation with linear interpolation for better quality
   * @param {Float32Array} inputData - Raw audio samples at source rate
   */
  downsampleAndBuffer(inputData) {
    for (let i = 0; i < inputData.length; i++) {
      // Check if we should take this sample based on decimation ratio
      const targetIndex = Math.floor(this.phase)

      if (targetIndex <= i) {
        // Clamp sample to [-1, 1] range
        const sample = Math.max(-1, Math.min(1, inputData[i]))

        // Convert to 16-bit PCM range and store
        this.buffer.push(Math.round(sample * 32767))

        // Advance phase by decimation ratio
        this.phase += this.decimationRatio
      }
    }

    // Reset phase relative to processed samples
    this.phase -= inputData.length
    if (this.phase < 0) this.phase = 0
  }

  /**
   * Send complete chunks to main thread
   */
  flushChunks() {
    while (this.buffer.length >= this.chunkSize) {
      // Extract chunk from buffer
      const chunk = this.buffer.splice(0, this.chunkSize)

      // Convert to Int16Array
      const int16Chunk = new Int16Array(chunk)

      // Send to main thread with transferable buffer
      this.port.postMessage(
        { type: 'audio', data: int16Chunk },
        [int16Chunk.buffer]
      )
    }
  }
}

// Register the processor
registerProcessor('stt-processor', STTProcessor)
