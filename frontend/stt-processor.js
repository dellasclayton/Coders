/**
 * stt-processor.js - AudioWorklet Processor for STT
 * Downsamples audio from source rate (48kHz) to 16kHz PCM16
 */

class STTProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();

    const opts = options.processorOptions || {};
    this.targetSampleRate = opts.targetSampleRate || 16000;
    this.sourceSampleRate = sampleRate; // Global in AudioWorklet scope
    this.ratio = this.sourceSampleRate / this.targetSampleRate;

    // Accumulator for downsampled samples
    this.buffer = [];

    // Fractional sample position for accurate resampling
    this.sampleIndex = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const inputData = input[0];
    const inputLength = inputData.length;

    // Downsample using linear interpolation
    while (this.sampleIndex < inputLength) {
      const index = Math.floor(this.sampleIndex);
      const frac = this.sampleIndex - index;

      // Linear interpolation between samples
      let sample;
      if (index + 1 < inputLength) {
        sample = inputData[index] * (1 - frac) + inputData[index + 1] * frac;
      } else {
        sample = inputData[index];
      }

      // Clamp and convert to 16-bit integer
      sample = Math.max(-1, Math.min(1, sample));
      this.buffer.push(Math.round(sample * 32767));

      this.sampleIndex += this.ratio;
    }

    // Adjust index for next frame
    this.sampleIndex -= inputLength;

    // Send 320-sample chunks (20ms at 16kHz)
    while (this.buffer.length >= 320) {
      const chunk = new Int16Array(this.buffer.splice(0, 320));
      this.port.postMessage(chunk.buffer, [chunk.buffer]);
    }

    return true;
  }
}

registerProcessor('stt-processor', STTProcessor);
