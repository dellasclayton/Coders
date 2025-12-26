/**
 * STT AudioWorklet Processor for microphone capture
 * Converts microphone input to PCM16 @ 16kHz for speech recognition
 */

class STTProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        
        const opts = options.processorOptions || {};
        this.targetSampleRate = opts.targetSampleRate || 16000;
        this.sourceSampleRate = sampleRate;
        this.decimationRatio = this.sourceSampleRate / this.targetSampleRate;
        
        // Buffer for downsampling
        this.buffer = [];
        this.phase = 0;
    }
    
    process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true;
        
        const inputData = input[0];
        
        // Downsample from source rate to target rate (16kHz)
        for (let i = 0; i < inputData.length; i++) {
            const sample = Math.max(-1, Math.min(1, inputData[i]));
            
            // Simple decimation - take every Nth sample
            if (Math.floor(this.phase) === i) {
                this.buffer.push(sample * 0x7FFF); // Convert to 16-bit range
                this.phase += this.decimationRatio;
            }
        }
        
        // Send chunks of 320 samples (20ms @ 16kHz)
        while (this.buffer.length >= 320) {
            const chunk = new Int16Array(this.buffer.splice(0, 320));
            this.port.postMessage(chunk.buffer);
        }
        
        // Reset phase for next frame
        this.phase = this.phase % inputData.length;
        
        return true;
    }
}

registerProcessor('stt-processor', STTProcessor);