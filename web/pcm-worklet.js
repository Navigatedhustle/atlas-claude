// Downsample mic audio to 16 kHz PCM16 and post 80 ms frames to the main thread.
class PCMWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.buf = [];
    this.pos = 0;
    this.port.onmessage = (e) => { if (e.data === 'flush') this.flush(true); };
  }
  flush(final) {
    if (!this.buf.length && !final) return;
    const out = new Int16Array(this.buf.length);
    for (let i = 0; i < this.buf.length; i++) {
      const s = Math.max(-1, Math.min(1, this.buf[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    this.port.postMessage({ pcm: out.buffer, final: !!final }, [out.buffer]);
    this.buf = [];
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    // simple averaging decimator
    for (let i = 0; i < ch.length; i++) {
      this.acc = (this.acc || 0) + ch[i];
      this.n = (this.n || 0) + 1;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        this.buf.push(this.acc / this.n);
        this.acc = 0; this.n = 0;
      }
    }
    if (this.buf.length >= 1280) {
      const out = new Int16Array(this.buf.length);
      let peak = 0;
      for (let i = 0; i < this.buf.length; i++) {
        const s = Math.max(-1, Math.min(1, this.buf[i]));
        out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        peak = Math.max(peak, Math.abs(s));
      }
      this.port.postMessage({ pcm: out.buffer, peak }, [out.buffer]);
      this.buf = [];
    }
    return true;
  }
}
registerProcessor('pcm-worklet', PCMWorklet);
