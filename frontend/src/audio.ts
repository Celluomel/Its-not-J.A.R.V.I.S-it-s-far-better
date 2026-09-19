export async function listenContinuously(options: {
  blocked: () => boolean;
  onLevel: (level: number) => void;
  onPartial?: (audio: Blob) => Promise<void>;
  onSpeechStart?: () => void;
  onUtterance: (audio: Blob) => void;
  onError: () => void;
}) {
  const { MicVAD, utils } = await import('@ricky0123/vad-web');
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }, video: false });
  const context = new AudioContext({ sampleRate: 16000 });
  let recorder: AudioWorkletNode | undefined;
  let source: MediaStreamAudioSourceNode | undefined;
  let mute: GainNode | undefined;
  const speechChunks: Float32Array[] = [];
  let speechActive = false;
  let partialInFlight: Promise<void> = Promise.resolve();
  let partialBusy = false;
  let partialTimer: ReturnType<typeof setInterval> | undefined;

  const encodeWav = (chunks: Float32Array[]) => {
    const samples = chunks.reduce((total, chunk) => total + chunk.length, 0);
    const buffer = new ArrayBuffer(44 + samples * 2);
    const view = new DataView(buffer);
    const ascii = (offset: number, value: string) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
    ascii(0, 'RIFF'); view.setUint32(4, 36 + samples * 2, true); ascii(8, 'WAVE'); ascii(12, 'fmt ');
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, 16000, true); view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    ascii(36, 'data'); view.setUint32(40, samples * 2, true);
    let offset = 44;
    for (const chunk of chunks) for (const sample of chunk) { view.setInt16(offset, Math.max(-1, Math.min(1, sample)) * 32767, true); offset += 2; }
    return new Blob([buffer], { type: 'audio/wav' });
  };

  let vad: Awaited<ReturnType<typeof MicVAD.new>>;
  try {
    await context.audioWorklet.addModule(`${import.meta.env.BASE_URL}recorder.js`);
    await context.resume();
    source = context.createMediaStreamSource(stream);
    recorder = new AudioWorkletNode(context, 'lumina-recorder');
    mute = context.createGain();
    mute.gain.value = 0;
    source.connect(recorder).connect(mute).connect(context.destination);
    recorder.port.onmessage = ({ data }: MessageEvent<Float32Array>) => {
      if (speechActive) speechChunks.push(data);
    };
    vad = await MicVAD.new({
      model: 'v5', startOnLoad: false,
      baseAssetPath: `${import.meta.env.BASE_URL}voice/`,
      onnxWASMBasePath: `${import.meta.env.BASE_URL}voice/`,
      ortConfig: ort => { ort.env.wasm.numThreads = 1; },
      getStream: async () => stream,
      pauseStream: async () => {},
      resumeStream: async () => stream,
      // A shorter natural endpoint reduces the silence before the response.
      redemptionMs: 500, preSpeechPadMs: 250, minSpeechMs: 250,
      positiveSpeechThreshold: .6, negativeSpeechThreshold: .4,
      submitUserSpeechOnPause: false,
      onFrameProcessed: (_, frame) => {
        if (!options.blocked()) options.onLevel(Math.min(1, Math.sqrt(frame.reduce((sum, x) => sum + x*x, 0)/frame.length)*5));
      },
      onSpeechStart: () => {
        speechActive = true;
        speechChunks.length = 0;
        options.onSpeechStart?.();
      },
      onSpeechEnd: samples => {
        speechActive = false;
        if (partialTimer) clearInterval(partialTimer);
        const finalAudio = new Blob([utils.encodeWAV(samples)], { type: 'audio/wav' });
        // The VAD sample includes its speech padding and is authoritative. Any
        // in-flight preview must finish before the final request uses the audio
        // lock, otherwise the final turn could receive a false 409 busy state.
        void partialInFlight.finally(() => {
          if (!options.blocked()) options.onUtterance(finalAudio);
        });
      },
    });
    if (options.onPartial) {
      partialTimer = setInterval(() => {
        if (!speechActive || options.blocked() || speechChunks.length < 12 || partialBusy) return;
        const snapshot = speechChunks.slice();
        partialBusy = true;
        partialInFlight = options.onPartial!(encodeWav(snapshot)).catch(() => {}).finally(() => { partialBusy = false; });
      }, 900);
    }
  } catch (error) {
    stream.getTracks().forEach(track => track.stop());
    recorder?.disconnect(); source?.disconnect(); mute?.disconnect();
    await context.close();
    throw error;
  }
  let closed = false;
  let updating = false;
  // Keep the detector alive for the whole voice session.  Pausing MicVAD and
  // trying to restart it after TTS can leave some browser/ONNX combinations
  // listening visually but no longer delivering speech-end callbacks.  The
  // blocked gate still drops all echo while PandoraBOX is speaking.
  const sync = async () => {
    if (closed || updating) return;
    updating = true;
    try {
      if (options.blocked()) options.onLevel(0);
      else if (!vad.listening) await vad.start();
    } catch { if (!closed) options.onError(); }
    finally { updating = false; }
  };
  await sync();
  const interval = setInterval(() => void sync(), 100);
  return async () => {
    closed = true;
    clearInterval(interval);
    if (partialTimer) clearInterval(partialTimer);
    speechActive = false;
    stream.getTracks().forEach(track => track.stop());
    while (updating) await new Promise(resolve => setTimeout(resolve, 20));
    await vad.destroy();
    recorder?.port.close(); recorder?.disconnect(); source?.disconnect(); mute?.disconnect();
    await context.close();
    options.onLevel(0);
  };
}

export async function recordMicrophone(onLevel: (level: number) => void) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true }, video: false });
  const context = new AudioContext({ sampleRate: 16000 });
  try {
    await context.audioWorklet.addModule(`${import.meta.env.BASE_URL}recorder.js`);
    await context.resume();
    const source = context.createMediaStreamSource(stream);
    const recorder = new AudioWorkletNode(context, 'lumina-recorder');
    const mute = context.createGain();
    mute.gain.value = 0;
    source.connect(recorder).connect(mute).connect(context.destination);
    const chunks: Float32Array[] = [];
    let last = 0;
    recorder.port.onmessage = ({ data }: MessageEvent<Float32Array>) => {
      chunks.push(data);
      if (performance.now() - last > 80) {
        onLevel(Math.min(1, Math.sqrt(data.reduce((a, b) => a + b * b, 0) / data.length) * 5));
        last = performance.now();
      }
    };
    let stopped = false;
    return async () => {
      if (stopped) return new Blob();
      stopped = true;
      recorder.port.onmessage = null;
      source.disconnect(); recorder.disconnect(); mute.disconnect();
      stream.getTracks().forEach(track => track.stop());
      const sampleRate = context.sampleRate;
      await context.close();
      onLevel(0);
      const samples = chunks.reduce((n, chunk) => n + chunk.length, 0);
      const buffer = new ArrayBuffer(44 + samples * 2);
      const view = new DataView(buffer);
      const ascii = (offset: number, text: string) => [...text].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
      ascii(0, 'RIFF'); view.setUint32(4, 36 + samples * 2, true); ascii(8, 'WAVE'); ascii(12, 'fmt ');
      view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true); view.setUint16(34, 16, true); ascii(36, 'data'); view.setUint32(40, samples * 2, true);
      let offset = 44;
      for (const chunk of chunks) for (const sample of chunk) { view.setInt16(offset, Math.max(-1, Math.min(1, sample)) * 32767, true); offset += 2; }
      return new Blob([buffer], { type: 'audio/wav' });
    };
  } catch (error) {
    stream.getTracks().forEach(track => track.stop());
    await context.close();
    throw error;
  }
}
