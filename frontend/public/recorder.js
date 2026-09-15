class Recorder extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0]?.[0];
    if (input) this.port.postMessage(input.slice());
    return true;
  }
}
registerProcessor('lumina-recorder', Recorder);
