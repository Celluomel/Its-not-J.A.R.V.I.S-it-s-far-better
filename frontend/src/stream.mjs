// Preserve UTF-8 characters and event boundaries across arbitrary network chunks.
export async function* readEvents(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      pending += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let newline;
      while ((newline = pending.indexOf('\n')) !== -1) {
        const line = pending.slice(0, newline).trim();
        pending = pending.slice(newline + 1);
        if (line) yield JSON.parse(line);
      }
      if (done) break;
    }
    if (pending.trim()) yield JSON.parse(pending);
  } finally { reader.releaseLock(); }
}

export function visibleText(text) {
  let clean = String(text || '')
    .replace(/\\+\s*(<\/?(?:think|analysis)>)/gi, '$1')
    .replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, '')
    .replace(/<analysis>[\s\S]*?(?:<\/analysis>|$)/gi, '')
    .replace(/<\/?(?:think|analysis)>/gi, '')
    .trim();
  // Some local models return the final answer twice after their reasoning
  // block. Remove only an exact repeated half, never merely similar prose.
  for (let split = Math.floor(clean.length / 2) - 2; split <= Math.ceil(clean.length / 2) + 2; split += 1) {
    if (split > 0 && clean.slice(0, split).trim() === clean.slice(split).trim()) {
      clean = clean.slice(0, split).trim();
      break;
    }
  }
  return clean;
}

export function appendReasoning(current, fragment) {
  const combined = `${current || ''}${fragment || ''}`.replace(/^[\s.,;:!?…]+/u, '');
  return /[\p{L}\p{N}]/u.test(combined) ? combined : '';
}
