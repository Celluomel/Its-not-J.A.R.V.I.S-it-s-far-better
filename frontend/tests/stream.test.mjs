import { test } from 'node:test';
import assert from 'node:assert/strict';
import { appendReasoning, readEvents, visibleText } from '../src/stream.mjs';

test('stream survives single-byte chunks, UTF-8 splits and final line without newline', async () => {
  const expected = [{ type: 'delta', text: 'Bonjour, Frédéric' }, { type: 'done' }];
  const bytes = new TextEncoder().encode(expected.map(JSON.stringify).join('\n'));
  const body = new ReadableStream({ start(controller) { for (const byte of bytes) controller.enqueue(new Uint8Array([byte])); controller.close(); } });
  const received = [];
  for await (const event of readEvents(body)) received.push(event);
  assert.deepEqual(received, expected);
});

test('unfinished reasoning blocks are withheld', () => {
  assert.equal(visibleText('<think>unfinished reasoning'), '');
  assert.equal(visibleText('<think>private</think> Hello'), 'Hello');
});

test('escaped reasoning marker and exact duplicated final answer are cleaned', () => {
  const answer = 'Ah, aller chercher Théodore... Je vais attendre.';
  assert.equal(visibleText(`${answer} \\</think> ${answer}`), answer);
  assert.equal(visibleText(`${answer} ${answer} avec une nuance différente.`), `${answer} ${answer} avec une nuance différente.`);
  assert.equal(visibleText(`A normal answer \\</think>`), 'A normal answer');
  assert.equal(visibleText('<think>private</think> Visible answer'), 'Visible answer');
});

test('reasoning fragments accumulate and punctuation-only fragments do not erase text', () => {
  let text = appendReasoning('', '.');
  assert.equal(text, '');
  text = appendReasoning(text, 'The user is asking about the cause');
  text = appendReasoning(text, ', so I should answer directly');
  text = appendReasoning(text, '.');
  assert.equal(text, 'The user is asking about the cause, so I should answer directly.');
});
