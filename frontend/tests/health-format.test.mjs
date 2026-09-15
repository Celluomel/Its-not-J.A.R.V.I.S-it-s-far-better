import test from 'node:test';
import assert from 'node:assert/strict';
import { formatHealthTimestamp } from '../src/health-format.mjs';

test('health timestamps accept seconds and milliseconds without changing cycle counters', () => {
  const seconds = 1788903019;
  assert.equal(formatHealthTimestamp('last_updated', seconds), new Date(seconds * 1000).toLocaleString());
  assert.equal(formatHealthTimestamp('timestamp', seconds * 1000), formatHealthTimestamp('last_updated', seconds));
  assert.equal(formatHealthTimestamp('slow_cycle', seconds), null);
  assert.equal(formatHealthTimestamp('last_updated', 0), 'Not recorded');
  assert.equal(formatHealthTimestamp('timestamp', 'invalid'), null);
});
