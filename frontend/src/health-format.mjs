export function formatHealthTimestamp(key, value) {
  if (!/^(timestamp|fetched_at|last_updated|computed_at|created_at|updated_at)$/.test(key)) return null;
  if (value === null || value === undefined || value === '' || value === 0) return 'Not recorded';
  const date = typeof value === 'number'
    ? new Date(value < 1e12 ? value * 1000 : value)
    : new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toLocaleString();
}
