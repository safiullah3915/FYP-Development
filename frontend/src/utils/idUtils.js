export const normalizeId = (id) => {
  if (!id) return id;
  if (typeof id !== 'string') return id;

  // Trim and normalize casing
  const trimmed = id.trim();
  const lower = trimmed.toLowerCase();

  // If already hyphenated UUID (any case), return lowercased form
  if (lower.includes('-')) {
    return lower;
  }

  // If it's a 32-hex string without hyphens, insert hyphens in UUID positions
  const hex32Regex = /^[0-9a-f]{32}$/;
  if (hex32Regex.test(lower)) {
    return `${lower.slice(0, 8)}-${lower.slice(8, 12)}-${lower.slice(12, 16)}-${lower.slice(16, 20)}-${lower.slice(20)}`;
  }

  // Fallback: return lowercased trimmed string
  return lower;
};

export const getStartupDetailPath = (id) => {
  const normalizedId = normalizeId(id);
  // Ensure the id is URL-safe
  const encoded = encodeURIComponent(normalizedId);
  return `/startupdetail/${encoded}`;
};

