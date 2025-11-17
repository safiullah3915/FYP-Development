export const normalizeId = (id) => {
  if (!id) return id;
  if (typeof id !== 'string') return id;

  const trimmed = id.trim();
  if (trimmed.includes('-')) {
    return trimmed;
  }

  const hex32Regex = /^[0-9a-fA-F]{32}$/;
  if (hex32Regex.test(trimmed)) {
    return `${trimmed.slice(0, 8)}-${trimmed.slice(8, 12)}-${trimmed.slice(12, 16)}-${trimmed.slice(16, 20)}-${trimmed.slice(20)}`;
  }

  return trimmed;
};

export const getStartupDetailPath = (id) => {
  const normalizedId = normalizeId(id);
  return `/startupdetail/${normalizedId}`;
};

