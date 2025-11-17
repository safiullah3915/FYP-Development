// Canonical investor preference schema used by onboarding + API payloads.
// Keep this synced with the backend serializer mapping.

export const INVESTOR_DEFAULTS = {
  thesis_summary: '',
  sectors: [],
  stages: [],
  round_types: [],
  instruments: [],
  geographies: [],
  business_models: [],
  check_size: {
    min: '',
    max: '',
    currency: 'USD',
  },
  target_ownership: {
    min_pct: '',
    max_pct: '',
  },
  valuation_caps: {
    post_money_max: '',
    revenue_multiple_cap: '',
    ebitda_multiple_cap: '',
  },
  traction: {
    arr_min: '',
    revenue_min: '',
    users_min: '',
    growth_min_pct: '',
  },
  support_preferences: [],
  collaboration_style: '',
  lead_preference: 'lead-or-co',
  co_investor_profile: [],
  decision_speed: 'standard',
};

export const INVESTOR_SECTOR_OPTIONS = [
  'fintech',
  'climate',
  'enterprise-saas',
  'future-of-work',
  'healthtech',
  'bio',
  'consumer',
  'marketplaces',
  'devtools',
  'supply-chain',
  'mobility',
  'web3',
  'ai-infrastructure',
  'cybersecurity',
];

export const INVESTOR_STAGE_OPTIONS = [
  'pre-seed',
  'seed',
  'post-seed',
  'series-a',
  'series-b',
  'growth',
];

export const INVESTOR_ROUND_TYPES = [
  'equity',
  'safes',
  'convertible-notes',
  'secondaries',
  'spvs',
  'credit',
];

export const INVESTOR_INSTRUMENT_OPTIONS = [
  'priced-round',
  'safe',
  'convertible-note',
  'venture-debt',
  'revenue-based-financing',
  'secondary-purchase',
];

export const INVESTOR_BUSINESS_MODELS = [
  'b2b-saas',
  'b2c-subscription',
  'b2b2c',
  'marketplace-take-rate',
  'usage-based',
  'hardware-enabled',
  'fintech-origination',
];

export const INVESTOR_VALUE_ADD_OPTIONS = [
  'go-to-market',
  'talent',
  'capital-planning',
  'introductions',
  'follow-on-support',
  'board-seat',
];

export const INVESTOR_SUPPORT_STYLES = [
  'hands-on',
  'strategic-coach',
  'light-touch',
];

export const INVESTOR_DECISION_SPEED = [
  { value: 'fast', label: 'Move Fast (<2 weeks)' },
  { value: 'standard', label: 'Standard (3-5 weeks)' },
  { value: 'deliberate', label: 'Deliberate (6+ weeks)' },
];

export const INVESTOR_LEAD_PREFS = [
  { value: 'lead', label: 'Prefer to lead rounds' },
  { value: 'lead-or-co', label: "Happy to lead or co-lead" },
  { value: 'follow', label: 'Prefer to follow strong leads' },
];

export const INVESTOR_SECTIONS = [
  {
    id: 'thesis_summary',
    title: 'Investment Thesis',
    description: 'One-liner that explains what you invest in and why you win deals.',
    type: 'textarea',
    required: true,
  },
  {
    id: 'sectors',
    title: 'Sector Focus',
    description: 'Pick all categories where you actively lead or co-lead investments.',
    type: 'chips',
    options: INVESTOR_SECTOR_OPTIONS,
    required: true,
  },
  {
    id: 'business_models',
    title: 'Business Models',
    description: 'Select the revenue motions or unit economics you grok best.',
    type: 'chips',
    options: INVESTOR_BUSINESS_MODELS,
    required: true,
  },
  {
    id: 'stages',
    title: 'Stage Appetite',
    description: 'Where do you underwrite conviction quickest?',
    type: 'chips',
    options: INVESTOR_STAGE_OPTIONS,
    required: true,
  },
  {
    id: 'round_types',
    title: 'Round / Instrument',
    description: 'What structures do you usually invest through?',
    type: 'chips',
    options: INVESTOR_ROUND_TYPES,
    required: true,
  },
  {
    id: 'instruments',
    title: 'Capital Instruments',
    description: 'Select all instruments you can deploy.',
    type: 'chips',
    options: INVESTOR_INSTRUMENT_OPTIONS,
    required: true,
  },
  {
    id: 'check_size',
    title: 'Check Size & Currency',
    description: 'Share your typical initial check window (min / max).',
    type: 'range',
    required: true,
  },
  {
    id: 'target_ownership',
    title: 'Ownership Goals',
    description: 'Tell founders how much of the round you try to own.',
    type: 'range-percentage',
    required: true,
  },
  {
    id: 'valuation_caps',
    title: 'Valuation Guardrails',
    description: 'List the point where you tap out on price (post-money or multiples).',
    type: 'key-value',
    required: true,
  },
  {
    id: 'traction',
    title: 'Traction Baselines',
    description: 'Minimum proof-points you need to start a deal (ARR, revenue, users).',
    type: 'key-value',
    required: true,
  },
  {
    id: 'geographies',
    title: 'Geography',
    description: 'Regions where you can lead or support founders.',
    type: 'chips',
    options: ['US', 'Canada', 'Latin America', 'Europe', 'Africa', 'MENA', 'India', 'SEA', 'ANZ'],
    required: true,
  },
  {
    id: 'support_preferences',
    title: 'Value-Add',
    description: 'What tangible help can founders expect post-check?',
    type: 'chips',
    options: INVESTOR_VALUE_ADD_OPTIONS,
    required: true,
  },
  {
    id: 'collaboration_style',
    title: 'Collaboration Style',
    description: 'Describe how you partner with founding teams.',
    type: 'select',
    options: INVESTOR_SUPPORT_STYLES,
    required: true,
  },
  {
    id: 'lead_preference',
    title: 'Lead / Follow',
    description: 'Clarify whether you lead, co-lead, or follow on deals.',
    type: 'select',
    options: INVESTOR_LEAD_PREFS,
    required: true,
  },
  {
    id: 'co_investor_profile',
    title: 'Ideal Co-Investors',
    description: 'Optional: funds or angels you like to syndicate with.',
    type: 'chips',
    options: ['sector-specialists', 'angels', 'family-offices', 'emerging-managers', 'tier-1-funds'],
    required: true,
  },
  {
    id: 'decision_speed',
    title: 'Decision Speed',
    description: 'Set expectations on how quickly you can issue a term sheet.',
    type: 'select',
    options: INVESTOR_DECISION_SPEED,
    required: true,
  },
];


