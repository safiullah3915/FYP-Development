const textField = (id, label, helperText, options = {}) => ({
  id,
  label,
  helperText,
  type: 'text',
  ...options,
});

const textareaField = (id, label, helperText, options = {}) => ({
  id,
  label,
  helperText,
  type: 'textarea',
  ...options,
});

const multiSelectField = (id, label, helperText, choices, options = {}) => ({
  id,
  label,
  helperText,
  type: 'multi-select',
  choices,
  ...options,
});

const checkboxField = (id, label, helperText, choices, options = {}) => ({
  id,
  label,
  helperText,
  type: 'checkbox-group',
  choices,
  ...options,
});

const sliderField = (id, label, helperText, options = {}) => ({
  id,
  label,
  helperText,
  type: 'range',
  ...options,
});

const numberField = (id, label, helperText, options = {}) => ({
  id,
  label,
  helperText,
  type: 'number',
  ...options,
});

export const INVESTOR_SECTOR_OPTIONS = [
  'ai + infrastructure',
  'b2b saas',
  'consumer marketplaces',
  'climate + sustainability',
  'deeptech',
  'devtools',
  'digital health',
  'edtech',
  'fintech',
  'gaming + entertainment',
  'logistics + supply chain',
  'proptech',
  'web3',
];

export const INVESTOR_ROUND_OPTIONS = [
  'pre-seed',
  'seed',
  'series a',
  'series b',
  'growth',
  'secondary',
];

export const INVESTOR_INSTRUMENT_OPTIONS = [
  'equity',
  'safe',
  'convertible note',
  'venture debt',
  'revenue share',
  'spv',
];

export const INVESTOR_STAGE_OPTIONS = [
  'idea',
  'mvp',
  'pre-revenue',
  'early revenue',
  'scaling',
];

export const INVESTOR_GEO_OPTIONS = [
  'north america',
  'latin america',
  'europe',
  'middle east',
  'africa',
  'south asia',
  'southeast asia',
  'australia',
];

export const INVESTOR_VALUE_ADD_OPTIONS = [
  'gtm + enterprise intros',
  'talent + recruiting',
  'product + roadmap support',
  'follow-on capital',
  'regulatory guidance',
  'global expansion',
];

export const INVESTOR_SUPPORT_STYLE_OPTIONS = [
  'hands-on (weekly touchpoints)',
  'strategic (monthly cadences)',
  'lightweight (as needed)',
];

export const INVESTOR_COINVEST_OPTIONS = [
  'solo lead',
  'co-lead',
  'flexible / SPV',
  'follow-on only',
];

export const INVESTOR_CHECK_SIZE_PRESETS = [
  { label: '$50k - $150k', value: { min: 50000, max: 150000 } },
  { label: '$150k - $500k', value: { min: 150000, max: 500000 } },
  { label: '$500k - $1.5M', value: { min: 500000, max: 1500000 } },
  { label: '$1.5M - $5M', value: { min: 1500000, max: 5000000 } },
];

export const INVESTOR_DEFAULT_PROFILE = {
  thesis_summary: '',
  sectors: [],
  stages: [],
  round_types: [],
  instruments: [],
  geographies: [],
  business_models: [],
  check_size: {
    min: null,
    max: null,
    currency: 'USD',
  },
  target_ownership: {
    min_pct: null,
    max_pct: null,
  },
  valuation_caps: {
    post_money_max: null,
    preferred_metric: 'post_money',
  },
  traction: {
    revenue_min: '',
    mrr_min: '',
    users_min: '',
  },
  support_preferences: [],
  support_style: '',
  co_investment: '',
  follow_on_reserve: '',
  decision_speed_days: '',
  notes: '',
};

export const INVESTOR_FORM_SECTIONS = [
  {
    id: 'thesis',
    title: 'Investment Thesis',
    description: 'Set the tone for startups—what themes and superpowers define your thesis?',
    fields: [
      textareaField(
        'thesis_summary',
        'Thesis Summary',
        'Share the 2–3 sentence summary you send to founders.'
      ),
      multiSelectField(
        'sectors',
        'Priority Sectors',
        'Pick the verticals you underwrite with conviction.',
        INVESTOR_SECTOR_OPTIONS,
        { required: true }
      ),
      multiSelectField(
        'business_models',
        'Business Model Comfort Zone',
        'We translate these into tags when building your embeddings.',
        ['b2b', 'b2c', 'marketplace', 'infrastructure', 'hardware', 'consumer subscription']
      ),
    ],
  },
  {
    id: 'rounds',
    title: 'Stage & Round Focus',
    description: 'Help founders know when to loop you in.',
    fields: [
      multiSelectField(
        'stages',
        'Company Stage',
        'Your sweet spot across MVP → scaling.',
        INVESTOR_STAGE_OPTIONS,
        { required: true }
      ),
      multiSelectField(
        'round_types',
        'Round Types',
        'Dial in where you typically write checks.',
        INVESTOR_ROUND_OPTIONS,
        { required: true }
      ),
      multiSelectField(
        'instruments',
        'Instruments',
        'Founders see this inside the deal room when prepping for you.',
        INVESTOR_INSTRUMENT_OPTIONS,
        { required: true }
      ),
    ],
  },
  {
    id: 'capital',
    title: 'Capital Deployment',
    description: 'Ticket sizing, currency, and ownership targets drive better ranking.',
    fields: [
      sliderField(
        'check_size',
        'Typical Check Size',
        'Drag the handles or pick a preset to lock in your range.',
        {
          presetOptions: INVESTOR_CHECK_SIZE_PRESETS,
          min: 25000,
          max: 5000000,
          step: 5000,
          unit: 'USD',
          required: true,
        }
      ),
      numberField(
        'target_ownership.min_pct',
        'Target Ownership (Min %)',
        'What floor do you set for meaningful positions?',
        { min: 1, max: 40, required: true }
      ),
      numberField(
        'target_ownership.max_pct',
        'Target Ownership (Max %)',
        'Upper bound for check + reserve strategy.',
        { min: 1, max: 60 }
      ),
    ],
  },
  {
    id: 'geos',
    title: 'Geographies & Traction Guardrails',
    description: 'We use this to filter the candidate pool before scoring.',
    fields: [
      multiSelectField(
        'geographies',
        'Regions You Invest In',
        'Your pipeline will bias toward these regions.',
        INVESTOR_GEO_OPTIONS,
        { required: true }
      ),
      numberField(
        'traction.revenue_min',
        'Minimum Revenue ($ ARR)',
        'Use 0 for truly pre-revenue bets.',
        { min: 0 }
      ),
      numberField(
        'traction.mrr_min',
        'Minimum MRR',
        'Optional but helpful for SaaS filters.',
        { min: 0 }
      ),
      numberField(
        'traction.users_min',
        'Minimum Active Users',
        'Great proxy for consumer traction.',
        { min: 0 }
      ),
      numberField(
        'valuation_caps.post_money_max',
        'Ceiling Post-Money ($M)',
        'We avoid surfacing startups priced above this.',
        { min: 1 }
      ),
    ],
  },
  {
    id: 'support',
    title: 'Support & Collaboration',
    description: 'Signal how you plug in so matchmaking is honest.',
    fields: [
      checkboxField(
        'support_preferences',
        'Value-Add Focus',
        'Choose everything you actively help with after wiring.',
        INVESTOR_VALUE_ADD_OPTIONS
      ),
      multiSelectField(
        'support_style',
        'Engagement Style',
        'Founders can filter by the operating cadence that fits them best.',
        INVESTOR_SUPPORT_STYLE_OPTIONS,
        { required: true, maxSelect: 1 }
      ),
      multiSelectField(
        'co_investment',
        'Co-investment Preferences',
        'We pair you with founders who want the same syndication plan.',
        INVESTOR_COINVEST_OPTIONS,
        { required: true, maxSelect: 1 }
      ),
      textField(
        'decision_speed_days',
        'Decision Speed (days)',
        'Typical turnaround from first call → term sheet.',
        { inputMode: 'numeric' }
      ),
      textareaField(
        'notes',
        'Internal Notes',
        'Optional context (fund size, reserves, strategic LPs).',
        { maxLength: 600 }
      ),
    ],
  },
];

export const INVESTOR_REQUIRED_FIELDS = [
  'sectors',
  'stages',
  'round_types',
  'instruments',
  'geographies',
  'check_size',
  'support_style',
  'co_investment',
];


