export type Project = {
  id: string;
  title: string;
  description: string;
  archived: boolean;
};
export type Document = {
  id: string;
  filename: string;
  page_count: number;
  rendered_pages: number;
  approved_pages: number;
  geometry_pages: number;
  split: string;
  group: string;
  status: string;
  error: string | null;
};
export type Page = {
  id: string;
  number: number;
  width: number;
  height: number;
  document_id: string;
  current_annotation_id: string | null;
};
export type Block = {
  id: string;
  type: string;
  content: string;
  bbox: number[] | null;
};
export type OCRResult = {
  id: string;
  engine: string;
  markdown: string;
  blocks: Block[];
  status: string;
  error: string | null;
  created_at: string;
  seconds: number;
};
export type Annotation = {
  id: string;
  markdown: string;
  blocks: Block[];
  source_result_id: string | null;
  text_approved: boolean;
  geometry_approved: boolean;
  rejected: boolean;
  note: string;
  created_at: string;
};
export type PageDetails = {
  page: Page;
  document: Document;
  annotation: Annotation | null;
  results: OCRResult[];
  history: Annotation[];
  comparisons: {
    id: string;
    agrees: boolean;
    lighton_result_id: string;
    qwen_result_id: string;
    details: Record<string, unknown>;
  }[];
};
export type Service = {
  name: string;
  state: string;
  reachable: boolean;
  error?: string;
  checked_at: number;
  model_loaded?: boolean;
};
export type Job = {
  id: string;
  kind: string;
  status: string;
  total: number;
  completed: number;
  failed: number;
  error: string | null;
  cancel_requested: boolean;
  created_at: string;
  config: Record<string, unknown>;
  result?: unknown;
  progress?: Record<string, unknown>;
  training_history?: unknown[];
  tasks?: {
    id: string;
    page_id: string;
    status: string;
    error: string | null;
  }[];
};
export type Dataset = {
  id: string;
  name: string;
  counts: Record<string, { pages: number; grounding: number }>;
  created_at: string;
};
export type Checkpoint = {
  id: string;
  step: number;
  path: string;
  job_id: string;
  metadata_json: Record<string, unknown>;
};

export const statusLabels: Record<string, string> = {
  queued: "В очереди",
  running: "Выполняется",
  ready: "Готово",
  completed: "Завершено",
  failed: "Ошибка",
  cancelled: "Отменено",
  loading: "Загрузка модели",
  unknown: "Проверка состояния",
  waiting_service: "Ожидание службы",
  waiting_gpu: "Ожидание GPU",
  busy: "Занята",
  unreachable: "Недоступна",
};
export const stageLabels: Record<string, string> = {
  ingest: "Подготовка страниц",
  lighton: "LightOn 4B",
  qwen: "Qwen",
  compare: "Сравнение",
  train: "Обучение",
  evaluate: "Оценка",
  probe: "Пробное распознавание",
  export: "Экспорт GGUF",
};
export const splitLabels: Record<string, string> = {
  unassigned: "Не распределено",
  train: "Обучение",
  validation: "Валидация",
  test: "Тест",
};
