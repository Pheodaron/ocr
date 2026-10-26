import { useState } from "react";
import { useCheckpoints, useDatasets, useDocuments, usePages } from "../hooks";
import type { StartStage } from "../hooks";

export function ModelPanel({
  projectId,
  tab,
  onStart,
}: {
  projectId: string;
  tab: string;
  onStart: StartStage;
}) {
  const datasets = useDatasets(projectId);
  const checkpoints = useCheckpoints(projectId);
  const documents = useDocuments(projectId);
  const [datasetId, setDatasetId] = useState("");
  const [checkpointId, setCheckpointId] = useState("");
  const [documentId, setDocumentId] = useState("");
  const [pageId, setPageId] = useState("");
  const [prompt, setPrompt] = useState("");
  const pages = usePages(projectId, documentId);
  const activePage = pageId || pages.data?.[0]?.id || "";
  const start = (
    kind: string,
    config: Record<string, unknown> = {},
    onlyPage = false,
  ) =>
    onStart(kind, config, {
      page_ids: onlyPage && activePage ? [activePage] : [],
      mode: "all",
    });
  const [training, setTraining] = useState({
    epochs: 3,
    learning_rate: 0.0001,
    rank: 16,
    gradient_accumulation: 8,
    save_steps: 50,
    validation_steps: 50,
    seed: 42,
    longest_edge: 2048,
    max_tokens: 16384,
    max_output_tokens: 8192,
    mode: "ocr",
    grounding_fraction: 0.2,
    train_vision: false,
  });

  return (
    <>
      <section className="section-heading">
        <div>
          <h2>
            {tab === "training"
              ? "Дообучение LightOnOCR-3-0.8B"
              : "Оценка и пробное распознавание"}
          </h2>
          <p>
            {tab === "training"
              ? "Обучаются исходные веса Hugging Face через LoRA. Набор данных неизменяем."
              : "Исходная и обученная модели проверяются на одинаковых независимых тестовых страницах."}
          </p>
        </div>
      </section>
      <div className="form-grid">
        <label>
          Версия данных
          <select
            value={datasetId}
            onChange={(e) => setDatasetId(e.target.value)}
          >
            <option value="">Выберите набор</option>
            {datasets.data?.map((item) => (
              <option value={item.id} key={item.id}>
                {item.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Контрольная точка
          <select
            value={checkpointId}
            onChange={(e) => setCheckpointId(e.target.value)}
          >
            <option value="">Исходная модель / новый запуск</option>
            {checkpoints.data?.map((item) => (
              <option value={item.id} key={item.id}>
                Шаг {item.step} · {item.job_id.slice(0, 8)}
              </option>
            ))}
          </select>
        </label>
      </div>
      {tab === "training" && (
        <>
          <div className="form-grid">
            <label>
              Режим
              <select
                value={training.mode}
                onChange={(e) =>
                  setTraining({ ...training, mode: e.target.value })
                }
              >
                <option value="ocr">Обычный OCR</option>
                <option value="mixed">OCR + grounding</option>
              </select>
            </label>
            {(
              [
                "epochs",
                "learning_rate",
                "rank",
                "gradient_accumulation",
                "save_steps",
                "validation_steps",
                "longest_edge",
                "max_tokens",
                "max_output_tokens",
                "seed",
              ] as const
            ).map((key) => (
              <label key={key}>
                {
                  {
                    epochs: "Эпохи",
                    learning_rate: "Скорость обучения",
                    rank: "Ранг LoRA",
                    gradient_accumulation: "Накопление градиента",
                    save_steps: "Сохранение каждые N шагов",
                    validation_steps: "Валидация каждые N шагов",
                    longest_edge: "Длинная сторона, пикс.",
                    max_tokens: "Предельная длина примера",
                    max_output_tokens: "Лимит вывода при проверке",
                    seed: "Случайное зерно",
                  }[key]
                }
                <input
                  type="number"
                  step={key === "learning_rate" ? 0.00001 : 1}
                  value={training[key]}
                  onChange={(e) =>
                    setTraining({
                      ...training,
                      [key]: Number(e.target.value),
                    })
                  }
                />
              </label>
            ))}
            <label>
              <input
                type="checkbox"
                checked={training.train_vision}
                onChange={(e) =>
                  setTraining({
                    ...training,
                    train_vision: e.target.checked,
                  })
                }
              />{" "}
              Обучать визуальные слои через LoRA
            </label>
          </div>
          <button
            className="primary"
            disabled={!datasetId}
            onClick={() =>
              void start("train", {
                ...training,
                dataset_id: datasetId,
                resume_checkpoint_id: checkpointId || null,
              })
            }
          >
            4. {checkpointId ? "Продолжить обучение" : "Начать обучение"}
          </button>
        </>
      )}
      {tab === "evaluation" && (
        <>
          <label>
            Режим распознавания{" "}
            <select
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
            >
              <option value="">Текст и таблицы</option>
              <option value="grounding">Текст, таблицы и рамки</option>
            </select>
          </label>
          <div className="toolbar wrap">
            <button
              className="primary"
              disabled={!datasetId}
              onClick={() =>
                void start("evaluate", {
                  prompt,
                  dataset_id: datasetId,
                  checkpoint_id: checkpointId || null,
                })
              }
            >
              Оценить на тестовом наборе
            </button>
            <button
              disabled={!checkpointId}
              onClick={() =>
                void start("export", {
                  checkpoint_id: checkpointId,
                  dataset_id: datasetId || undefined,
                })
              }
            >
              Объединить и экспортировать GGUF Q4
            </button>
          </div>
          <h3>Личная проверка на документе или изображении</h3>
          <p className="hint">
            Загрузите файл в проект, выберите документ и страницу. Без эталона
            показатели качества не вычисляются.
          </p>
          <div className="toolbar wrap">
            <select
              value={documentId}
              onChange={(e) => {
                setDocumentId(e.target.value);
                setPageId("");
              }}
            >
              <option value="">Выберите документ</option>
              {documents.data?.map((item) => (
                <option value={item.id} key={item.id}>
                  {item.filename}
                </option>
              ))}
            </select>
            <select
              value={activePage}
              onChange={(e) => setPageId(e.target.value)}
            >
              {pages.data?.map((item) => (
                <option key={item.id} value={item.id}>
                  Страница {item.number}
                </option>
              ))}
            </select>
            <button
              disabled={!activePage}
              onClick={() => {
                setPageId(activePage);
                void start(
                  "probe",
                  { checkpoint_id: checkpointId || null, prompt },
                  true,
                );
              }}
            >
              Распознать выбранной моделью
            </button>
          </div>
        </>
      )}
    </>
  );
}
