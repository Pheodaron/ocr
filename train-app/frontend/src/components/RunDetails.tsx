import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import type { Job } from "../types";
import { Preview } from "./Preview";
import { useState } from "react";
import { LossChart } from "./LossChart";
import { ImageEditor } from "./ImageEditor";
import type { Block } from "../types";

type EvaluationPage = {
  page_id: string;
  markdown: string;
  expected: string;
  error: string | null;
  metrics: { cer: number | null; tables: { teds: number }[] };
};
type Report = {
  complete: boolean;
  summary: Record<string, unknown>;
  pages: EvaluationPage[];
};

export function RunDetails({
  projectId,
  jobId,
}: {
  projectId: string;
  jobId: string;
}) {
  const [showLog, setShowLog] = useState(false);
  const log = useQuery({
    queryKey: ["log", jobId],
    queryFn: () =>
      api<{ text: string }>(`/projects/${projectId}/jobs/${jobId}/log`),
    enabled: showLog,
    refetchInterval: showLog ? 3000 : false,
  });
  const { data, error } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api<Job>(`/projects/${projectId}/jobs/${jobId}`),
    refetchInterval: 2000,
  });
  if (!data) return <p>{error ? String(error) : "Загрузка…"}</p>;
  const output = data.result as {
    models?: Record<string, Report>;
    text?: string;
    markdown?: string;
    blocks?: Block[];
    page?: { id: string; width: number; height: number };
    error?: string;
    validation?: unknown;
    history?: unknown;
    checkpoints?: unknown;
    files?: unknown;
    excluded?: unknown;
  } | null;
  return (
    <section className="run-details">
      <h3>Подробности задания {jobId.slice(0, 8)}</h3>
      {data.error && <p className="error">{data.error}</p>}
      {!!data.training_history?.length && (
        <LossChart samples={data.training_history} />
      )}
      <details onToggle={(event) => setShowLog(event.currentTarget.open)}>
        <summary>Журнал вычислительной службы</summary>
        <pre>{log.data?.text || "Загрузка журнала…"}</pre>
      </details>
      {data.progress && (
        <pre className="progress-details">
          {JSON.stringify(data.progress, null, 2)}
        </pre>
      )}
      {output?.text !== undefined && (
        <>
          <p className="error">{output.error}</p>
          <div className={output.page ? "editor-columns" : ""}>
            {output.page && (
              <ImageEditor
                url={`/api/projects/${projectId}/pages/${output.page.id}/image`}
                width={output.page.width}
                height={output.page.height}
                blocks={output.blocks || []}
                selected=""
                onSelect={() => {}}
                onChange={() => {}}
                readOnly
              />
            )}
            <Preview content={output.markdown || output.text} />
          </div>
        </>
      )}
      {output?.models && (
        <>
          <div className="metrics-grid">
            {Object.entries(output.models).map(([name, report]) => (
              <section className="card" key={name}>
                <h3>
                  {name === "baseline" ? "Исходная модель" : "После обучения"}
                </h3>
                <p>
                  {report.complete ? "Полная оценка" : "Оценка не завершена"}
                </p>
                <dl>
                  {Object.entries(report.summary)
                    .filter(
                      ([, value]) =>
                        typeof value === "number" || value === null,
                    )
                    .map(([key, value]) => (
                      <div key={key}>
                        <dt>{key}</dt>
                        <dd>
                          {typeof value === "number"
                            ? Number(value.toFixed(5))
                            : "не применимо"}
                        </dd>
                      </div>
                    ))}
                </dl>
              </section>
            ))}
          </div>
          <p className="hint">
            Меньше CER — лучше. Больше TEDS и точных числовых ячеек — лучше.
            Пропуски и ошибки входят в расчёт.
          </p>
          {output.excluded && (
            <details>
              <summary>Исключения из независимого теста</summary>
              <pre>{JSON.stringify(output.excluded, null, 2)}</pre>
            </details>
          )}
          {(output.models.trained || output.models.baseline).pages.map(
            (page, index) => (
              <details key={page.page_id} className="evaluation-page">
                <summary>
                  Страница {index + 1} · CER{" "}
                  {page.metrics.cer?.toFixed(4) ?? "—"}{" "}
                  {page.error && `· ${page.error}`}
                </summary>
                <img
                  className="evaluation-image"
                  src={`/api/projects/${projectId}/pages/${page.page_id}/image`}
                  alt="Исходная страница"
                  loading="lazy"
                />
                <div className="comparison-columns">
                  <section>
                    <h4>Эталон</h4>
                    <Preview content={page.expected} />
                  </section>
                  <section>
                    <h4>Результат</h4>
                    <Preview content={page.markdown} />
                  </section>
                </div>
              </details>
            ),
          )}
        </>
      )}
      {Boolean(output?.history) && (
        <details open>
          <summary>Ход обучения</summary>
          <pre>{JSON.stringify(output!.history, null, 2)}</pre>
        </details>
      )}
      {Boolean(output?.checkpoints) && (
        <details>
          <summary>Контрольные точки</summary>
          <pre>{JSON.stringify(output!.checkpoints, null, 2)}</pre>
        </details>
      )}
      {Boolean(output?.validation) && (
        <details>
          <summary>Валидация</summary>
          <pre>{JSON.stringify(output!.validation, null, 2)}</pre>
        </details>
      )}
      {Boolean(output?.files) && (
        <pre>{JSON.stringify(output!.files, null, 2)}</pre>
      )}
      <details>
        <summary>Ошибки отдельных страниц и параметры</summary>
        <pre>
          {JSON.stringify({ config: data.config, tasks: data.tasks }, null, 2)}
        </pre>
      </details>
    </section>
  );
}
