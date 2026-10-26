import { post, api } from "../api";
import { useJobs } from "../hooks";
import type { Action } from "../hooks";
import { stageLabels, statusLabels } from "../types";

export function ActivityPanel({
  projectId,
  onSelect,
  onAction,
}: {
  projectId: string;
  onSelect: (id: string) => void;
  onAction: Action;
}) {
  const jobs = useJobs(projectId);
  return (
    <section className="activity">
      <div className="section-heading">
        <h2>Задания</h2>
        <span className="muted">Следующий этап запускается вручную</span>
      </div>
      {jobs.data?.map((job) => (
        <div className="job-row" key={job.id}>
          <button className="text-button" onClick={() => onSelect(job.id)}>
            {stageLabels[job.kind] || job.kind}
            <small>{new Date(job.created_at).toLocaleString("ru-RU")}</small>
          </button>
          <div className="job-progress">
            <progress max={job.total || 1} value={job.completed + job.failed} />
            <span>
              {job.completed} / {job.total} · ошибок {job.failed}
            </span>
          </div>
          <span className={`badge ${job.status === "failed" ? "error" : ""}`}>
            {statusLabels[job.status] || job.status}
          </span>
          {!["completed", "failed", "cancelled"].includes(job.status) ? (
            <button
              className="danger"
              onClick={() =>
                void onAction(
                  () => post(`/projects/${projectId}/jobs/${job.id}/cancel`),
                  "Остановка запрошена; ожидается завершение текущей операции",
                )
              }
            >
              Остановить
            </button>
          ) : (
            <button
              onClick={() =>
                void onAction(
                  () =>
                    api(`/projects/${projectId}/jobs/${job.id}`, {
                      method: "DELETE",
                    }),
                  "Задание убрано из списка",
                )
              }
            >
              {job.kind === "train" ? "Убрать из списка" : "Удалить"}
            </button>
          )}
        </div>
      ))}
      {!jobs.data?.length && <p className="hint">Заданий пока нет.</p>}
    </section>
  );
}
