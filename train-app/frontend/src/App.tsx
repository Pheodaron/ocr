import { lazy, Suspense, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, post } from "./api";
import type { Job, Project, Service } from "./types";
import { statusLabels } from "./types";
import type { StartStage } from "./hooks";
import { DocumentsPanel } from "./components/DocumentsPanel";
import { DatasetsPanel } from "./components/DatasetsPanel";
import { ModelPanel } from "./components/ModelPanel";
import { ReviewPanel } from "./components/ReviewPanel";
import { ActivityPanel } from "./components/ActivityPanel";

const RunDetails = lazy(() =>
  import("./components/RunDetails").then((module) => ({
    default: module.RunDetails,
  })),
);
const tabs = [
  ["documents", "Документы"],
  ["review", "Проверка и правка"],
  ["datasets", "Наборы данных"],
  ["training", "Обучение"],
  ["evaluation", "Оценка и проверка"],
];

export default function App() {
  const client = useQueryClient();
  const [projectId, setProjectId] = useState(
    localStorage.getItem("ocr-project") || "",
  );
  const [tab, setTab] = useState("documents");
  const [message, setMessage] = useState("");
  const [reviewDocument, setReviewDocument] = useState("");
  const [jobId, setJobId] = useState("");
  const projects = useQuery({
    queryKey: ["projects"],
    queryFn: () => api<Project[]>("/projects"),
  });
  const services = useQuery({
    queryKey: ["services"],
    queryFn: () => api<Service[]>("/services"),
    staleTime: 10000,
  });
  const project = projects.data?.find((item) => item.id === projectId);

  function chooseProject(id: string) {
    setProjectId(id);
    localStorage.setItem("ocr-project", id);
    setJobId("");
    setReviewDocument("");
    setTab("documents");
  }
  async function action(work: () => Promise<unknown>, success = "Сохранено") {
    try {
      await work();
      await client.invalidateQueries();
      if (success) setMessage(success);
    } catch (error) {
      setMessage(String(error));
    }
  }
  async function createProject() {
    const title = prompt("Название проекта");
    if (!title) return;
    await action(async () => {
      const result = await post<Project>("/projects", { title });
      chooseProject(result.id);
    }, "Проект создан");
  }
  const start: StartStage = async (kind, config = {}, scope = {}) => {
    await action(async () => {
      const job = await post<Job>(`/projects/${projectId}/jobs`, {
        kind,
        config,
        ...scope,
      });
      setJobId(job.id);
      void services.refetch();
    }, "Этап поставлен в очередь. Следующий этап запускается отдельно.");
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="/">
          ◩{" "}
          <span>
            Мастерская OCR<small>Разметка · обучение · оценка</small>
          </span>
        </a>
        <div className="sidebar-heading">
          ПРОЕКТЫ{" "}
          <button title="Создать проект" onClick={() => void createProject()}>
            +
          </button>
        </div>
        <nav>
          {projects.data?.map((item) => (
            <button
              key={item.id}
              className={`project-link ${projectId === item.id ? "active" : ""}`}
              onClick={() => chooseProject(item.id)}
            >
              {item.title}
              {item.archived && <small>Архив</small>}
            </button>
          ))}
        </nav>
        {!projects.data?.length && (
          <p className="hint">Создайте проект и загрузите документы.</p>
        )}
        <div className="sidebar-footer">
          Данные сохраняются локально.
          <br />
          Модели запускаются отдельными службами.
        </div>
      </aside>
      <main>
        <header>
          <div>
            <p className="eyebrow">LIGHTONOCR · ПОДГОТОВКА ДАННЫХ</p>
            <h1>{project?.title || "Мастерская обучения"}</h1>
          </div>
          <button onClick={() => void services.refetch()}>
            Проверить службы
          </button>
        </header>
        <div className="service-row">
          {services.data?.map((service) => (
            <div
              key={service.name}
              className={`service-chip ${service.reachable ? "online" : ""}`}
              title={`Проверено: ${new Date(service.checked_at * 1000).toLocaleString("ru-RU")}`}
            >
              <span className="dot" />
              <strong>{service.name}</strong>
              <span>
                {service.name === "lighton4b" &&
                service.state === "ready" &&
                !service.model_loaded
                  ? "Служба готова · модель загрузится при запуске этапа"
                  : statusLabels[service.state] || service.state}
              </span>
              <code>
                podman compose --profile {service.name} up -d {service.name}
              </code>
            </div>
          ))}
        </div>
        {message && (
          <div
            role="status"
            className={`notice ${message.startsWith("Error") ? "error" : ""}`}
          >
            <span>{message}</span>
            <button
              onClick={() => setMessage("")}
              aria-label="Закрыть сообщение"
            >
              ×
            </button>
          </div>
        )}
        {project ? (
          <>
            <div className="tabs">
              {tabs.map(([id, title]) => (
                <button
                  key={id}
                  className={tab === id ? "active" : ""}
                  onClick={() => setTab(id)}
                >
                  {title}
                </button>
              ))}
            </div>
            {project.archived && (
              <p className="notice">
                Проект архивирован. Для изменения верните его в работу.
              </p>
            )}
            {tab === "documents" && (
              <DocumentsPanel
                key={projectId}
                project={project}
                onOpenDocument={(id) => {
                  setReviewDocument(id);
                  setTab("review");
                }}
                onReview={() => setTab("review")}
                onProjectDeleted={() => chooseProject("")}
                onStart={start}
                onAction={action}
                notify={setMessage}
              />
            )}
            {tab === "review" && (
              <ReviewPanel
                key={`${projectId}:${reviewDocument}`}
                projectId={projectId}
                initialDocument={reviewDocument}
                notify={setMessage}
              />
            )}
            {tab === "datasets" && (
              <DatasetsPanel
                key={projectId}
                projectId={projectId}
                onAction={action}
              />
            )}
            {["training", "evaluation"].includes(tab) && (
              <ModelPanel
                key={projectId}
                projectId={projectId}
                tab={tab}
                onStart={start}
              />
            )}
            <ActivityPanel
              projectId={projectId}
              onSelect={setJobId}
              onAction={action}
            />
            {jobId && (
              <Suspense fallback={<p>Загрузка результатов…</p>}>
                <RunDetails projectId={projectId} jobId={jobId} />
              </Suspense>
            )}
          </>
        ) : (
          <section className="welcome card">
            <h2>От документа до проверенной модели</h2>
            <p>
              Создайте проект. Загрузите документы, получите предложения двух
              моделей, проверьте таблицы и сформируйте набор для дообучения.
            </p>
            <button className="primary" onClick={() => void createProject()}>
              Создать проект
            </button>
            <label className="button">
              Восстановить из резервной копии
              <input
                type="file"
                accept=".zip"
                hidden
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file)
                    void action(async () => {
                      const form = new FormData();
                      form.append("file", file);
                      const result = await api<{ project_id: string }>(
                        "/restore",
                        { method: "POST", body: form },
                      );
                      chooseProject(result.project_id);
                    });
                }}
              />
            </label>
          </section>
        )}
      </main>
    </div>
  );
}
