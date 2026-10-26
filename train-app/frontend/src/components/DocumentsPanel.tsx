import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, post } from "../api";
import { useDocuments } from "../hooks";
import { splitLabels, statusLabels } from "../types";
import type { Project } from "../types";
import type { Action, StartStage } from "../hooks";

export function DocumentsPanel({
  project,
  onOpenDocument,
  onReview,
  onProjectDeleted,
  onStart,
  onAction,
  notify,
}: {
  project: Project;
  onOpenDocument: (id: string) => void;
  onReview: () => void;
  onProjectDeleted: () => void;
  onStart: StartStage;
  onAction: Action;
  notify: (message: string) => void;
}) {
  const projectId = project.id;
  const client = useQueryClient();
  const documents = useDocuments(projectId);
  const [selected, setSelected] = useState<string[]>([]);
  const [mode, setMode] = useState("missing");
  const [split, setSplit] = useState("all");
  const [uploadProgress, setUploadProgress] = useState("");
  const visible = (documents.data || []).filter(
    (item) => split === "all" || item.split === split,
  );
  const action = onAction;
  const chooseDocument = onOpenDocument;
  const chooseProject = (_id: string) => onProjectDeleted();
  const setTab = (_tab: string) => onReview();
  const start = (kind: string) =>
    onStart(kind, {}, { document_ids: selected, mode });
  async function upload(files: FileList | File[]) {
    if (!projectId) return;
    const queue = Array.from(files);
    let completed = 0;
    const errors: string[] = [];
    setUploadProgress(`Загрузка: 0 / ${queue.length}`);
    async function worker() {
      while (queue.length) {
        const file = queue.shift()!;
        const form = new FormData();
        form.append("file", file);
        try {
          await api(`/projects/${projectId}/documents`, {
            method: "POST",
            body: form,
          });
        } catch (error) {
          errors.push(`${file.name}: ${error}`);
        }
        completed++;
        setUploadProgress(`Загрузка: ${completed} / ${files.length}`);
        await client.invalidateQueries({ queryKey: ["documents", projectId] });
      }
    }
    await Promise.all(
      Array.from({ length: Math.min(3, queue.length) }, worker),
    );
    setUploadProgress("");
    notify(
      errors.length
        ? errors.join("\n")
        : "Файлы загружены. Страницы подготавливаются.",
    );
  }

  return (
    <>
      <section
        className="upload-zone"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          void upload(e.dataTransfer.files);
        }}
      >
        <div>
          <h2>Документы проекта</h2>
          <p>
            PDF, PNG или JPEG. Выберите сразу несколько файлов или перетащите их
            сюда.
          </p>
        </div>
        <label className="button primary">
          Загрузить документы
          <input
            hidden
            type="file"
            multiple
            accept=".pdf,.png,.jpg,.jpeg"
            disabled={project.archived}
            onChange={(e) => {
              if (e.target.files) void upload(e.target.files);
              e.target.value = "";
            }}
          />
        </label>
        {uploadProgress && <p>{uploadProgress}</p>}
      </section>
      <div className="toolbar wrap">
        <span>Выбрано: {selected.length || "все документы"}</span>
        <label>
          Режим{" "}
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="missing">Не обработанные</option>
            <option value="failed">Ошибочные</option>
            <option value="all">Повторить все</option>
          </select>
        </label>
        <button onClick={() => void start("lighton")}>
          1. Распознать LightOn
        </button>
        <button onClick={() => void start("qwen")}>2. Распознать Qwen</button>
        <button onClick={() => void start("compare")}>Сравнить ответы</button>
        <button onClick={() => setTab("review")}>3. Проверить вручную</button>
      </div>
      <div className="toolbar wrap">
        <label>
          Выборка{" "}
          <select value={split} onChange={(e) => setSplit(e.target.value)}>
            <option value="all">Все</option>
            {Object.entries(splitLabels).map(([id, title]) => (
              <option value={id} key={id}>
                {title}
              </option>
            ))}
          </select>
        </label>
        {Object.entries(splitLabels).map(([id, title]) => (
          <button
            key={id}
            disabled={!selected.length}
            onClick={() =>
              void action(() =>
                api(`/projects/${projectId}/splits`, {
                  method: "PATCH",
                  body: JSON.stringify({
                    document_ids: selected,
                    split: id,
                  }),
                }),
              )
            }
          >
            {title}
          </button>
        ))}
        <button
          disabled={!selected.length}
          onClick={() => {
            const group = prompt(
              "Общее семейство документов (для защиты от пересечения выборок)",
            );
            if (group !== null)
              void action(() =>
                api(`/projects/${projectId}/splits`, {
                  method: "PATCH",
                  body: JSON.stringify({
                    document_ids: selected,
                    group,
                  }),
                }),
              );
          }}
        >
          Задать семейство
        </button>
        <button
          onClick={() => {
            if (
              confirm(
                "Распределить все документы по семействам: 80% обучение, 10% валидация, 10% тест?",
              )
            )
              void action(() =>
                post(`/projects/${projectId}/splits/automatic`, {
                  fractions: [0.8, 0.1, 0.1],
                  seed: 42,
                }),
              );
          }}
        >
          Распределить 80 / 10 / 10
        </button>
        <button
          onClick={() =>
            void action(async () => {
              const found = await api<unknown[]>(
                `/projects/${projectId}/duplicates`,
              );
              notify(
                `Кандидаты похожих страниц: ${found.length}.\n${JSON.stringify(found, null, 2)}`,
              );
            }, "")
          }
        >
          Проверить похожие страницы
        </button>
      </div>
      <div className="table-scroll">
        <table className="documents-table">
          <thead>
            <tr>
              <th>
                <input
                  type="checkbox"
                  aria-label="Выбрать все документы"
                  checked={
                    !!visible.length &&
                    visible.every((d) => selected.includes(d.id))
                  }
                  onChange={(e) =>
                    setSelected(
                      e.target.checked ? visible.map((d) => d.id) : [],
                    )
                  }
                />
              </th>
              <th>Документ</th>
              <th>Страницы</th>
              <th>Утверждено</th>
              <th>Выборка</th>
              <th>Состояние</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((document) => (
              <tr key={document.id}>
                <td>
                  <input
                    type="checkbox"
                    aria-label={`Выбрать ${document.filename}`}
                    checked={selected.includes(document.id)}
                    onChange={(e) =>
                      setSelected(
                        e.target.checked
                          ? [...selected, document.id]
                          : selected.filter((id) => id !== document.id),
                      )
                    }
                  />
                </td>
                <td>
                  <button
                    className="text-button"
                    onClick={() => chooseDocument(document.id)}
                  >
                    {document.filename}
                  </button>
                  <small>{document.group}</small>
                </td>
                <td>
                  {document.rendered_pages} / {document.page_count}
                </td>
                <td>
                  {document.approved_pages} · рамки {document.geometry_pages}
                </td>
                <td>{splitLabels[document.split]}</td>
                <td>
                  <span className="badge">
                    {statusLabels[document.status] || document.status}
                  </span>
                  {document.error && (
                    <small className="error">{document.error}</small>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!visible.length && (
        <p className="empty">В этой выборке пока нет документов.</p>
      )}
      <details className="project-settings">
        <summary>Управление проектом</summary>
        <div className="toolbar wrap">
          <button
            onClick={() => {
              const title = prompt("Название проекта", project.title);
              if (title)
                void action(() =>
                  api(`/projects/${projectId}`, {
                    method: "PATCH",
                    body: JSON.stringify({ ...project, title }),
                  }),
                );
            }}
          >
            Переименовать
          </button>
          <button
            onClick={() =>
              void action(() =>
                api(`/projects/${projectId}`, {
                  method: "PATCH",
                  body: JSON.stringify({
                    ...project,
                    archived: !project.archived,
                  }),
                }),
              )
            }
          >
            {project.archived ? "Вернуть в работу" : "Архивировать"}
          </button>
          <a className="button" href={`/api/projects/${projectId}/backup`}>
            Резервная копия
          </a>
          <button
            className="danger"
            onClick={() => {
              if (
                confirm(
                  "Удалить проект вместе с документами, редакциями и обученными моделями?",
                )
              )
                void action(async () => {
                  await api(`/projects/${projectId}`, {
                    method: "DELETE",
                  });
                  chooseProject("");
                });
            }}
          >
            Удалить проект
          </button>
        </div>
      </details>
    </>
  );
}
