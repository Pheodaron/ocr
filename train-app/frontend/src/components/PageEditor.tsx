import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import CodeMirror from "@uiw/react-codemirror";
import { markdown } from "@codemirror/lang-markdown";
import { api, post } from "../api";
import type { Block, PageDetails } from "../types";
import { Preview } from "./Preview";
import { ImageEditor } from "./ImageEditor";
import { TableEditor } from "./TableEditor";
import { joinBlocks } from "../annotationModel";

type Draft = {
  blocks: Block[];
  source: string | null;
  expected: string | null;
  textApproved: boolean;
  geometryApproved: boolean;
  note: string;
  rejected: boolean;
};

export function PageEditor({
  projectId,
  pageId,
  notify,
}: {
  projectId: string;
  pageId: string;
  notify: (text: string) => void;
}) {
  const query = useQuery({
    queryKey: ["page", pageId],
    queryFn: () => api<PageDetails>(`/projects/${projectId}/pages/${pageId}`),
  });
  if (!query.data)
    return <p>{query.error ? String(query.error) : "Загрузка страницы…"}</p>;
  return (
    <Editor
      key={`${pageId}:${query.data.annotation?.id || "new"}`}
      projectId={projectId}
      details={query.data}
      notify={notify}
    />
  );
}

function Editor({
  projectId,
  details,
  notify,
}: {
  projectId: string;
  details: PageDetails;
  notify: (text: string) => void;
}) {
  const client = useQueryClient();
  const { page, annotation, results } = details;
  const initial =
    annotation || results.find((item) => item.status === "completed");
  const initialDraft: Draft = {
    blocks: initial?.blocks.length
      ? structuredClone(initial.blocks)
      : [{ id: crypto.randomUUID(), type: "text", content: "", bbox: null }],
    source:
      annotation?.source_result_id ||
      results.find((item) => item.status === "completed")?.id ||
      null,
    expected: annotation?.id || null,
    textApproved: annotation?.text_approved || false,
    geometryApproved: annotation?.geometry_approved || false,
    note: annotation?.note || "",
    rejected: annotation?.rejected || false,
  };
  const key = `ocr-draft:${projectId}:${page.id}`;
  const [draft, setDraft] = useState<Draft>(() => {
    try {
      const saved = JSON.parse(
        sessionStorage.getItem(key) || "null",
      ) as Draft | null;
      if (saved?.expected === initialDraft.expected) return saved;
    } catch {
      /* Повреждённый черновик не заменяет серверный ответ. */
    }
    return initialDraft;
  });
  const [selected, setSelected] = useState(draft.blocks[0]?.id || ""),
    [mode, setMode] = useState("preview");
  const [version, setVersion] = useState("current"),
    [saving, setSaving] = useState(false);
  const [undo, setUndo] = useState<Block[][]>([]),
    [redo, setRedo] = useState<Block[][]>([]);
  const [savedFingerprint, setSavedFingerprint] = useState(
    JSON.stringify(initialDraft),
  );
  const dirty = JSON.stringify(draft) !== savedFingerprint;
  const currentBlock =
    draft.blocks.find((b) => b.id === selected) || draft.blocks[0];
  const content = joinBlocks(draft.blocks);
  const tableMatches = useMemo(
    () => [
      ...(currentBlock?.content || "").matchAll(
        /<table\b[^>]*>[\s\S]*?<\/table\s*>/gi,
      ),
    ],
    [currentBlock?.content],
  );
  useEffect(() => {
    sessionStorage.setItem(key, JSON.stringify(draft));
  }, [draft, key]);
  useEffect(() => {
    const handler = (event: BeforeUnloadEvent) => {
      if (dirty) event.preventDefault();
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [dirty]);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "s") {
        event.preventDefault();
        void save(false, false);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });
  const [fullPage, setFullPage] = useState(false);
  function changeBlocks(blocks: Block[], geometryOnly = false) {
    setUndo((previous) => [
      ...previous.slice(-99),
      structuredClone(draft.blocks),
    ]);
    setRedo([]);
    setDraft((previous) => ({
      ...previous,
      blocks,
      textApproved:
        geometryOnly && joinBlocks(blocks) === joinBlocks(previous.blocks)
          ? previous.textApproved
          : false,
      geometryApproved: false,
      rejected: false,
    }));
  }
  function setBlock(text: string) {
    changeBlocks(
      draft.blocks.map((b) =>
        b.id === currentBlock.id ? { ...b, content: text } : b,
      ),
    );
  }
  function reorder(direction: number) {
    const index = draft.blocks.findIndex((b) => b.id === currentBlock.id),
      next = index + direction;
    if (next < 0 || next >= draft.blocks.length) return;
    const blocks = [...draft.blocks];
    [blocks[index], blocks[next]] = [blocks[next], blocks[index]];
    changeBlocks(blocks);
  }
  function chooseVersion(id: string) {
    if (id === "current") return;
    const result = results.find((r) => r.id === id);
    const historical = details.history.find((r) => r.id === id);
    const chosen = result || historical;
    if (!chosen) return;
    // Текущий черновик хранится независимо от неизменяемых ответов моделей.
    sessionStorage.setItem(`${key}:previous`, JSON.stringify(draft));
    changeBlocks(structuredClone(chosen.blocks));
    setDraft((previous) => ({
      ...previous,
      source: result?.id || historical?.source_result_id || null,
    }));
    setSelected(chosen.blocks[0]?.id || "");
    setVersion(id);
    notify(
      "Вариант открыт для правки. Предыдущий черновик можно восстановить.",
    );
  }
  function restoreDraft() {
    const previous = sessionStorage.getItem(`${key}:previous`);
    if (previous) {
      setDraft(JSON.parse(previous));
      setVersion("current");
    }
  }
  async function save(
    textApproved: boolean,
    geometryApproved: boolean,
    rejected = false,
  ) {
    setSaving(true);
    try {
      const next = { ...draft, textApproved, geometryApproved, rejected };
      await post(`/projects/${projectId}/pages/${page.id}/annotations`, {
        expected_revision: draft.expected,
        source_result_id: draft.source,
        markdown: content,
        blocks: draft.blocks,
        text_approved: textApproved,
        geometry_approved: geometryApproved,
        rejected,
        note: draft.note,
      });
      sessionStorage.removeItem(key);
      setSavedFingerprint(JSON.stringify(next));
      await client.invalidateQueries({ queryKey: ["page", page.id] });
      await client.invalidateQueries({ queryKey: ["documents", projectId] });
      notify(
        rejected
          ? "Страница отмечена непригодной"
          : geometryApproved
            ? "Текст и геометрия утверждены"
            : textApproved
              ? "Текст и таблицы утверждены"
              : "Черновик сохранён",
      );
    } catch (error) {
      notify(String(error));
    } finally {
      setSaving(false);
    }
  }
  return (
    <div className="page-editor">
      <div className="editor-heading">
        <div>
          <strong>{details.document.filename}</strong>
          <span className="muted">
            {" "}
            · Страница {page.number} · {page.width} × {page.height}
          </span>
        </div>
        <span
          className={`badge ${draft.geometryApproved ? "success" : draft.textApproved ? "success" : ""}`}
        >
          {draft.geometryApproved
            ? "Текст и рамки утверждены"
            : draft.textApproved
              ? "Текст утверждён"
              : dirty
                ? "Несохранённый черновик"
                : "Ожидает проверки"}
        </span>
      </div>
      <div className="toolbar wrap">
        <label>
          Вариант{" "}
          <select
            value={version}
            onChange={(e) => chooseVersion(e.target.value)}
          >
            <option value="current">Текущая редакция</option>
            {results.map((result) => (
              <option value={result.id} key={result.id}>
                {result.engine === "lighton" ? "LightOn 4B" : "Qwen"} ·{" "}
                {new Date(result.created_at).toLocaleTimeString("ru-RU")} ·{" "}
                {result.status}
              </option>
            ))}
            {details.history.map((item) => (
              <option value={item.id} key={item.id}>
                Редакция · {new Date(item.created_at).toLocaleString("ru-RU")}
              </option>
            ))}
          </select>
        </label>
        <button onClick={restoreDraft}>Предыдущий черновик</button>
        <button
          disabled={!undo.length}
          onClick={() => {
            setRedo([...redo, draft.blocks]);
            const previous = undo.at(-1)!;
            setUndo(undo.slice(0, -1));
            setDraft({
              ...draft,
              blocks: previous,
              textApproved: false,
              geometryApproved: false,
            });
          }}
        >
          ↶
        </button>
        <button
          disabled={!redo.length}
          onClick={() => {
            setUndo([...undo, draft.blocks]);
            const next = redo.at(-1)!;
            setRedo(redo.slice(0, -1));
            setDraft({
              ...draft,
              blocks: next,
              textApproved: false,
              geometryApproved: false,
            });
          }}
        >
          ↷
        </button>
        <button disabled={saving} onClick={() => void save(false, false)}>
          Сохранить черновик
        </button>
        <button
          className="primary"
          disabled={saving}
          onClick={() => void save(true, false)}
        >
          Утвердить текст и таблицы
        </button>
        <button disabled={saving} onClick={() => void save(true, true)}>
          Утвердить рамки
        </button>
        <button
          className="danger"
          disabled={saving}
          onClick={() => void save(false, false, true)}
        >
          Непригодна
        </button>
      </div>
      {details.comparisons[0] && (
        <p className="hint">
          Сравнение сохранённых версий:{" "}
          {details.comparisons[0].agrees
            ? "модели согласны — требуется проверка человеком"
            : "обнаружены расхождения"}
          . Пара: {details.comparisons[0].lighton_result_id.slice(0, 6)} /{" "}
          {details.comparisons[0].qwen_result_id.slice(0, 6)}.
        </p>
      )}
      {initial && "error" in initial && initial.error && (
        <p className="error">{initial.error}</p>
      )}
      <div className="editor-columns">
        <ImageEditor
          url={`/api/projects/${projectId}/pages/${page.id}/image`}
          width={page.width}
          height={page.height}
          blocks={draft.blocks}
          selected={selected}
          onSelect={(id) => {
            setSelected(id);
            setFullPage(false);
          }}
          onChange={(blocks) => changeBlocks(blocks, true)}
        />
        <div className="result-pane">
          <div className="tabs compact">
            {[
              ["preview", "Просмотр"],
              ["edit", "Текст"],
              ["tables", "Таблицы"],
              ["blocks", "Области"],
              ["diff", "Сравнение"],
            ].map(([id, title]) => (
              <button
                key={id}
                className={mode === id ? "active" : ""}
                onClick={() => {
                  setMode(id);
                  setFullPage(false);
                }}
              >
                {title}
              </button>
            ))}
          </div>
          {mode === "preview" && <Preview content={content} />}
          {["edit", "tables", "blocks"].includes(mode) && (
            <div className="block-picker">
              <label>
                Фрагмент{" "}
                <select
                  value={fullPage ? "__page__" : currentBlock.id}
                  onChange={(e) => {
                    setFullPage(e.target.value === "__page__");
                    if (e.target.value !== "__page__")
                      setSelected(e.target.value);
                  }}
                >
                  {mode === "edit" && (
                    <option value="__page__">Вся страница</option>
                  )}
                  {draft.blocks.map((b, i) => (
                    <option key={b.id} value={b.id}>
                      {i + 1}. {b.type} · {b.content.slice(0, 45)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          )}
          {mode === "edit" && (
            <>
              {fullPage && (
                <p className="hint">
                  Правка всей страницы сохраняет текст, но требует повторной
                  привязки пространственных областей.
                </p>
              )}
              <CodeMirror
                value={fullPage ? content : currentBlock.content}
                height="360px"
                extensions={[markdown()]}
                onChange={(text) =>
                  fullPage
                    ? changeBlocks([
                        {
                          id:
                            draft.blocks.length === 1
                              ? draft.blocks[0].id
                              : crypto.randomUUID(),
                          type: "text",
                          bbox: null,
                          content: text,
                        },
                      ])
                    : setBlock(text)
                }
              />
              <Preview content={currentBlock.content} />
            </>
          )}
          {mode === "tables" && (
            <>
              {!tableMatches.length && (
                <p className="hint">
                  В выбранном фрагменте нет HTML-таблицы. Выберите область
                  таблицы или добавьте её в текстовом режиме.
                </p>
              )}
              {tableMatches.map((match, index) => (
                <TableEditor
                  key={`${currentBlock.id}-${index}`}
                  source={match[0]}
                  onChange={(source) =>
                    setBlock(
                      currentBlock.content.slice(0, match.index!) +
                        source +
                        currentBlock.content.slice(
                          match.index! + match[0].length,
                        ),
                    )
                  }
                />
              ))}
            </>
          )}
          {mode === "blocks" && (
            <>
              <div className="toolbar compact">
                <label>
                  Тип{" "}
                  <input
                    value={currentBlock.type}
                    onChange={(e) =>
                      changeBlocks(
                        draft.blocks.map((b) =>
                          b.id === currentBlock.id
                            ? { ...b, type: e.target.value }
                            : b,
                        ),
                        true,
                      )
                    }
                  />
                </label>
                <button onClick={() => reorder(-1)}>↑</button>
                <button onClick={() => reorder(1)}>↓</button>
                <button
                  disabled={draft.blocks.length === 1}
                  onClick={() => {
                    changeBlocks(
                      draft.blocks.filter((b) => b.id !== currentBlock.id),
                    );
                    setSelected(
                      draft.blocks.find((b) => b.id !== currentBlock.id)!.id,
                    );
                  }}
                >
                  Удалить область
                </button>
                <button
                  onClick={() =>
                    changeBlocks(
                      draft.blocks.map((b) =>
                        b.id === currentBlock.id ? { ...b, bbox: null } : b,
                      ),
                      true,
                    )
                  }
                >
                  Убрать рамку
                </button>
              </div>
              <p className="hint">
                Порядок в списке — порядок чтения. Суффикс «+» обозначает
                продолжение. Рамка:{" "}
                {currentBlock.bbox?.join(", ") || "не задана"}. Новую область
                нарисуйте слева.
              </p>
              <CodeMirror
                value={currentBlock.content}
                height="300px"
                extensions={[markdown()]}
                onChange={setBlock}
              />
            </>
          )}
          {mode === "diff" && (
            <div className="comparison-columns">
              {["lighton", "qwen"].map((engine) => {
                const result = results.find(
                  (r) => r.engine === engine && r.status === "completed",
                );
                return (
                  <section key={engine}>
                    <h3>{engine === "lighton" ? "LightOn 4B" : "Qwen"}</h3>
                    {result ? (
                      <Preview content={result.markdown} />
                    ) : (
                      <p>Нет успешного результата</p>
                    )}
                  </section>
                );
              })}
            </div>
          )}
        </div>
      </div>
      <label className="note">
        Примечание{" "}
        <textarea
          value={draft.note}
          onChange={(e) => setDraft({ ...draft, note: e.target.value })}
          rows={2}
        />
      </label>
      <details>
        <summary>Исходные ответы моделей</summary>
        {results.map((result) => (
          <p key={result.id}>
            {result.engine} · {result.seconds.toFixed(1)} с ·{" "}
            {result.error || result.status} ·{" "}
            <a href={`/api/projects/${projectId}/results/${result.id}/raw`}>
              Скачать исходный JSON
            </a>
          </p>
        ))}
      </details>
    </div>
  );
}
