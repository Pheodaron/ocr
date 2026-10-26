import { lazy, Suspense, useState } from "react";
import { useDocuments, usePages } from "../hooks";

const PageEditor = lazy(() =>
  import("./PageEditor").then((module) => ({ default: module.PageEditor })),
);

export function ReviewPanel({
  projectId,
  initialDocument,
  notify,
}: {
  projectId: string;
  initialDocument: string;
  notify: (text: string) => void;
}) {
  const key = `ocr-position:${projectId}`;
  const [documentId, setDocumentId] = useState(
    initialDocument || localStorage.getItem(key + ":document") || "",
  );
  const [pageId, setPageId] = useState(
    localStorage.getItem(key + ":page") || "",
  );
  const documents = useDocuments(projectId),
    pages = usePages(projectId, documentId);
  const activePage = pages.data?.some((page) => page.id === pageId)
    ? pageId
    : pages.data?.[0]?.id || "";
  const index = pages.data?.findIndex((page) => page.id === activePage) ?? -1;
  function chooseDocument(id: string) {
    setDocumentId(id);
    setPageId("");
    localStorage.setItem(key + ":document", id);
    localStorage.removeItem(key + ":page");
  }
  function choosePage(id: string) {
    setPageId(id);
    localStorage.setItem(key + ":page", id);
  }
  return (
    <>
      <div className="toolbar wrap">
        <label>
          Документ{" "}
          <select
            value={documentId}
            onChange={(event) => chooseDocument(event.target.value)}
          >
            <option value="">Выберите документ</option>
            {documents.data?.map((document) => (
              <option value={document.id} key={document.id}>
                {document.filename}
              </option>
            ))}
          </select>
        </label>
        <button
          disabled={index <= 0}
          onClick={() => choosePage(pages.data![index - 1].id)}
        >
          ←
        </button>
        <label>
          Страница{" "}
          <select
            value={activePage}
            onChange={(event) => choosePage(event.target.value)}
          >
            {pages.data?.map((page) => (
              <option key={page.id} value={page.id}>
                {page.number}
                {page.current_annotation_id ? " · есть редакция" : ""}
              </option>
            ))}
          </select>
        </label>
        <button
          disabled={!pages.data || index >= pages.data.length - 1}
          onClick={() => choosePage(pages.data![index + 1].id)}
        >
          →
        </button>
      </div>
      {activePage ? (
        <Suspense fallback={<p>Загрузка редактора…</p>}>
          <PageEditor
            key={activePage}
            projectId={projectId}
            pageId={activePage}
            notify={notify}
          />
        </Suspense>
      ) : (
        <p className="empty">
          Выберите документ. Изображения доступны после подготовки, результаты —
          после распознавания.
        </p>
      )}
    </>
  );
}
