import { useState } from "react";
import { api, post } from "../api";
import { useDatasets } from "../hooks";
import type { Action } from "../hooks";
import { splitLabels } from "../types";
import { Preview } from "./Preview";

export function DatasetsPanel({
  projectId,
  onAction,
}: {
  projectId: string;
  onAction: Action;
}) {
  const action = onAction;
  const datasets = useDatasets(projectId);
  const [manifest, setManifest] = useState<{
    items: {
      page_id: string;
      filename: string;
      markdown: string;
      split: string;
    }[];
  } | null>(null);
  const [manifestPage, setManifestPage] = useState(0);
  return (
    <>
      <section className="section-heading">
        <div>
          <h2>Зафиксированные наборы данных</h2>
          <p>
            Включаются только утверждённые страницы. Для grounding дополнительно
            нужны проверенные рамки.
          </p>
        </div>
        <button
          className="primary"
          onClick={() => {
            const name = prompt("Название версии данных");
            if (name)
              void action(() =>
                post(`/projects/${projectId}/datasets`, { name }),
              );
          }}
        >
          Создать версию
        </button>
      </section>
      <div className="cards">
        {datasets.data?.map((dataset) => (
          <section className="card" key={dataset.id}>
            <h3>{dataset.name}</h3>
            <dl>
              {Object.entries(dataset.counts).map(([id, count]) => (
                <div key={id}>
                  <dt>{splitLabels[id]}</dt>
                  <dd>
                    {count.pages} стр. · grounding {count.grounding}
                  </dd>
                </div>
              ))}
            </dl>
            <div className="toolbar">
              <button
                onClick={() =>
                  void action(async () => {
                    setManifest(
                      await api(
                        `/projects/${projectId}/datasets/${dataset.id}`,
                      ),
                    );
                    setManifestPage(0);
                  }, "")
                }
              >
                Посмотреть эталоны
              </button>
              <a
                className="button"
                href={`/api/projects/${projectId}/datasets/${dataset.id}/download`}
              >
                Скачать
              </a>
            </div>
          </section>
        ))}
      </div>
      {manifest && (
        <section className="card">
          <div className="toolbar">
            <label>
              Страница набора{" "}
              <select
                value={manifestPage}
                onChange={(e) => setManifestPage(Number(e.target.value))}
              >
                {manifest.items.map((item, i) => (
                  <option key={item.page_id} value={i}>
                    {item.filename} · {splitLabels[item.split]} · {i + 1}
                  </option>
                ))}
              </select>
            </label>
            <button onClick={() => setManifest(null)}>Закрыть</button>
          </div>
          <div className="comparison-columns">
            <img
              className="snapshot-image"
              src={`/api/projects/${projectId}/pages/${manifest.items[manifestPage].page_id}/image`}
              alt="Изображение эталона"
            />
            <Preview content={manifest.items[manifestPage].markdown} />
          </div>
        </section>
      )}
    </>
  );
}
