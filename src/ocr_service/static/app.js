"use strict";
const $ = (id) => document.getElementById(id);
const labels = {queued:"В очереди",processing:"Распознаётся",cancelling:"Останавливается…",cancelled:"Остановлено",completed:"Готово",partial:"Частичный результат",failed:"Ошибка"};
const terminal = new Set(["cancelled","completed","partial","failed"]);
const pendingActions = new Set();
let selectedId = localStorage.getItem("ocr-document");
let pollTimer;
let maxUploadMb = 64;
let comparisonId = localStorage.getItem("ocr-comparison");
let comparisonTimer;
const completedResults = new Map();
const backendLabels = new Map();
const backendOnline = new Map();

function configurationLabel(job) {
  const profile = job.configuration;
  if (!profile) return backendLabels.get(job.backend) || job.backend;
  return profile.label + (profile.quantization ? ` · ${profile.quantization}` : "")
    + (profile.task ? ` · ${profile.task}` : "") + (profile.context ? ` · ctx ${profile.context}` : "");
}

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail;
    try { detail = (await response.json()).detail; } catch { detail = `HTTP ${response.status}`; }
    const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function badge(element, status) {
  element.textContent = labels[status] || status;
  element.className = "badge " + (status === "completed" ? "good" : status === "failed" ? "bad" : status === "partial" ? "warning" : "");
}

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function forgetJob(id) {
  completedResults.delete(id);
  if (selectedId === id) {
    clearTimeout(pollTimer); selectedId = null;
    localStorage.removeItem("ocr-document"); $("job").hidden = true;
    $("pages").replaceChildren(); $("job-actions").replaceChildren();
  }
}

function forgetComparison() {
  clearTimeout(comparisonTimer); comparisonId = null;
  localStorage.removeItem("ocr-comparison"); $("comparison").hidden = true;
  $("comparison-rows").replaceChildren();
}

async function controlJob(job, action) {
  if (pendingActions.has(job.id)) return;
  pendingActions.add(job.id);
  document.querySelectorAll("button[data-job-id]").forEach((button) => {
    if (button.dataset.jobId === job.id) button.disabled = true;
  });
  try {
    await request(`/api/documents/${job.id}${action === "cancel" ? "/cancel" : ""}`, {method: action === "cancel" ? "POST" : "DELETE"});
    completedResults.delete(job.id);
    if (action === "delete") forgetJob(job.id);
    $("message").textContent = action === "cancel" ? "Запрошена остановка распознавания. Готовые страницы сохраняются." : "Распознавание удаляется вместе с исходным PDF и результатами.";
  } catch (error) {
    if (error.status === 404) forgetJob(job.id);
    else $("message").textContent = error.message;
  } finally {
    pendingActions.delete(job.id);
  }
  await refreshHistory();
  if (selectedId === job.id) await showJob(job.id);
  if (comparisonId) await showComparison(comparisonId);
}

function jobActions(job) {
  const actions = node("div", undefined, "job-actions");
  function add(text, action, disabled = false) {
    const button = node("button", text, action === "delete" ? "danger" : "secondary");
    button.type = "button"; button.dataset.jobId = job.id;
    button.disabled = disabled || pendingActions.has(job.id);
    button.addEventListener("click", () => controlJob(job, action).catch((error) => { $("message").textContent = error.message; }));
    actions.append(button);
  }
  if (!terminal.has(job.status)) add(job.status === "cancelling" ? "Останавливается…" : "Остановить", "cancel", job.status === "cancelling");
  add("Удалить", "delete");
  return actions;
}

function renderPages(result) {
  $("pages").replaceChildren();
  for (const page of result.pages) {
    const section = node("article", undefined, "page");
    const heading = node("div", undefined, "page-head");
    heading.append(node("h3", `Страница ${page.number}`));
    const metrics = page.status === "cancelled" ? "Остановлена" : page.status === "failed" ? "Не обработана" : `${page.seconds.toFixed(1)} с · ${page.dpi} dpi` + (page.confidence === null ? "" : ` · OCR score ${page.confidence.toFixed(3)}`);
    heading.append(node("span", metrics, "muted"));
    section.append(heading);
    if (page.error) section.append(node("p", page.error, page.status === "cancelled" ? "muted" : "error"));
    if (page.warnings.length) section.append(node("p", page.warnings.join("\n"), "warning-note"));
    if (page.text) {
      const details = node("details"); details.open = result.pages.length <= 3;
      details.append(node("summary", "Распознанный текст"), node("pre", page.text));
      section.append(details);
    }
    for (const [index, table] of page.tables.entries()) {
      section.append(node("h3", `Таблица ${index + 1}`));
      const scroll = node("div", undefined, "table-scroll");
      const element = node("table");
      const rows = new Map();
      for (const cell of table.cells) {
        if (!rows.has(cell.row)) { const row = node("tr"); rows.set(cell.row, row); element.append(row); }
        const td = node(cell.header ? "th" : "td", cell.text);
        td.rowSpan = cell.rowspan; td.colSpan = cell.colspan; rows.get(cell.row).append(td);
      }
      scroll.append(element); section.append(scroll);
    }
    if (page.image_url) {
      const details = node("details"); details.append(node("summary", "Изображение страницы"));
      details.addEventListener("toggle", () => {
        if (details.open && !details.querySelector("img")) {
          const image = node("img"); image.src = page.image_url; image.alt = `Страница ${page.number}`; image.loading = "lazy"; details.append(image);
        }
      });
      section.append(details);
    }
    $("pages").append(section);
  }
}

async function showJob(id) {
  clearTimeout(pollTimer);
  selectedId = id; localStorage.setItem("ocr-document", id);
  $("job").hidden = false; $("downloads").hidden = true; $("pages").replaceChildren();
  $("job-actions").replaceChildren(); $("engine-details").textContent = "";
  try { await pollJob(id); } catch (error) { $("message").textContent = error.message; }
}

async function pollJob(id) {
  try {
    const job = await request(`/api/documents/${id}`);
    if (selectedId !== id) return;
    $("filename").textContent = `${job.filename} · ${configurationLabel(job)}`; badge($("job-state"), job.status);
    $("job-actions").replaceChildren(jobActions(job));
    $("progress").max = job.page_count; $("progress").value = job.processed_pages;
    $("progress-text").textContent = `Обработано ${job.processed_pages} из ${job.page_count} · ошибок: ${job.failed_pages} · на проверку: ${job.review_pages}` + (job.cancelled_pages ? ` · остановлено страниц: ${job.cancelled_pages}` : "");
    if (terminal.has(job.status)) {
      const result = await request(`/api/documents/${id}/result`);
      if (selectedId !== id) return;
      renderPages(result); $("downloads").hidden = false;
      $("engine-details").textContent = `Загрузка модели: ${job.model_load_seconds.toFixed(1)} с · обработка: ${job.processing_seconds.toFixed(1)} с`;
      if (job.error) $("engine-details").textContent += ` · ${job.error}`;
      document.querySelectorAll("[data-format]").forEach((link) => { link.href = `/api/documents/${id}/download?format=${link.dataset.format}`; });
      await refreshHistory();
    } else pollTimer = setTimeout(() => pollJob(id), 1500);
  } catch (error) {
    if (selectedId === id) {
      if (error.status === 404) { forgetJob(id); return; }
      $("message").textContent = `Не удалось получить статус: ${error.message}`;
      pollTimer = setTimeout(() => pollJob(id), 5000);
    }
  }
}

async function refreshHistory() {
  const jobs = await request("/api/documents"); $("history").replaceChildren();
  if (!jobs.length) $("history").append(node("li", "Документов пока нет", "muted"));
  for (const job of jobs) {
    const item = node("li"); const button = node("button", `${job.filename} · ${configurationLabel(job)}`);
    button.addEventListener("click", () => { showJob(job.id); if (job.comparison_id) showComparison(job.comparison_id); });
    const state = node("span"); badge(state, job.status); item.append(button, state, jobActions(job)); $("history").append(item);
  }
}

async function showComparison(id) {
  clearTimeout(comparisonTimer);
  comparisonId = id; localStorage.setItem("ocr-comparison", id);
  $("comparison").hidden = false;
  try {
    const comparison = await request(`/api/comparisons/${id}`);
    if (comparisonId !== id) return;
    $("comparison-json").href = `/api/comparisons/${id}`;
    $("comparison-json").download = `comparison-${id}.json`;
    const rows = [];
    for (const job of comparison.jobs) {
      let result = completedResults.get(job.id);
      if (!result && terminal.has(job.status)) {
        try { result = await request(`/api/documents/${job.id}/result`); }
        catch (error) { if (error.status === 404) continue; throw error; }
        // Final timing is written just after the document status. Cache only the result.
        completedResults.set(job.id, result);
      }
      const row = node("tr"); row.append(node("td", configurationLabel(job)));
      const state = node("td"), stateBadge = node("span"); badge(stateBadge, job.status);
      if (job.status === "queued" && backendOnline.get(job.backend) === false) stateBadge.textContent = "Ожидает запуска сервиса";
      state.append(stateBadge);
      row.append(state);
      row.append(node("td", job.model_load_seconds.toFixed(1)), node("td", job.processing_seconds.toFixed(1)));
      let tables = 0, cells = 0, tokens = 0, milliseconds = 0;
      for (const page of result?.pages || []) {
        tables += page.tables.length;
        cells += page.tables.reduce((total, table) => total + table.cells.length, 0);
        const timing = page.metrics?.timings;
        if (timing?.predicted_ms > 0 && timing.predicted_n > 0) {
          tokens += timing.predicted_n; milliseconds += timing.predicted_ms;
        }
      }
      row.append(node("td", result ? `${tables} / ${cells}` : "—"), node("td", milliseconds ? (1000 * tokens / milliseconds).toFixed(2) : "—"));
      const action = node("td"), button = node("button", "Открыть");
      button.addEventListener("click", () => showJob(job.id)); action.append(button, jobActions(job)); row.append(action); rows.push(row);
    }
    if (comparisonId !== id) return;
    $("comparison-rows").replaceChildren(...rows);
    if (comparison.jobs.some((job) => !terminal.has(job.status))) comparisonTimer = setTimeout(() => showComparison(id), 2000);
  } catch (error) {
    if (error.status === 404) { if (comparisonId === id) forgetComparison(); return; }
    $("message").textContent = error.message;
    if (comparisonId === id) comparisonTimer = setTimeout(() => showComparison(id), 5000);
  }
}

async function refreshHealth() {
  try {
    const {worker, workers} = await request("/health");
    for (const [backend, status] of Object.entries(workers || {})) {
      backendOnline.set(backend, status.ready);
      const indicator = document.getElementById(`backend-${backend}`);
      if (indicator) indicator.textContent = status.ready ? " · запущен" : " · сервис выключен / не готов";
    }
    const descriptions = {loading:"Загрузка моделей…",idle:"Сервис готов",processing:"Сервис работает",offline:"Worker не запущен",error:"Ошибка запуска OCR"};
    $("worker-state").textContent = descriptions[worker.state] || worker.state;
    $("worker-state").className = "badge " + (worker.ready ? "good" : "warning");
    $("worker-error").hidden = !worker.error; $("worker-error").textContent = worker.error || "";
  } catch { $("worker-state").textContent = "Нет соединения с сервисом"; }
  refreshHistory().catch(() => {});
  setTimeout(refreshHealth, 5000);
}

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const file = $("pdf").files[0];
  if (!file) return;
  if (file.size > maxUploadMb * 1024 * 1024) { $("message").textContent = `Лимит файла: ${maxUploadMb} МБ`; return; }
  $("upload-button").disabled = true; $("message").textContent = "Загрузка PDF…";
  const form = new FormData(); form.append("file", file);
  try {
    const backends = [...document.querySelectorAll("input[name=backend]:checked")].map((input) => input.value);
    if (!backends.length) throw new Error("Выберите хотя бы одну конфигурацию");
    let job;
    if (backends.length > 1) {
      backends.forEach((backend) => form.append("backends", backend));
      const comparison = await request("/api/comparisons", {method:"POST",body:form});
      job = comparison.jobs[0]; await showComparison(comparison.id);
    } else {
      form.append("backend", backends[0]);
      job = await request("/api/documents", {method:"POST",body:form});
      clearTimeout(comparisonTimer); comparisonId = null;
      localStorage.removeItem("ocr-comparison"); $("comparison").hidden = true;
    }
    $("message").textContent = "Документ принят. Обработка продолжится, даже если закрыть страницу.";
    await showJob(job.id); await refreshHistory();
  } catch (error) { $("message").textContent = error.message; }
  finally { $("upload-button").disabled = false; }
});
$("refresh").addEventListener("click", () => refreshHistory().catch((error) => { $("message").textContent = error.message; }));
request("/api/config").then((config) => {
  maxUploadMb = config.max_upload_mb;
  $("limits").textContent = `До ${config.max_upload_mb} МБ и ${config.max_pages} страниц. Обработка на CPU; результаты сохраняются на сервере.`;
  $("backends").replaceChildren();
  config.backends.forEach((backend, index) => {
    const label = node("label"), input = node("input"), indicator = node("span", "", "muted");
    indicator.id = `backend-${backend.id}`;
    input.type = "checkbox"; input.name = "backend"; input.value = backend.id; input.checked = index === 0;
    const text = backend.label + (backend.quantization ? ` · ${backend.quantization}` : "") + (backend.task ? ` · ${backend.task}` : "");
    backendLabels.set(backend.id, text); label.append(input, document.createTextNode(text), indicator); $("backends").append(label);
  });
}).catch((error) => { $("message").textContent = error.message; });
refreshHealth(); refreshHistory().catch((error) => { $("message").textContent = error.message; });
if (selectedId) showJob(selectedId);
if (comparisonId) showComparison(comparisonId);
