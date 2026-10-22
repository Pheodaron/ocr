"use strict";
const $ = (id) => document.getElementById(id);
const labels = {queued:"В очереди",processing:"Распознаётся",completed:"Готово",partial:"Частичный результат",failed:"Ошибка"};
const terminal = new Set(["completed","partial","failed"]);
let selectedId = localStorage.getItem("ocr-document");
let pollTimer;
let maxUploadMb = 64;

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail;
    try { detail = (await response.json()).detail; } catch { detail = `HTTP ${response.status}`; }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
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

function renderPages(result) {
  $("pages").replaceChildren();
  for (const page of result.pages) {
    const section = node("article", undefined, "page");
    const heading = node("div", undefined, "page-head");
    heading.append(node("h3", `Страница ${page.number}`));
    const metrics = page.status === "failed" ? "Не обработана" : `${page.seconds.toFixed(1)} с · ${page.dpi} dpi` + (page.confidence === null ? "" : ` · OCR score ${page.confidence.toFixed(3)}`);
    heading.append(node("span", metrics, "muted"));
    section.append(heading);
    if (page.error) section.append(node("p", page.error, "error"));
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
  try { await pollJob(id); } catch (error) { $("message").textContent = error.message; }
}

async function pollJob(id) {
  try {
    const job = await request(`/api/documents/${id}`);
    if (selectedId !== id) return;
    $("filename").textContent = job.filename; badge($("job-state"), job.status);
    $("progress").max = job.page_count; $("progress").value = job.processed_pages;
    $("progress-text").textContent = `Обработано ${job.processed_pages} из ${job.page_count} · ошибок: ${job.failed_pages} · на проверку: ${job.review_pages}`;
    if (terminal.has(job.status)) {
      const result = await request(`/api/documents/${id}/result`);
      if (selectedId !== id) return;
      renderPages(result); $("downloads").hidden = false;
      document.querySelectorAll("[data-format]").forEach((link) => { link.href = `/api/documents/${id}/download?format=${link.dataset.format}`; });
      await refreshHistory();
    } else pollTimer = setTimeout(() => pollJob(id), 1500);
  } catch (error) {
    if (selectedId === id) {
      $("message").textContent = `Не удалось получить статус: ${error.message}`;
      pollTimer = setTimeout(() => pollJob(id), 5000);
    }
  }
}

async function refreshHistory() {
  const jobs = await request("/api/documents"); $("history").replaceChildren();
  if (!jobs.length) $("history").append(node("li", "Документов пока нет", "muted"));
  for (const job of jobs) {
    const item = node("li"); const button = node("button", job.filename);
    button.addEventListener("click", () => showJob(job.id));
    const state = node("span"); badge(state, job.status); item.append(button, state); $("history").append(item);
  }
}

async function refreshHealth() {
  try {
    const {worker} = await request("/health");
    const descriptions = {loading:"Загрузка моделей…",idle:"Сервис готов",processing:"Сервис работает",offline:"Worker не запущен",error:"Ошибка запуска OCR"};
    $("worker-state").textContent = descriptions[worker.state] || worker.state;
    $("worker-state").className = "badge " + (worker.ready ? "good" : "warning");
    $("worker-error").hidden = !worker.error; $("worker-error").textContent = worker.error || "";
  } catch { $("worker-state").textContent = "Нет соединения с сервисом"; }
  setTimeout(refreshHealth, 5000);
}

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const file = $("pdf").files[0];
  if (!file) return;
  if (file.size > maxUploadMb * 1024 * 1024) { $("message").textContent = `Лимит файла: ${maxUploadMb} МБ`; return; }
  $("upload-button").disabled = true; $("message").textContent = "Загрузка PDF…";
  const form = new FormData(); form.append("file", file);
  try {
    const job = await request("/api/documents", {method:"POST",body:form});
    $("message").textContent = "Документ принят. Обработка продолжится, даже если закрыть страницу.";
    await showJob(job.id); await refreshHistory();
  } catch (error) { $("message").textContent = error.message; }
  finally { $("upload-button").disabled = false; }
});
$("refresh").addEventListener("click", () => refreshHistory().catch((error) => { $("message").textContent = error.message; }));
request("/api/config").then((config) => { maxUploadMb = config.max_upload_mb; $("limits").textContent = `До ${config.max_upload_mb} МБ и ${config.max_pages} страниц. Обработка на CPU; результаты сохраняются на сервере.`; }).catch(() => {});
refreshHealth(); refreshHistory().catch((error) => { $("message").textContent = error.message; });
if (selectedId) showJob(selectedId);
