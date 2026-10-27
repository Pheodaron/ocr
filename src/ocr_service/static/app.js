"use strict";
const $ = (id) => document.getElementById(id);
const labels = {queued:"В очереди",processing:"Распознаётся",cancelling:"Останавливается…",cancelled:"Остановлено",completed:"Готово",partial:"Частичный результат",failed:"Ошибка"};
const terminal = new Set(["cancelled","completed","partial","failed"]);
const pendingActions = new Set();
let selectedId = null;
let pollTimer;
let maxUploadMb = 64;
const backendLabels = new Map();
const backendOnline = new Map();
const backendStatuses = new Map();
let healthState = "checking";
let backendSelectionChosen = false;
let uploading = false;
let reportDownloading = false;
const runViews = new Map();
let runsVisible = 20;
let runsRefreshing = false;
let runsRefreshPending = false;
let uploadingRunId = null;
let openRuns = new Set();
try { openRuns = new Set(JSON.parse(localStorage.getItem("ocr-open-runs") || "[]")); } catch {}

function saveOpenRuns() {
  try { localStorage.setItem("ocr-open-runs", JSON.stringify([...openRuns])); } catch {}
}

function updateReportButtons() {
  for (const [id, view] of runViews) {
    for (const button of view.reportButtons) button.disabled = reportDownloading || id === uploadingRunId || !view.data;
  }
  $("document-report").disabled = reportDownloading;
}

function createRunView(id) {
  const element = node("details", undefined, "run-card"); element.dataset.runId = id;
  const summary = node("summary", undefined, "run-heading");
  const chevron = node("span", "›", "run-chevron"); chevron.setAttribute("aria-hidden", "true");
  const title = node("span", undefined, "run-title"), subtitle = node("span", undefined, "muted");
  const info = node("span", undefined, "run-info"), state = node("span");
  info.append(title, subtitle); summary.append(chevron, info, state);
  const body = node("div", undefined, "run-body");
  const progress = node("p", "Загрузка документов…", "muted");
  const scroll = node("div", undefined, "table-scroll"), table = node("table", undefined, "batch-table");
  const head = node("thead"), headings = node("tr"), rows = node("tbody");
  for (const label of ["Документ", "Провайдер", "Статус", "Страницы", "Обработка, с", "Действия"]) headings.append(node("th", label));
  head.append(headings); table.append(head, rows); scroll.append(table);
  const reports = node("div", undefined, "downloads"), message = node("p", "", "muted");
  message.setAttribute("role", "status");
  const reportButtons = ["html", "json"].map((format) => {
    const button = node("button", `Скачать отчёт ${format.toUpperCase()}`, "secondary");
    button.type = "button"; button.disabled = true; button.dataset.reportFormat = format;
    button.addEventListener("click", () => downloadReport([], format, message, [], id));
    reports.append(button); return button;
  });
  body.append(node("h3", "Пакет документов"), progress, scroll, reports, message);
  element.append(summary, body);
  const view = {element, title, subtitle, state, body, progress, rows, message, reportButtons, rowViews:new Map(), data:null, loading:false};
  runViews.set(id, view);
  element.open = openRuns.has(id);
  element.addEventListener("toggle", () => {
    if (element.open) { openRuns.add(id); loadRunDetails(id); }
    else openRuns.delete(id);
    saveOpenRuns();
  });
  return view;
}

function updateRunDetails(view, data) {
  view.data = data;
  let accepted = 0, finished = 0, errors = 0;
  const present = new Set();
  for (const file of data.files) {
    present.add(file.position);
    const job = file.job;
    if (job) accepted++;
    if (job && terminal.has(job.status)) finished++;
    if (file.error || job?.status === "failed") errors++;
    let row = view.rowViews.get(file.position);
    if (!row) {
      const element = node("tr");
      const cells = Array.from({length:6}, () => node("td")); element.append(...cells);
      row = {element, cells, jobId:null, actionsKey:null}; view.rowViews.set(file.position, row);
      view.rows.append(element);
    }
    if (job && row.jobId !== job.id) {
      const button = node("button", file.filename, "document-link"); button.type = "button";
      button.setAttribute("aria-label", `Открыть документ ${file.filename}`);
      button.addEventListener("click", () => showJob(job.id)); row.cells[0].replaceChildren(button);
    } else if (!job) row.cells[0].textContent = file.filename;
    row.jobId = job?.id || null;
    row.cells[1].textContent = job ? configurationLabel(job) : backendLabels.get(data.backend) || data.backend;
    if (file.error) { row.cells[2].textContent = file.error; row.cells[2].className = "error"; }
    else {
      row.cells[2].className = "";
      const state = node("span");
      if (job) badge(state, job.status);
      else { state.textContent = "Ожидает загрузки"; state.className = "muted"; }
      row.cells[2].replaceChildren(state);
    }
    row.cells[3].textContent = job ? `${job.processed_pages} / ${job.page_count}` : "—";
    row.cells[4].textContent = job && job.processed_pages ? job.processing_seconds.toFixed(2) : "—";
    const actionsKey = job ? `${job.id}:${job.status}:${pendingActions.has(job.id)}` : "missing";
    if (row.actionsKey !== actionsKey) {
      row.cells[5].replaceChildren(...(job ? [jobActions(job)] : [])); row.actionsKey = actionsKey;
    }
  }
  for (const [position, row] of view.rowViews) if (!present.has(position)) {
    row.element.remove(); view.rowViews.delete(position);
  }
  view.progress.textContent = `Файлов: ${data.files.length} · принято: ${accepted} · завершено: ${finished}`
    + (errors ? ` · ошибок: ${errors}` : "") + (uploadingRunId === data.id ? " · идёт загрузка…" : "");
  if (!view.message.textContent || view.detailError) view.message.textContent = "Отчёт содержит время страниц, ресурсы и подробный пайплайн обработки.";
  view.detailError = false; updateReportButtons();
}

async function loadRunDetails(id) {
  const view = runViews.get(id);
  if (!view || view.loading) return;
  view.loading = true;
  try { updateRunDetails(view, await request(`/api/runs/${id}`)); }
  catch (error) { view.message.textContent = `Не удалось обновить запуск: ${error.message}`; view.detailError = true; }
  finally { view.loading = false; }
}

async function refreshRuns() {
  if (runsRefreshing) { runsRefreshPending = true; return; }
  runsRefreshing = true; $("more-runs").disabled = true;
  try {
    const pages = [];
    for (let offset = 0; offset < runsVisible; offset += 100) pages.push(request(`/api/runs?limit=${Math.min(100, runsVisible - offset)}&offset=${offset}`));
    const responses = await Promise.all(pages), runs = responses.flatMap((page) => page.runs);
    const ids = new Set(runs.map((run) => run.id));
    for (const [id, view] of runViews) if (!ids.has(id)) { view.element.remove(); runViews.delete(id); }
    let previous = null;
    for (const run of runs) {
      const view = runViews.get(run.id) || createRunView(run.id);
      const date = new Intl.DateTimeFormat("ru-RU", {dateStyle:"short", timeStyle:"short", timeZone:"Europe/Moscow"}).format(new Date(run.created_at));
      view.title.textContent = `Запуск ${date} МСК`;
      view.subtitle.textContent = `${run.total} файлов · ${backendLabels.get(run.backend) || (run.backend === "comparison" ? "Сравнение моделей" : run.backend)}`;
      badge(view.state, run.status === "pending" ? "queued" : run.status);
      if (run.status === "pending") view.state.textContent = "Ожидает загрузки";
      // Keep existing DOM nodes/focus and expansion while inserting new runs in order.
      const next = previous ? previous.nextElementSibling : $("runs-list").firstElementChild;
      if (next !== view.element) $("runs-list").insertBefore(view.element, next);
      previous = view.element;
    }
    $("runs-message").textContent = runs.length ? "Раскройте запуск, затем нажмите на документ для просмотра результата." : "Запусков пока нет.";
    $("more-runs").hidden = !responses.at(-1).has_more;
    await Promise.all([...runViews].filter(([, view]) => view.element.open).map(([id]) => loadRunDetails(id)));
  } catch (error) { $("runs-message").textContent = `Не удалось загрузить запуски: ${error.message}`; }
  finally {
    runsRefreshing = false; $("more-runs").disabled = false;
    if (runsRefreshPending) { runsRefreshPending = false; refreshRuns(); }
  }
}

async function downloadReport(ids, format, status, uploadErrors = [], runId = null) {
  if ((!runId && !ids.length && !uploadErrors.length) || reportDownloading) return;
  reportDownloading = true; updateReportButtons();
  status.textContent = "Формирование отчёта…";
  try {
    const response = await fetch(runId ? `/api/runs/${runId}/report?format=${format}` : `/api/reports?format=${format}`, runId ? {} : {
      method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({document_ids:ids, upload_errors:uploadErrors}),
    });
    if (!response.ok) {
      const body = await response.json();
      throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`);
    }
    const url = URL.createObjectURL(await response.blob());
    const link = node("a"); link.href = url; link.download = `ocr-processing-report.${format}`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status.textContent = "Отчёт скачан.";
  } catch (error) { status.textContent = `Не удалось получить отчёт: ${error.message}`; }
  finally { reportDownloading = false; updateReportButtons(); }
}

function backendAvailability(id) {
  const status = backendStatuses.get(id);
  if (status?.ready) return "Доступен";
  if (status?.state === "loading") return "Загружается";
  if (status?.state === "error") return "Ошибка запуска";
  if (status) return "Недоступен";
  return healthState === "offline" ? "Нет связи" : healthState === "online" ? "Нет данных" : "Проверка доступности…";
}

function updateBackendOptions() {
  const select = $("backend");
  for (const option of select.options) {
    if (backendLabels.has(option.value)) option.textContent = `${backendLabels.get(option.value)} — ${backendAvailability(option.value)}`;
  }
  // Prefer a ready provider on first load; never replace the user's selection on a health update.
  if (!backendSelectionChosen) {
    const ready = [...select.options].find((option) => backendOnline.get(option.value));
    if (ready) { select.value = ready.value; backendSelectionChosen = true; }
  }
  const status = $("backend-status");
  if (select.disabled) return;
  const available = backendOnline.get(select.value);
  select.title = select.selectedOptions[0]?.textContent || "";
  status.className = available ? "provider-ready" : "muted";
  status.textContent = available ? "Провайдер доступен для распознавания."
    : healthState === "offline" ? "Нет связи с сервисом. Доступность провайдера неизвестна."
    : healthState === "checking" ? "Проверка доступности провайдера…"
    : `${backendAvailability(select.value)}. Задание будет ожидать готовности провайдера.`;
  const selectedStatus = backendStatuses.get(select.value);
  const descriptions = {loading:"Загрузка модели…",idle:"Провайдер готов",processing:"Провайдер работает",offline:"Провайдер недоступен",error:"Ошибка запуска OCR"};
  $("worker-state").textContent = descriptions[selectedStatus?.state] || backendAvailability(select.value);
  $("worker-state").className = "badge " + (available ? "good" : selectedStatus?.state === "error" ? "bad" : "warning");
  $("worker-error").hidden = !selectedStatus?.error;
  $("worker-error").textContent = selectedStatus?.error || "";
}

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
  if (selectedId === id) $("document-dialog").close();
}

async function controlJob(job, action) {
  if (pendingActions.has(job.id)) return;
  pendingActions.add(job.id);
  document.querySelectorAll("button[data-job-id]").forEach((button) => {
    if (button.dataset.jobId === job.id) button.disabled = true;
  });
  try {
    await request(`/api/documents/${job.id}${action === "cancel" ? "/cancel" : ""}`, {method: action === "cancel" ? "POST" : "DELETE"});
    if (action === "delete") forgetJob(job.id);
    $("message").textContent = action === "cancel" ? "Запрошена остановка распознавания. Готовые страницы сохраняются." : "Распознавание удаляется вместе с исходным PDF и результатами.";
  } catch (error) {
    if (error.status === 404) forgetJob(job.id);
    else $("message").textContent = error.message;
  } finally {
    pendingActions.delete(job.id);
  }
  await refreshRuns();
  if (selectedId === job.id) await showJob(job.id);
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
    const source = page.markdown || page.text;
    if (source) {
      const details = node("details"); details.open = result.pages.length <= 3;
      details.append(node("summary", "Распознанный текст"));
      const modes = node("div", undefined, "text-modes");
      modes.setAttribute("role", "group");
      modes.setAttribute("aria-label", "Режим отображения распознанного текста");
      const rawButton = node("button", "Сырой текст", "secondary");
      const previewButton = node("button", "Предпросмотр", "secondary");
      const raw = node("pre", source, "text-source");
      const preview = node("div", undefined, "text-preview");
      preview.setAttribute("role", "region");
      preview.setAttribute("aria-label", `Предпросмотр текста страницы ${page.number}`);
      let loaded = false, loading = false;
      async function setMode(rendered) {
        raw.hidden = rendered; preview.hidden = !rendered;
        rawButton.setAttribute("aria-pressed", String(!rendered));
        previewButton.setAttribute("aria-pressed", String(rendered));
        if (!rendered || loaded || loading) return;
        loading = true;
        preview.setAttribute("aria-busy", "true");
        preview.textContent = "Формирование предпросмотра…";
        try {
          const response = await request(`/api/documents/${result.document.id}/pages/${page.number}/preview`);
          // Only the server's sanitized Markdown/HTML goes into this container.
          preview.innerHTML = response.html;
          loaded = true;
        } catch (error) {
          preview.textContent = `Не удалось загрузить предпросмотр: ${error.message}. Нажмите «Предпросмотр», чтобы повторить.`;
        } finally {
          loading = false;
          preview.setAttribute("aria-busy", "false");
        }
      }
      rawButton.type = previewButton.type = "button";
      rawButton.addEventListener("click", () => setMode(false));
      previewButton.addEventListener("click", () => setMode(true));
      modes.append(rawButton, previewButton);
      setMode(false);
      details.append(modes, raw, preview);
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
  selectedId = id;
  if (!$("document-dialog").open) $("document-dialog").showModal();
  document.body.classList.add("modal-open");
  $("filename").textContent = "Загрузка документа…";
  $("document-message").textContent = "";
  $("job").hidden = false; $("downloads").hidden = true; $("pages").replaceChildren();
  $("job-actions").replaceChildren(); $("engine-details").textContent = "";
  try { await pollJob(id); } catch (error) { $("document-message").textContent = error.message; }
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
      await refreshRuns();
    } else pollTimer = setTimeout(() => pollJob(id), 1500);
  } catch (error) {
    if (selectedId === id) {
      if (error.status === 404) { forgetJob(id); return; }
      $("document-message").textContent = `Не удалось получить статус: ${error.message}`;
      pollTimer = setTimeout(() => pollJob(id), 5000);
    }
  }
}

async function refreshHealth() {
  try {
    const {workers} = await request("/health");
    healthState = "online";
    backendOnline.clear(); backendStatuses.clear();
    for (const [backend, status] of Object.entries(workers || {})) {
      backendOnline.set(backend, status.ready);
      backendStatuses.set(backend, status);
    }
  } catch {
    healthState = "offline"; backendOnline.clear(); backendStatuses.clear();
    $("worker-state").textContent = "Нет соединения с сервисом";
    $("worker-state").className = "badge warning";
    $("worker-error").hidden = true;
  }
  updateBackendOptions();
  refreshRuns().catch(() => {});
  setTimeout(refreshHealth, 5000);
}

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const files = [...$("pdf").files], backend = $("backend").value;
  if (!files.length || !backend || uploading) return;
  if (files.length > 1000) { $("message").textContent = "Выберите не более 1000 файлов за один раз."; return; }
  uploading = true; backendSelectionChosen = true;
  $("upload-button").disabled = true; $("backend").disabled = true; $("pdf").disabled = true;
  const failures = [];
  let accepted = 0, runId, lastDetailsUpdate = 0;
  try {
    const run = await request("/api/runs", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({backend, filenames:files.map((file) => file.name)}),
    });
    runId = run.id; uploadingRunId = runId; openRuns.add(runId); saveOpenRuns();
    await refreshRuns();
    for (const [index, file] of files.entries()) {
      $("message").textContent = `Загрузка ${index + 1} из ${files.length}: ${file.name}`;
      try {
        if (file.size > maxUploadMb * 1024 * 1024) throw new Error(`Лимит файла: ${maxUploadMb} МБ`);
        const form = new FormData(); form.append("file", file); form.append("backend", backend);
        form.append("run_id", runId); form.append("file_index", index);
        await request("/api/documents", {method:"POST", body:form}); accepted++;
      } catch (error) { failures.push({position:index, error:error.message.slice(0, 2000)}); }
      if (performance.now() - lastDetailsUpdate > 1000 || index === files.length - 1) {
        await loadRunDetails(runId); lastDetailsUpdate = performance.now();
      }
    }
    $("message").textContent = `Принято документов: ${accepted} из ${files.length}.`
      + (accepted ? " Обработка продолжится на сервере, даже если закрыть страницу." : " Проверьте ошибки в запуске.");
  } catch (error) { $("message").textContent = error.message; }
  finally {
    if (runId) {
      try {
        await request(`/api/runs/${runId}/finish-upload`, {
          method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({failures}),
        });
      } catch (error) { $("message").textContent += ` Не удалось завершить загрузку запуска: ${error.message}`; }
    }
    uploading = false; uploadingRunId = null;
    $("backend").disabled = false; $("pdf").disabled = false; $("upload-button").disabled = false;
    updateBackendOptions(); await refreshRuns();
  }
});
$("document-report").addEventListener("click", () => { if (selectedId) downloadReport([selectedId], "html", $("document-message")); });
$("close-document").addEventListener("click", () => $("document-dialog").close());
$("document-dialog").addEventListener("close", () => {
  clearTimeout(pollTimer); selectedId = null; document.body.classList.remove("modal-open");
  $("job").hidden = true; $("pages").replaceChildren();
});
$("document-dialog").addEventListener("keydown", (event) => {
  if (event.key !== "Tab") return;
  const controls = [...$("document-dialog").querySelectorAll('a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])')]
    .filter((element) => element.tabIndex >= 0 && element.getClientRects().length);
  if (!controls.length) { event.preventDefault(); return; }
  if (event.shiftKey && document.activeElement === controls[0]) {
    event.preventDefault(); controls.at(-1).focus();
  } else if (!event.shiftKey && document.activeElement === controls.at(-1)) {
    event.preventDefault(); controls[0].focus();
  }
});
$("document-dialog").addEventListener("click", (event) => {
  const rect = $("document-dialog").getBoundingClientRect();
  if (event.target === $("document-dialog") && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) $("document-dialog").close();
});
$("more-runs").addEventListener("click", () => { runsVisible += 20; refreshRuns(); });
$("backend").addEventListener("change", () => {
  backendSelectionChosen = true;
  localStorage.setItem("ocr-backend", $("backend").value);
  updateBackendOptions();
});
$("refresh").addEventListener("click", () => refreshRuns().catch((error) => { $("message").textContent = error.message; }));
request("/api/config").then((config) => {
  maxUploadMb = config.max_upload_mb;
  $("limits").textContent = `Каждый PDF: до ${config.max_upload_mb} МБ и ${config.max_pages} страниц. Обработка на CPU; результаты сохраняются на сервере.`;
  const select = $("backend");
  select.replaceChildren();
  config.backends.forEach((backend) => {
    const option = node("option"); option.value = backend.id;
    const text = backend.label + (backend.quantization ? ` · ${backend.quantization}` : "") + (backend.task ? ` · ${backend.task}` : "");
    backendLabels.set(backend.id, text); select.append(option);
  });
  const saved = localStorage.getItem("ocr-backend");
  if (config.backends.some((backend) => backend.id === saved)) {
    select.value = saved; backendSelectionChosen = true;
  }
  select.disabled = !config.backends.length;
  $("upload-button").disabled = select.disabled;
  if (select.disabled) {
    const option = node("option", "Нет доступных конфигураций"); option.value = "";
    select.append(option);
    $("backend-status").textContent = "Провайдеры OCR не настроены.";
  }
  updateBackendOptions();
}).catch((error) => {
  $("message").textContent = error.message;
  const option = node("option", "Не удалось загрузить провайдеров"); option.value = "";
  $("backend").replaceChildren(option);
  $("backend-status").textContent = "Обновите страницу, чтобы повторить загрузку провайдеров.";
});
refreshHealth(); refreshRuns().catch((error) => { $("message").textContent = error.message; });
