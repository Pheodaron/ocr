export type Cell = {
  row: number;
  column: number;
  rowspan: number;
  colspan: number;
  html: string;
  header: boolean;
};
export type TableModel = { rows: number; columns: number; cells: Cell[] };

export function parseTable(source: string): TableModel {
  const root = new DOMParser().parseFromString(source, "text/html");
  const table = root.querySelector("table");
  if (!table || root.querySelectorAll("table").length !== 1)
    throw new Error("Нужна одна таблица без вложенных таблиц");
  const cells: Cell[] = [];
  const occupied = new Set<string>();
  let columns = 0;
  const rows = Array.from(table.rows);
  rows.forEach((row, r) => {
    let c = 0;
    Array.from(row.cells).forEach((cell) => {
      while (occupied.has(`${r}:${c}`)) c++;
      const rowspan = Number(cell.getAttribute("rowspan") || "1");
      const colspan = Number(cell.getAttribute("colspan") || "1");
      if (
        !Number.isInteger(rowspan) ||
        !Number.isInteger(colspan) ||
        rowspan < 1 ||
        colspan < 1 ||
        r + rowspan > rows.length ||
        colspan > 200
      )
        throw new Error("Некорректное объединение");
      for (let y = r; y < r + rowspan; y++)
        for (let x = c; x < c + colspan; x++) {
          if (occupied.has(`${y}:${x}`))
            throw new Error("Пересечение объединений");
          occupied.add(`${y}:${x}`);
        }
      cells.push({
        row: r,
        column: c,
        rowspan,
        colspan,
        html: cell.innerHTML,
        header: cell.tagName === "TH",
      });
      c += colspan;
      columns = Math.max(columns, c);
    });
  });
  if (!cells.length) throw new Error("В таблице нет ячеек");
  for (let r = 0; r < rows.length; r++)
    for (let c = 0; c < columns; c++)
      if (!occupied.has(`${r}:${c}`))
        throw new Error("В таблице есть незаполненные места");
  return { rows: rows.length, columns, cells };
}

export function serializeTable(table: TableModel): string {
  const rows = Array.from({ length: table.rows }, (_, row) => {
    const cells = table.cells
      .filter((c) => c.row === row)
      .sort((a, b) => a.column - b.column)
      .map((cell) => {
        const tag = cell.header ? "th" : "td";
        return `<${tag}${cell.rowspan > 1 ? ` rowspan="${cell.rowspan}"` : ""}${cell.colspan > 1 ? ` colspan="${cell.colspan}"` : ""}>${cell.html}</${tag}>`;
      })
      .join("");
    return `<tr>${cells}</tr>`;
  }).join("\n");
  return `<table>\n<tbody>\n${rows}\n</tbody>\n</table>`;
}

function fill(table: TableModel): TableModel {
  const occupied = new Set<string>();
  for (const cell of table.cells)
    for (let r = cell.row; r < cell.row + cell.rowspan; r++)
      for (let c = cell.column; c < cell.column + cell.colspan; c++)
        occupied.add(`${r}:${c}`);
  for (let r = 0; r < table.rows; r++)
    for (let c = 0; c < table.columns; c++)
      if (!occupied.has(`${r}:${c}`))
        table.cells.push({
          row: r,
          column: c,
          rowspan: 1,
          colspan: 1,
          html: "",
          header: false,
        });
  return table;
}

export function insertAxis(
  original: TableModel,
  axis: "row" | "column",
  at: number,
): TableModel {
  const table = structuredClone(original);
  const span = axis === "row" ? "rowspan" : "colspan";
  for (const cell of table.cells) {
    if (cell[axis] < at && cell[axis] + cell[span] > at) cell[span]++;
    else if (cell[axis] >= at) cell[axis]++;
  }
  if (axis === "row") table.rows++;
  else table.columns++;
  return fill(table);
}

export function removeAxis(
  original: TableModel,
  axis: "row" | "column",
  at: number,
): TableModel {
  const table = structuredClone(original);
  if ((axis === "row" ? table.rows : table.columns) <= 1)
    throw new Error("Нельзя удалить последнюю строку или столбец");
  const span = axis === "row" ? "rowspan" : "colspan";
  table.cells = table.cells.filter((cell) => {
    if (cell[axis] <= at && cell[axis] + cell[span] > at) {
      if (cell[span] === 1) return false;
      cell[span]--;
    } else if (cell[axis] > at) cell[axis]--;
    return true;
  });
  if (axis === "row") table.rows--;
  else table.columns--;
  return fill(table);
}

export function splitCell(original: TableModel, selected: Cell): TableModel {
  const table = structuredClone(original);
  table.cells = table.cells.filter(
    (cell) => cell.row !== selected.row || cell.column !== selected.column,
  );
  table.cells.push({ ...selected, rowspan: 1, colspan: 1 });
  return fill(table);
}

export function mergeCells(
  original: TableModel,
  first: Cell,
  last: Cell,
): TableModel {
  const table = structuredClone(original);
  const top = Math.min(first.row, last.row),
    left = Math.min(first.column, last.column);
  const bottom = Math.max(first.row + first.rowspan, last.row + last.rowspan);
  const right = Math.max(
    first.column + first.colspan,
    last.column + last.colspan,
  );
  const inside = table.cells.filter(
    (cell) =>
      cell.row < bottom &&
      cell.row + cell.rowspan > top &&
      cell.column < right &&
      cell.column + cell.colspan > left,
  );
  if (
    inside.some(
      (cell) =>
        cell.row < top ||
        cell.column < left ||
        cell.row + cell.rowspan > bottom ||
        cell.column + cell.colspan > right,
    )
  )
    throw new Error("Выделение пересекает объединённую ячейку");
  table.cells = table.cells.filter((cell) => !inside.includes(cell));
  table.cells.push({
    row: top,
    column: left,
    rowspan: bottom - top,
    colspan: right - left,
    html: inside
      .sort((a, b) => a.row - b.row || a.column - b.column)
      .map((c) => c.html)
      .filter(Boolean)
      .join("<br>"),
    header: first.header,
  });
  return table;
}

export function cellText(source: string): string {
  const root = new DOMParser().parseFromString(
    source.replace(/<br\s*\/?>/gi, "\n"),
    "text/html",
  );
  return root.body.textContent || "";
}

export function escapeCell(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\n/g, "<br>");
}
