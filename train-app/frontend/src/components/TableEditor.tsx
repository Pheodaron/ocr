import { useMemo, useState } from "react";
import {
  cellText,
  escapeCell,
  insertAxis,
  mergeCells,
  parseTable,
  removeAxis,
  serializeTable,
  splitCell,
} from "../tableModel";
import type { Cell, TableModel } from "../tableModel";

export function TableEditor({
  source,
  onChange,
}: {
  source: string;
  onChange: (source: string) => void;
}) {
  const [selection, setSelection] = useState<[number, number][]>([[0, 0]]);
  const [error, setError] = useState("");
  const parsed = useMemo(() => {
    try {
      return { table: parseTable(source), error: "" };
    } catch (e) {
      return { table: null, error: String(e) };
    }
  }, [source]);
  if (!parsed.table)
    return <p className="error">{parsed.error}. Исправьте исходный HTML.</p>;
  const table = parsed.table;
  const first =
    table.cells.find(
      (c) => c.row === selection[0][0] && c.column === selection[0][1],
    ) || table.cells[0];
  const lastPoint = selection.at(-1)!;
  const last =
    table.cells.find(
      (c) => c.row === lastPoint[0] && c.column === lastPoint[1],
    ) || first;
  function change(action: () => TableModel) {
    try {
      onChange(serializeTable(action()));
      setError("");
    } catch (e) {
      setError(String(e));
    }
  }
  function select(cell: Cell, extend: boolean) {
    setSelection(
      extend
        ? [selection[0], [cell.row, cell.column]]
        : [[cell.row, cell.column]],
    );
  }
  return (
    <div className="table-editor">
      <div className="toolbar compact">
        <button
          onClick={() =>
            change(() => insertAxis(table, "row", first.row + first.rowspan))
          }
        >
          + Строка
        </button>
        <button
          onClick={() =>
            change(() =>
              insertAxis(table, "column", first.column + first.colspan),
            )
          }
        >
          + Столбец
        </button>
        <button
          onClick={() => change(() => removeAxis(table, "row", first.row))}
        >
          − Строка
        </button>
        <button
          onClick={() =>
            change(() => removeAxis(table, "column", first.column))
          }
        >
          − Столбец
        </button>
        <button onClick={() => change(() => mergeCells(table, first, last))}>
          Объединить
        </button>
        <button onClick={() => change(() => splitCell(table, first))}>
          Разделить
        </button>
      </div>
      <p className="hint">
        Нажмите ячейку для выбора; Shift + нажатие выделяет прямоугольник для
        объединения.
      </p>
      {error && <p className="error">{error}</p>}
      <div className="table-scroll">
        <table>
          <tbody>
            {Array.from({ length: table.rows }, (_, row) => (
              <tr key={row}>
                {table.cells
                  .filter((c) => c.row === row)
                  .sort((a, b) => a.column - b.column)
                  .map((cell) => (
                    <td
                      key={cell.column}
                      rowSpan={cell.rowspan}
                      colSpan={cell.colspan}
                      className={
                        selection.some(
                          ([r, c]) => r === cell.row && c === cell.column,
                        )
                          ? "selected-cell"
                          : ""
                      }
                      onClick={(event) => select(cell, event.shiftKey)}
                    >
                      <textarea
                        aria-label={`Строка ${cell.row + 1}, столбец ${cell.column + 1}`}
                        value={cellText(cell.html)}
                        rows={Math.max(
                          2,
                          cellText(cell.html).split("\n").length,
                        )}
                        onChange={(event) =>
                          change(() => ({
                            ...table,
                            cells: table.cells.map((c) =>
                              c === cell
                                ? { ...c, html: escapeCell(event.target.value) }
                                : c,
                            ),
                          }))
                        }
                      />
                    </td>
                  ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
