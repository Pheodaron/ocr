import { describe, expect, it } from "vitest";
import {
  insertAxis,
  mergeCells,
  parseTable,
  removeAxis,
  serializeTable,
  splitCell,
} from "./tableModel";

const original =
  '<table><tr><th rowspan="2">0012</th><td>−0,25</td></tr><tr><td></td></tr></table>';

describe("Сложные таблицы", () => {
  it("сохраняет объединения, пустые ячейки и строковые числа", () => {
    const table = parseTable(original);
    expect(parseTable(serializeTable(table))).toEqual(table);
    expect(table.cells[0].html).toBe("0012");
    expect(table.cells[0].rowspan).toBe(2);
  });
  it("вставляет строку внутри объединения и возвращает исходную структуру", () => {
    const table = parseTable(original);
    const inserted = insertAxis(table, "row", 1);
    expect(inserted.cells[0].rowspan).toBe(3);
    expect(parseTable(serializeTable(removeAxis(inserted, "row", 1)))).toEqual(
      table,
    );
  });
  it("объединяет и разделяет ячейки без потери текста", () => {
    const table = parseTable("<table><tr><td>001</td><td>2</td></tr></table>");
    const merged = mergeCells(table, table.cells[0], table.cells[1]);
    expect(merged.cells[0].colspan).toBe(2);
    expect(merged.cells[0].html).toBe("001<br>2");
    expect(splitCell(merged, merged.cells[0]).cells.length).toBe(2);
  });
  it("не принимает объединение, выходящее за границы", () => {
    expect(() =>
      parseTable('<table><tr><td rowspan="3">x</td></tr></table>'),
    ).toThrow();
  });
});
