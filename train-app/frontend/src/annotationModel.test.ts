import { describe, expect, it } from "vitest";
import { joinBlocks } from "./annotationModel";
import type { Block } from "./types";

describe("Порядок фрагментов", () => {
  it("соединяет продолжения абзаца и сохраняет границу следующего", () => {
    const blocks = [
      { id: "1", type: "text", content: "Первая часть", bbox: null },
      { id: "2", type: "text+", content: "продолжение", bbox: null },
      { id: "3", type: "text", content: "Новый абзац", bbox: null },
    ] as Block[];
    expect(joinBlocks(blocks)).toBe("Первая часть продолжение\n\nНовый абзац");
  });
});
