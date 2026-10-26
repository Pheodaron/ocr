import type { Block } from "./types";

export function joinBlocks(blocks: Block[]): string {
  const parts: string[] = [];
  let previousKept = false;
  for (const block of blocks) {
    const text = block.content.trim();
    if (!text) {
      previousKept = false;
      continue;
    }
    if (block.type.endsWith("+") && previousKept)
      parts[parts.length - 1] += " " + text;
    else parts.push(text);
    previousKept = true;
  }
  return parts.join("\n\n");
}
