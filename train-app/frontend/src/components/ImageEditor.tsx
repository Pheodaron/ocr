import { useRef, useState } from "react";
import type { PointerEvent } from "react";
import type { Block } from "../types";

type Drag = {
  mode: "move" | "resize" | "create" | "pan";
  id?: string;
  x: number;
  y: number;
  box?: number[];
  left?: number;
  top?: number;
};

export function ImageEditor({
  url,
  width,
  height,
  blocks,
  selected,
  onSelect,
  onChange,
  readOnly = false,
}: {
  url: string;
  width: number;
  height: number;
  blocks: Block[];
  selected: string;
  onSelect: (id: string) => void;
  onChange: (blocks: Block[]) => void;
  readOnly?: boolean;
}) {
  const [zoom, setZoom] = useState(1),
    [show, setShow] = useState(true),
    [adding, setAdding] = useState(false);
  const drag = useRef<Drag | null>(null),
    viewport = useRef<HTMLDivElement>(null),
    svg = useRef<SVGSVGElement>(null);
  function point(event: PointerEvent): [number, number] {
    const matrix = svg.current?.getScreenCTM();
    if (!matrix) return [0, 0];
    const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(
      matrix.inverse(),
    );
    return [
      Math.max(0, Math.min(1000, (p.x / width) * 1000)),
      Math.max(0, Math.min(1000, (p.y / height) * 1000)),
    ];
  }
  function begin(event: PointerEvent, mode: Drag["mode"], item?: Block) {
    event.preventDefault();
    event.stopPropagation();
    const [x, y] =
      mode === "pan" ? [event.clientX, event.clientY] : point(event);
    drag.current = {
      mode,
      id: item?.id,
      box: item?.bbox || undefined,
      x,
      y,
      left: viewport.current?.scrollLeft,
      top: viewport.current?.scrollTop,
    };
    svg.current?.setPointerCapture(event.pointerId);
    if (item) onSelect(item.id);
    if (mode === "create") {
      const id = crypto.randomUUID();
      drag.current.id = id;
      onChange([
        ...blocks,
        {
          id,
          type: "text",
          content: "",
          bbox: [x, y, Math.min(1000, x + 1), Math.min(1000, y + 1)],
        },
      ]);
      onSelect(id);
    }
  }
  function move(event: PointerEvent) {
    const active = drag.current;
    if (!active) return;
    if (active.mode === "pan" && viewport.current) {
      viewport.current.scrollLeft = active.left! - (event.clientX - active.x);
      viewport.current.scrollTop = active.top! - (event.clientY - active.y);
      return;
    }
    const [x, y] = point(event);
    let box: number[];
    if (active.mode === "create")
      box = [
        Math.min(x, active.x),
        Math.min(y, active.y),
        Math.max(x, active.x),
        Math.max(y, active.y),
      ];
    else if (active.mode === "resize")
      box = [
        active.box![0],
        active.box![1],
        Math.max(active.box![0] + 1, x),
        Math.max(active.box![1] + 1, y),
      ];
    else {
      const [left, top, right, bottom] = active.box!;
      const dx = Math.max(-left, Math.min(1000 - right, x - active.x));
      const dy = Math.max(-top, Math.min(1000 - bottom, y - active.y));
      box = [left + dx, top + dy, right + dx, bottom + dy];
    }
    onChange(
      blocks.map((item) =>
        item.id === active.id ? { ...item, bbox: box.map(Math.round) } : item,
      ),
    );
  }
  return (
    <div className="image-pane">
      <div className="toolbar compact">
        <button onClick={() => setZoom(Math.max(0.5, zoom - 0.25))}>−</button>
        <span>{Math.round(zoom * 100)}%</span>
        <button onClick={() => setZoom(Math.min(5, zoom + 0.25))}>+</button>
        <label>
          <input
            type="checkbox"
            checked={show}
            onChange={(event) => setShow(event.target.checked)}
          />{" "}
          Рамки
        </label>
        <button
          hidden={readOnly}
          className={adding ? "active" : ""}
          onClick={() => setAdding(!adding)}
        >
          + Область
        </button>
      </div>
      <div className="image-viewport" ref={viewport}>
        <svg
          ref={svg}
          viewBox={`0 0 ${width} ${height}`}
          style={{
            width: `${zoom * 100}%`,
            cursor: adding ? "crosshair" : "grab",
          }}
          onPointerDown={(event) => begin(event, adding ? "create" : "pan")}
          onPointerMove={move}
          onPointerUp={() => {
            drag.current = null;
            setAdding(false);
          }}
          onPointerCancel={() => {
            drag.current = null;
          }}
        >
          <image href={url} width={width} height={height} />
          {show &&
            blocks
              .filter((item) => item.bbox)
              .map((item) => {
                const box = item.bbox!;
                const x = (box[0] * width) / 1000,
                  y = (box[1] * height) / 1000;
                const w = ((box[2] - box[0]) * width) / 1000,
                  h = ((box[3] - box[1]) * height) / 1000;
                return (
                  <g key={item.id}>
                    <rect
                      x={x}
                      y={y}
                      width={w}
                      height={h}
                      className={`annotation-box ${selected === item.id ? "selected" : ""}`}
                      onPointerDown={(event) =>
                        begin(
                          event,
                          readOnly ? "pan" : "move",
                          readOnly ? undefined : item,
                        )
                      }
                    />
                    <text
                      x={x + 4}
                      y={Math.max(18, y - 5)}
                      fontSize={width / 70}
                      className="box-label"
                      pointerEvents="none"
                    >
                      {blocks.indexOf(item) + 1} · {item.type}
                    </text>
                    {!readOnly && selected === item.id && (
                      <rect
                        x={x + w - width / 140}
                        y={y + h - width / 140}
                        width={width / 70}
                        height={width / 70}
                        className="resize-handle"
                        onPointerDown={(event) => begin(event, "resize", item)}
                      />
                    )}
                  </g>
                );
              })}
        </svg>
      </div>
      <p className="hint">
        Перетаскивайте изображение для перемещения. Рамки: движение за область,
        размер — за нижний правый угол. Координаты относятся к изображению.
      </p>
    </div>
  );
}
