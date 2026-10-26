export function LossChart({ samples }: { samples: unknown[] }) {
  const points = samples.filter(
    (value): value is { step: number; loss: number } =>
      typeof value === "object" &&
      value !== null &&
      "step" in value &&
      "loss" in value &&
      typeof value.step === "number" &&
      typeof value.loss === "number",
  );
  if (points.length < 2) return null;
  const low = Math.min(...points.map((p) => p.loss)),
    high = Math.max(...points.map((p) => p.loss));
  const left = points[0].step,
    right = points.at(-1)!.step;
  const line = points
    .map(
      (point) =>
        `${50 + ((point.step - left) / Math.max(1, right - left)) * 620},${170 - ((point.loss - low) / Math.max(0.00001, high - low)) * 130}`,
    )
    .join(" ");
  return (
    <figure className="loss-chart">
      <figcaption>
        Ошибка обучения по шагам · для выбора модели используйте также валидацию
      </figcaption>
      <svg viewBox="0 0 720 215" role="img" aria-label="График ошибки обучения">
        <path d="M50 20V180H680" stroke="#b6c3cf" fill="none" />
        <polyline points={line} stroke="#216452" fill="none" strokeWidth="2" />
        <text x="4" y="40">
          {high.toFixed(3)}
        </text>
        <text x="4" y="175">
          {low.toFixed(3)}
        </text>
        <text x="50" y="205">
          {left}
        </text>
        <text x="640" y="205">
          {right}
        </text>
      </svg>
    </figure>
  );
}
