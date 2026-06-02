// Small pure formatting helpers shared across pages.

export function fmtRating(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(2);
}

export function fmtPercent(fraction: number | null | undefined): string {
  if (fraction === null || fraction === undefined || Number.isNaN(fraction)) return "—";
  return `${(fraction * 100).toFixed(1)}%`;
}

export function fmtCount(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString("en-US");
}

export function fmtDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/** A review is treated as flagged if the model's p_fake exceeds 0.5. */
export const FLAG_THRESHOLD = 0.5;

export function isFlagged(pFake: number | null | undefined): boolean {
  return typeof pFake === "number" && pFake > FLAG_THRESHOLD;
}
