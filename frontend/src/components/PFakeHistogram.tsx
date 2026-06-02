import { FLAG_THRESHOLD } from "@/lib/format";

// Hand-rolled CSS bar histogram of p_fake across a business's reviews.
// 10 fixed bins over [0, 1]; no charting dependency. Bins at/above the flag
// threshold (0.5) are tinted to show where the model is most suspicious.

export function PFakeHistogram({ values }: { values: number[] }) {
  const BIN_COUNT = 10;
  const bins = new Array<number>(BIN_COUNT).fill(0);
  let scored = 0;
  for (const v of values) {
    if (typeof v !== "number" || Number.isNaN(v)) continue;
    scored += 1;
    // clamp to [0, 1) so 1.0 lands in the last bin
    const idx = Math.min(BIN_COUNT - 1, Math.max(0, Math.floor(v * BIN_COUNT)));
    bins[idx] += 1;
  }
  const max = Math.max(1, ...bins);

  if (scored === 0) {
    return (
      <p className="text-sm text-neutral-500">
        No model scores are available for these reviews yet.
      </p>
    );
  }

  return (
    <div>
      <div
        className="flex h-44 items-end gap-1"
        role="img"
        aria-label={`Distribution of estimated fake probability across ${scored} reviews`}
      >
        {bins.map((count, i) => {
          const lo = i / BIN_COUNT;
          const hi = (i + 1) / BIN_COUNT;
          const heightPct = (count / max) * 100;
          const flaggedBin = lo >= FLAG_THRESHOLD;
          return (
            <div key={i} className="flex flex-1 flex-col items-center justify-end">
              <span className="mb-1 text-[10px] tabular-nums text-neutral-400">
                {count > 0 ? count : ""}
              </span>
              <div
                className={`w-full rounded-t ${
                  flaggedBin
                    ? "bg-rose-400 dark:bg-rose-500"
                    : "bg-sky-400 dark:bg-sky-500"
                }`}
                style={{ height: `${Math.max(count > 0 ? 3 : 0, heightPct)}%` }}
                title={`p_fake ${lo.toFixed(1)}–${hi.toFixed(1)}: ${count} review${
                  count === 1 ? "" : "s"
                }`}
              />
            </div>
          );
        })}
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-neutral-400">
        <span>0.0 (looks authentic)</span>
        <span>0.5</span>
        <span>1.0 (looks suspicious)</span>
      </div>
      <p className="mt-2 text-xs text-neutral-500">
        Estimated fake probability for {scored.toLocaleString("en-US")} sampled reviews.
        Higher bars on the right (tinted) mean more reviews the model finds suspicious.
      </p>
    </div>
  );
}
