import { Stars } from "./Stars";
import { fmtRating } from "@/lib/format";

// Side-by-side Yelp rating vs Real Rating Score — the core comparison.
// `ciLow`/`ciHigh` render the RRS as a range, reinforcing it is an estimate.

export function RatingCompare({
  yelpRating,
  yelpReviewCount,
  rrs,
  ciLow,
  ciHigh,
  size = "lg",
}: {
  yelpRating: number;
  yelpReviewCount?: number;
  rrs: number | null;
  ciLow?: number | null;
  ciHigh?: number | null;
  size?: "sm" | "lg";
}) {
  const big = size === "lg";
  const numberCls = big ? "text-4xl font-semibold" : "text-2xl font-semibold";
  const starCls = big ? "text-xl" : "text-base";

  const hasCi =
    typeof ciLow === "number" && typeof ciHigh === "number" && rrs !== null;

  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="rounded-lg border border-neutral-200 bg-white p-4 dark:border-neutral-800 dark:bg-neutral-900">
        <div className="text-xs font-medium uppercase tracking-wide text-neutral-500">
          Yelp rating
        </div>
        <div className={`mt-1 ${numberCls} text-neutral-900 dark:text-neutral-100`}>
          {fmtRating(yelpRating)}
        </div>
        <Stars rating={yelpRating} className={`${starCls} text-amber-500`} />
        {typeof yelpReviewCount === "number" && (
          <div className="mt-1 text-xs text-neutral-500">
            {yelpReviewCount.toLocaleString("en-US")} reviews
          </div>
        )}
      </div>

      <div className="rounded-lg border border-sky-200 bg-sky-50 p-4 dark:border-sky-900 dark:bg-sky-950/40">
        <div className="text-xs font-medium uppercase tracking-wide text-sky-700 dark:text-sky-400">
          Real Rating Score
        </div>
        <div className={`mt-1 ${numberCls} text-sky-900 dark:text-sky-100`}>
          {fmtRating(rrs)}
        </div>
        <Stars rating={rrs} className={`${starCls} text-sky-500`} />
        {hasCi ? (
          <div className="mt-1 text-xs text-sky-700 dark:text-sky-400">
            estimated range {fmtRating(ciLow)}–{fmtRating(ciHigh)}
          </div>
        ) : (
          <div className="mt-1 text-xs text-neutral-500">
            {rrs === null ? "not yet scored" : "estimate"}
          </div>
        )}
      </div>
    </div>
  );
}
