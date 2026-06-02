import { Stars } from "./Stars";
import { fmtDate, fmtPercent, isFlagged } from "@/lib/format";
import type { ReviewItem } from "@/lib/api";

// A single review. Flagged reviews (p_fake > 0.5) are dimmed but never hidden,
// and carry an explainer panel listing the model's top signals. Signals
// describe the review, they do not accuse the reviewer.

export function ReviewCard({ review }: { review: ReviewItem }) {
  const flagged = isFlagged(review.p_fake);

  return (
    <article
      className={`rounded-lg border p-4 transition ${
        flagged
          ? "border-amber-200 bg-amber-50/50 opacity-70 dark:border-amber-900/50 dark:bg-amber-950/20"
          : "border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-900"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <Stars rating={review.stars} className="text-sm text-amber-500" />
        <time className="text-xs text-neutral-400">{fmtDate(review.date)}</time>
      </div>

      <p className="mt-2 whitespace-pre-line text-sm text-neutral-700 dark:text-neutral-300">
        {review.text}
      </p>

      {flagged && (
        <div className="mt-3 rounded-md border border-amber-200 bg-amber-100/60 p-3 text-xs dark:border-amber-900/50 dark:bg-amber-950/30">
          <div className="font-medium text-amber-900 dark:text-amber-200">
            Why the model finds this review unusual
            {typeof review.p_fake === "number" && (
              <span className="ml-1 font-normal text-amber-700 dark:text-amber-400">
                (estimated {fmtPercent(review.p_fake)} likely suspicious)
              </span>
            )}
          </div>
          {review.top_signals.length > 0 ? (
            <ul className="mt-1 list-disc space-y-0.5 pl-5 text-amber-800 dark:text-amber-300">
              {review.top_signals.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          ) : (
            <p className="mt-1 text-amber-800 dark:text-amber-300">
              No specific signals were surfaced for this review.
            </p>
          )}
          <p className="mt-2 text-amber-700/80 dark:text-amber-400/80">
            These are statistical signals, not proof — treat them as a prompt to read more
            carefully.
          </p>
        </div>
      )}
    </article>
  );
}
