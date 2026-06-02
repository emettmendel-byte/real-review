"use client";

import { useState } from "react";
import { getBusinessReviews, ApiError, type ReviewItem } from "@/lib/api";
import { isFlagged } from "@/lib/format";
import { ReviewCard } from "./ReviewCard";

const PAGE_SIZE = 50;

// Interactive review list: a "show flagged reviews" toggle (flagged ones are
// dimmed but visible by default) plus a "load more" pager over the API's
// limit/offset.

export function ReviewList({
  businessId,
  initial,
}: {
  businessId: string;
  initial: ReviewItem[];
}) {
  const [reviews, setReviews] = useState<ReviewItem[]>(initial);
  const [showFlagged, setShowFlagged] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(initial.length === PAGE_SIZE);

  const flaggedCount = reviews.filter((r) => isFlagged(r.p_fake)).length;
  const visible = showFlagged ? reviews : reviews.filter((r) => !isFlagged(r.p_fake));

  async function loadMore() {
    setLoading(true);
    setError(null);
    try {
      const next = await getBusinessReviews(businessId, {
        includeFlags: true,
        limit: PAGE_SIZE,
        offset: reviews.length,
      });
      if (next === null) {
        setHasMore(false);
      } else {
        setReviews((prev) => [...prev, ...next]);
        setHasMore(next.length === PAGE_SIZE);
      }
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Could not load more reviews.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-neutral-500">
          Showing {visible.length} of {reviews.length} loaded review
          {reviews.length === 1 ? "" : "s"}
          {flaggedCount > 0 && (
            <span>
              {" "}
              · {flaggedCount} flagged
            </span>
          )}
        </p>
        <label className="flex cursor-pointer items-center gap-2 text-sm text-neutral-600 dark:text-neutral-300">
          <input
            type="checkbox"
            checked={showFlagged}
            onChange={(e) => setShowFlagged(e.target.checked)}
            className="h-4 w-4 rounded border-neutral-300 text-sky-600 focus:ring-sky-500"
          />
          Show flagged reviews
        </label>
      </div>

      {visible.length === 0 ? (
        <p className="text-sm text-neutral-500">
          {reviews.length === 0
            ? "No reviews found for this business."
            : "All loaded reviews are flagged. Toggle them back on to read them — they are dimmed, not hidden."}
        </p>
      ) : (
        <div className="space-y-3">
          {visible.map((r) => (
            <ReviewCard key={r.review_id} review={r} />
          ))}
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800 dark:border-rose-900/60 dark:bg-rose-950/30 dark:text-rose-200">
          {error}
        </div>
      )}

      {hasMore && (
        <div className="flex justify-center">
          <button
            type="button"
            onClick={loadMore}
            disabled={loading}
            className="rounded-md border border-neutral-300 bg-white px-4 py-2 text-sm font-medium shadow-sm transition hover:bg-neutral-50 disabled:opacity-60 dark:border-neutral-700 dark:bg-neutral-900 dark:hover:bg-neutral-800"
          >
            {loading ? "Loading…" : "Load more reviews"}
          </button>
        </div>
      )}
    </div>
  );
}
