import Link from "next/link";
import { searchBusinesses, ApiError, type SearchResult } from "@/lib/api";
import { SearchForm } from "@/components/SearchForm";
import { Stars } from "@/components/Stars";
import { fmtRating } from "@/lib/format";

export default async function Home({
  searchParams,
}: {
  searchParams: Promise<{ q?: string; city?: string }>;
}) {
  const { q, city } = await searchParams;
  const query = (q ?? "").trim();
  const cityFilter = (city ?? "").trim();
  const hasQuery = query.length > 0;

  return (
    <div className="space-y-8">
      <section className="space-y-3">
        <h1 className="text-2xl font-semibold tracking-tight">
          How real are a business&rsquo;s reviews?
        </h1>
        <p className="max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
          Search a Philadelphia-area business to see its Yelp rating next to a Real Rating
          Score that down-weights reviews our model finds suspicious. It&rsquo;s a second
          opinion, not a verdict.
        </p>
        <SearchForm initialQ={query} initialCity={cityFilter} />
      </section>

      {hasQuery ? (
        <Results query={query} city={cityFilter} />
      ) : (
        <p className="text-sm text-neutral-500">
          Enter a business name to get started.
        </p>
      )}
    </div>
  );
}

async function Results({ query, city }: { query: string; city: string }) {
  // Only the data fetch lives in try/catch; JSX is constructed afterwards so
  // rendering errors aren't silently swallowed (react-hooks/error-boundaries).
  let results: SearchResult[];
  try {
    results = await searchBusinesses(query, city || undefined, 20);
  } catch (err) {
    const message =
      err instanceof ApiError ? err.message : "Something went wrong while searching.";
    return (
      <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800 dark:border-rose-900/60 dark:bg-rose-950/30 dark:text-rose-200">
        {message}
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <p className="text-sm text-neutral-500">
        No businesses found for &ldquo;{query}&rdquo;
        {city ? ` in ${city}` : ""}. Try a different name or drop the city filter.
      </p>
    );
  }

  return (
    <section className="space-y-2">
      <h2 className="text-sm font-medium text-neutral-500">
        {results.length} result{results.length === 1 ? "" : "s"}
      </h2>
      <ul className="divide-y divide-neutral-200 overflow-hidden rounded-lg border border-neutral-200 bg-white dark:divide-neutral-800 dark:border-neutral-800 dark:bg-neutral-900">
        {results.map((b) => (
          <li key={b.business_id}>
            <Link
              href={`/business/${encodeURIComponent(b.business_id)}`}
              className="flex flex-col gap-2 px-4 py-3 transition hover:bg-neutral-50 sm:flex-row sm:items-center sm:justify-between dark:hover:bg-neutral-800/60"
            >
              <div className="min-w-0">
                <div className="truncate font-medium">{b.name}</div>
                <div className="text-xs text-neutral-500">
                  {b.city}, {b.state} · {b.yelp_review_count.toLocaleString("en-US")}{" "}
                  reviews
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-6">
                <RatingCell label="Yelp" value={b.yelp_rating} tone="amber" />
                <RatingCell label="RRS" value={b.rrs} tone="sky" />
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}

function RatingCell({
  label,
  value,
  tone,
}: {
  label: string;
  value: number | null;
  tone: "amber" | "sky";
}) {
  const star = tone === "amber" ? "text-amber-500" : "text-sky-500";
  return (
    <div className="text-right">
      <div className="text-[10px] uppercase tracking-wide text-neutral-400">{label}</div>
      <div className="flex items-center justify-end gap-1">
        <span className="text-sm font-semibold tabular-nums">{fmtRating(value)}</span>
        <Stars rating={value} className={`text-xs ${star}`} />
      </div>
    </div>
  );
}
