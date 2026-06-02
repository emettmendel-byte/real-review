import Link from "next/link";
import { notFound } from "next/navigation";
import { getBusiness, getBusinessReviews, ApiError, type ReviewItem } from "@/lib/api";
import { RatingCompare } from "@/components/RatingCompare";
import { PFakeHistogram } from "@/components/PFakeHistogram";
import { Disclaimer } from "@/components/Disclaimer";
import { fmtCount, fmtPercent } from "@/lib/format";

export default async function BusinessPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  let business;
  try {
    business = await getBusiness(id);
  } catch (err) {
    return <ApiErrorPanel err={err} />;
  }
  if (!business) notFound();

  // Pull a page of reviews (no SHAP cost) just for the p_fake distribution.
  let reviews: ReviewItem[] | null = null;
  let reviewsError: unknown = null;
  try {
    reviews = await getBusinessReviews(id, {
      includeFlags: false,
      limit: 200,
      offset: 0,
    });
  } catch (err) {
    reviewsError = err;
  }

  const pFakeValues = (reviews ?? [])
    .map((r) => r.p_fake)
    .filter((v): v is number => typeof v === "number");

  const categories = business.categories
    ? business.categories
        .split(",")
        .map((c) => c.trim())
        .filter(Boolean)
    : [];

  return (
    <div className="space-y-8">
      <div>
        <Link href="/" className="text-sm text-sky-600 hover:underline">
          ← Back to search
        </Link>
      </div>

      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">{business.name}</h1>
        <p className="text-sm text-neutral-500">
          {[business.address, `${business.city}, ${business.state}`]
            .filter(Boolean)
            .join(" · ")}
        </p>
        {categories.length > 0 && (
          <div className="flex flex-wrap gap-1 pt-2">
            {categories.slice(0, 8).map((c) => (
              <span
                key={c}
                className="rounded-full bg-neutral-100 px-2 py-0.5 text-xs text-neutral-600 dark:bg-neutral-800 dark:text-neutral-300"
              >
                {c}
              </span>
            ))}
          </div>
        )}
      </header>

      <section className="space-y-2">
        <RatingCompare
          yelpRating={business.yelp_rating}
          yelpReviewCount={business.yelp_review_count}
          rrs={business.rrs}
          ciLow={business.rrs_ci_low}
          ciHigh={business.rrs_ci_high}
          size="lg"
        />
        <p className="text-xs text-neutral-500">
          The Real Rating Score weights each review by how authentic the model judges it,
          then shrinks toward the average so thin evidence doesn&rsquo;t swing the score.
          The estimated range reflects that uncertainty.
        </p>
      </section>

      <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Reviews analyzed" value={fmtCount(business.n_reviews)} />
        <Stat label="Flagged" value={fmtPercent(business.pct_flagged)} />
        <Stat label="Flagged reviews" value={fmtCount(business.n_flagged)} />
        <Stat label="Authentic reviews" value={fmtCount(business.n_authentic_reviews)} />
      </section>

      <section className="rounded-lg border border-neutral-200 bg-white p-5 dark:border-neutral-800 dark:bg-neutral-900">
        <h2 className="mb-3 text-sm font-medium text-neutral-700 dark:text-neutral-300">
          Where the suspicion sits
        </h2>
        {reviewsError ? (
          <ApiErrorPanel err={reviewsError} inline />
        ) : (
          <PFakeHistogram values={pFakeValues} />
        )}
      </section>

      <div>
        <Link
          href={`/business/${encodeURIComponent(business.business_id)}/reviews`}
          className="inline-block rounded-md bg-sky-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-sky-700"
        >
          See the reviews and their signals →
        </Link>
      </div>

      <Disclaimer variant="panel" />
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-neutral-200 bg-white p-3 text-center dark:border-neutral-800 dark:bg-neutral-900">
      <div className="text-lg font-semibold tabular-nums">{value}</div>
      <div className="text-xs text-neutral-500">{label}</div>
    </div>
  );
}

function ApiErrorPanel({ err, inline = false }: { err: unknown; inline?: boolean }) {
  const message =
    err instanceof ApiError ? err.message : "Something went wrong loading this business.";
  return (
    <div
      className={`rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800 dark:border-rose-900/60 dark:bg-rose-950/30 dark:text-rose-200 ${
        inline ? "" : "mt-6"
      }`}
    >
      {message}
    </div>
  );
}
