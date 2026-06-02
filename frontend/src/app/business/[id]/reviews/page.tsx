import Link from "next/link";
import { notFound } from "next/navigation";
import { getBusiness, getBusinessReviews, ApiError } from "@/lib/api";
import { ReviewList } from "@/components/ReviewList";
import { Disclaimer } from "@/components/Disclaimer";

export default async function ReviewsPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  let business;
  try {
    business = await getBusiness(id);
  } catch (err) {
    return <ErrorPanel err={err} />;
  }
  if (!business) notFound();

  let initialReviews;
  try {
    initialReviews = await getBusinessReviews(id, {
      includeFlags: true,
      limit: 50,
      offset: 0,
    });
  } catch (err) {
    return <ErrorPanel err={err} />;
  }
  // Should not be null here since the business exists, but stay defensive.
  if (initialReviews === null) notFound();

  return (
    <div className="space-y-6">
      <div>
        <Link
          href={`/business/${encodeURIComponent(id)}`}
          className="text-sm text-sky-600 hover:underline"
        >
          ← Back to {business.name}
        </Link>
      </div>

      <header className="space-y-1">
        <h1 className="text-xl font-semibold tracking-tight">
          Reviews for {business.name}
        </h1>
        <p className="text-sm text-neutral-500">
          Flagged reviews are dimmed but kept in place, each with the signals that drew the
          model&rsquo;s attention. Nothing is hidden, and no reviewer is accused.
        </p>
      </header>

      <ReviewList businessId={id} initial={initialReviews} />

      <Disclaimer variant="panel" />
    </div>
  );
}

function ErrorPanel({ err }: { err: unknown }) {
  const message =
    err instanceof ApiError ? err.message : "Something went wrong loading these reviews.";
  return (
    <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800 dark:border-rose-900/60 dark:bg-rose-950/30 dark:text-rose-200">
      {message}
    </div>
  );
}
