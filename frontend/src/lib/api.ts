// Thin client over the RRS FastAPI service (Phase 6).
// All response types mirror the documented v1 contract exactly.

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8011";

// ---- Response types (match the API contract) ----

export interface SearchResult {
  business_id: string;
  name: string;
  city: string;
  state: string;
  yelp_rating: number;
  yelp_review_count: number;
  rrs: number | null;
}

export interface BusinessDetail {
  business_id: string;
  name: string;
  address: string;
  city: string;
  state: string;
  categories: string; // comma-joined
  yelp_rating: number;
  yelp_review_count: number;
  rrs: number | null;
  rrs_ci_low: number | null;
  rrs_ci_high: number | null;
  pct_flagged: number; // 0..1 fraction
  n_flagged: number;
  n_authentic_reviews: number;
  n_reviews: number;
}

export interface ReviewItem {
  review_id: string;
  stars: number;
  text: string;
  date: string; // ISO string
  p_fake: number | null; // 0..1 or null
  top_signals: string[]; // 0..3 strings
}

/**
 * Thrown when the backend is unreachable or returns a non-OK, non-404 status.
 * 404s are handled by returning `null` from the relevant fetchers.
 */
export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function getJson<T>(path: string): Promise<T> {
  let res: Response;
  try {
    // Live data: never cache at the framework level.
    res = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store" });
  } catch {
    throw new ApiError(
      `Could not reach the RRS API at ${API_BASE_URL}. Is the backend running?`,
      undefined,
    );
  }
  if (!res.ok) {
    throw new ApiError(`API request failed (${res.status}) for ${path}`, res.status);
  }
  return (await res.json()) as T;
}

// ---- Endpoint wrappers ----

export async function searchBusinesses(
  q: string,
  city?: string,
  limit = 20,
): Promise<SearchResult[]> {
  const params = new URLSearchParams({ q, limit: String(limit) });
  if (city && city.trim()) params.set("city", city.trim());
  return getJson<SearchResult[]>(`/businesses/search?${params.toString()}`);
}

/** Returns null on 404 (unknown business). */
export async function getBusiness(id: string): Promise<BusinessDetail | null> {
  try {
    return await getJson<BusinessDetail>(`/businesses/${encodeURIComponent(id)}`);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null;
    throw err;
  }
}

/** Returns null on 404 (unknown business). */
export async function getBusinessReviews(
  id: string,
  opts: { includeFlags?: boolean; limit?: number; offset?: number } = {},
): Promise<ReviewItem[] | null> {
  const { includeFlags = true, limit = 50, offset = 0 } = opts;
  const params = new URLSearchParams({
    include_flags: String(includeFlags),
    limit: String(limit),
    offset: String(offset),
  });
  try {
    return await getJson<ReviewItem[]>(
      `/businesses/${encodeURIComponent(id)}/reviews?${params.toString()}`,
    );
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null;
    throw err;
  }
}
