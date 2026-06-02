import Link from "next/link";

export default function BusinessNotFound() {
  return (
    <div className="space-y-4 py-12 text-center">
      <h1 className="text-2xl font-semibold">Business not found</h1>
      <p className="text-sm text-neutral-500">
        We couldn&rsquo;t find a business with that id. It may not be in the Philadelphia
        dataset.
      </p>
      <Link
        href="/"
        className="inline-block rounded-md bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-700"
      >
        Back to search
      </Link>
    </div>
  );
}
