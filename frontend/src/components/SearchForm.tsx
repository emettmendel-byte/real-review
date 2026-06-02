"use client";

import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";

// Search box: name query (required) + optional city. On submit it pushes the
// query into the URL so results render server-side and the search is shareable.

export function SearchForm({
  initialQ = "",
  initialCity = "",
}: {
  initialQ?: string;
  initialCity?: string;
}) {
  const router = useRouter();
  const [q, setQ] = useState(initialQ);
  const [city, setCity] = useState(initialCity);
  const [pending, startTransition] = useTransition();

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = q.trim();
    if (!trimmed) return;
    const params = new URLSearchParams({ q: trimmed });
    if (city.trim()) params.set("city", city.trim());
    startTransition(() => {
      router.push(`/?${params.toString()}`);
    });
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-3 sm:flex-row">
      <div className="flex-1">
        <label htmlFor="q" className="sr-only">
          Business name
        </label>
        <input
          id="q"
          name="q"
          type="text"
          required
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search a business by name (e.g. Reading Terminal Market)"
          className="w-full rounded-md border border-neutral-300 bg-white px-3 py-2 text-sm shadow-sm outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-200 dark:border-neutral-700 dark:bg-neutral-900 dark:focus:ring-sky-900"
        />
      </div>
      <div className="sm:w-48">
        <label htmlFor="city" className="sr-only">
          City (optional)
        </label>
        <input
          id="city"
          name="city"
          type="text"
          value={city}
          onChange={(e) => setCity(e.target.value)}
          placeholder="City (optional)"
          className="w-full rounded-md border border-neutral-300 bg-white px-3 py-2 text-sm shadow-sm outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-200 dark:border-neutral-700 dark:bg-neutral-900 dark:focus:ring-sky-900"
        />
      </div>
      <button
        type="submit"
        disabled={pending}
        className="rounded-md bg-sky-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-sky-700 focus:outline-none focus:ring-2 focus:ring-sky-300 disabled:opacity-60"
      >
        {pending ? "Searching…" : "Search"}
      </button>
    </form>
  );
}
