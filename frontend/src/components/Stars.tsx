// Accessible star glyph rendering for a 0–5 rating (supports halves).

export function Stars({
  rating,
  className = "",
}: {
  rating: number | null | undefined;
  className?: string;
}) {
  if (rating === null || rating === undefined || Number.isNaN(rating)) {
    return <span className={className}>—</span>;
  }
  const rounded = Math.round(rating * 2) / 2;
  const full = Math.floor(rounded);
  const half = rounded - full === 0.5;
  const stars: string[] = [];
  for (let i = 0; i < 5; i++) {
    if (i < full) stars.push("★");
    else if (i === full && half) stars.push("◐");
    else stars.push("☆");
  }
  return (
    <span
      className={className}
      role="img"
      aria-label={`${rating.toFixed(1)} out of 5 stars`}
    >
      <span aria-hidden="true">{stars.join("")}</span>
    </span>
  );
}
