// Honest-limitations disclaimer. Rendered in the footer and (compact) on detail pages.
// Wording mirrors IMPLEMENTATION_PLAN.md "Honest limitations".

export function Disclaimer({ variant = "footer" }: { variant?: "footer" | "panel" }) {
  if (variant === "panel") {
    return (
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-200">
        <p className="font-medium">A second opinion, not a verdict.</p>
        <p className="mt-1">
          The Real Rating Score and per-review signals are model estimates, not proof.
          There is no ground truth here — the model learns from noisy automated labels, so
          false positives are expected. Genuine people do post short, enthusiastic reviews.
          Treat flagged reviews as worth a closer look, never as confirmed fakes.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <p className="font-medium text-neutral-700 dark:text-neutral-300">
        A second opinion, not a verdict.
      </p>
      <ul className="list-disc space-y-1 pl-5">
        <li>
          No ground truth: the model is trained on noisy weak labels, so every score and
          signal is directional. False positives matter — real people write short, glowing
          reviews too.
        </li>
        <li>
          Built on a static snapshot of the Yelp Open Dataset (~2022). This is a research
          and portfolio project, not a live fraud monitor.
        </li>
        <li>
          We surface <em>signals</em>, not accusations. Reviews are never hidden and no
          individual is labeled fraudulent.
        </li>
      </ul>
    </div>
  );
}
