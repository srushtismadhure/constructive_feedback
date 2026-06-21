interface StatusMetricProps {
  label: string;
  value: string | number;
  tone?: "default" | "success" | "warning";
}

export default function StatusMetric({ label, value, tone = "default" }: StatusMetricProps) {
  const valueColor =
    tone === "success" ? "text-success-green" : tone === "warning" ? "text-warning-yellow" : "text-off-white";

  return (
    <div className="border-b border-white/10 py-3 font-mono last:border-b-0">
      <dt className="text-[9px] uppercase tracking-[0.16em] text-muted-text">{label}</dt>
      <dd className={`mt-1 text-xs font-semibold uppercase tracking-[0.08em] ${valueColor}`}>{value}</dd>
    </div>
  );
}
