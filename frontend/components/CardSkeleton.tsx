export function CardSkeleton() {
  return (
    <div className="card space-y-4 p-5">
      <div className="flex items-center gap-4">
        <div className="skeleton h-7 w-7 rounded-lg" />
        <div className="flex-1 space-y-2">
          <div className="skeleton h-4 w-1/3 rounded" />
          <div className="skeleton h-3 w-1/4 rounded" />
        </div>
        <div className="skeleton h-16 w-16 rounded-full" />
      </div>
      <div className="skeleton h-8 w-full rounded-lg" />
    </div>
  );
}
