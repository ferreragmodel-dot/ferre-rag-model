export function TitleBar() {
  return (
    <div className="pointer-events-none fixed inset-x-0 top-0 z-50">
      <div className="mx-auto flex w-full max-w-[680px] justify-center px-4 pt-6 sm:px-8">
        <div className="inline-flex h-11 items-center justify-center rounded-full border border-border px-5 shadow-sm" style={{ backgroundColor: "#f7f7f5" }}>
          <h1
            style={{
              fontFamily: 'Didot, "Bodoni 72", "Times New Roman", serif',
              fontWeight: 700,
              fontStyle: "normal",
              fontSize: "20px",
              lineHeight: "100%",
              letterSpacing: "0px",
              textTransform: "uppercase",
            }}
          >
            Gianfranco Ferré Archive
          </h1>
        </div>
      </div>
    </div>
  );
}
