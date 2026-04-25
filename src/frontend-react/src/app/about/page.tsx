import { TitleBar } from "@/components/TitleBar";

const SERIF: React.CSSProperties = {
  fontFamily: 'Didot, "Bodoni 72", "Times New Roman", serif',
};

export default function AboutPage() {
  return (
    <main className="min-h-screen bg-[#f7f7f5]">
      <TitleBar />

      {/* Hero */}
      <section className="mx-auto max-w-6xl px-6 pb-20 pt-28 sm:px-10 lg:grid lg:grid-cols-2 lg:gap-20">
        <h1
          className="mb-10 text-4xl uppercase leading-tight tracking-tight lg:mb-0 lg:text-5xl"
          style={SERIF}
        >
          A Living Space<br />
          for Discovering<br />
          Fashion Heritage
        </h1>

        <div className="space-y-6 text-sm leading-relaxed text-foreground/75">
          <p>
            Gianfranco Ferré was one of the most influential designers of the late twentieth
            century, known for combining architectural precision, material experimentation,
            and a refined vision of elegance. His work moved across fashion, design, drawing,
            and cultural research, leaving behind a rich creative legacy that continues to
            inspire new generations.
          </p>
          <p>
            This digital platform offers a new way to explore the archive through artificial
            intelligence and interactive research tools. Rather than presenting the archive as
            a static collection, it becomes a dynamic environment where images, garments,
            sketches, materials, colors, and ideas can be navigated through multiple
            perspectives.
          </p>
          <p>
            Visitors can search through natural language, uncover hidden connections between
            looks and garments, explore recurring shapes, textures, and palettes, and follow
            unexpected paths through clusters of related works. The goal is not only to
            retrieve what is already known, but to open space for new interpretations,
            associations, and moments of serendipitous discovery.
          </p>
        </div>
      </section>

      {/* Three features */}
      <section className="border-t border-border">
        <div className="mx-auto max-w-6xl px-6 py-16 sm:px-10 lg:grid lg:grid-cols-3 lg:gap-12">
          {[
            {
              title: "Search intuitively",
              body: "Explore the archive through natural language and fluid visual navigation.",
            },
            {
              title: "Reveal connections",
              body: "Discover relationships between looks, garments, materials, colors, and recurring forms.",
            },
            {
              title: "Enable serendipity",
              body: "Move beyond search results and encounter unexpected inspirations across the collection.",
            },
          ].map(({ title, body }) => (
            <div key={title} className="mb-10 lg:mb-0">
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide">
                {title}
              </h2>
              <p className="text-sm leading-relaxed text-foreground/65">{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Credits */}
      <section className="border-t border-border">
        <div className="mx-auto max-w-6xl px-6 py-14 sm:px-10 lg:grid lg:grid-cols-3 lg:gap-12">
          <div className="mb-10 lg:mb-0">
            <h3 className="mb-3 text-[11px] uppercase tracking-widest text-foreground/45">
              Collaborators
            </h3>
            <ul className="space-y-1 text-sm text-foreground/75">
              {["Asia Capezzuoli", "Stefan Golic", "Filippo Longhi", "Jackson Webster", "Cecilia Zheng"].map(
                (name) => <li key={name}>{name}</li>
              )}
            </ul>
          </div>

          <div className="mb-10 lg:mb-0">
            <h3 className="mb-3 text-[11px] uppercase tracking-widest text-foreground/45">
              Professors
            </h3>
            <ul className="space-y-1 text-sm text-foreground/75">
              {["Marco Brambilla", "Pavlos Protopapas", "Chris Gumb"].map(
                (name) => <li key={name}>{name}</li>
              )}
            </ul>
          </div>

          <div>
            <h3 className="mb-3 text-[11px] uppercase tracking-widest text-foreground/45">
              Ferré Center Researchers
            </h3>
            <ul className="space-y-1 text-sm text-foreground/75">
              {["Angelica Vandi", "Federica Vacca"].map(
                (name) => <li key={name}>{name}</li>
              )}
            </ul>
          </div>
        </div>
      </section>
    </main>
  );
}
