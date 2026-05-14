import type { JSX } from "react";

/**
 * Parses inline citation markers like [1] or [1, 2] in a text string and
 * replaces them with interactive buttons that call onCitationClick.
 */
export function renderWithCitations(
  text: string,
  onCitationClick: (num: number) => void,
  activeCitation: number | null,
): (string | JSX.Element)[] {
  const parts: (string | JSX.Element)[] = [];
  const regex = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    const citationNumbers = match[1].split(",").map((n) => parseInt(n.trim(), 10));
    const citationButtons = citationNumbers.map((num, idx) => (
      <button
        key={`citation-${match!.index}-${idx}`}
        onClick={() => onCitationClick(num)}
        className={`inline cursor-pointer font-semibold text-xs transition-colors mx-0.5 ${
          activeCitation === num
            ? "text-blue-600"
            : "text-blue-500 hover:text-blue-700 hover:underline"
        }`}
      >
        [{num}]{idx < citationNumbers.length - 1 ? ", " : ""}
      </button>
    ));

    parts.push(...citationButtons);
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts;
}
