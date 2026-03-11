import { logger } from "./logger";

/**
 * Parse a JSON response from any LLM provider, handling common formatting
 * quirks like markdown code fences, trailing text, and partial output.
 *
 * Returns the parsed object/array, or null if parsing fails entirely.
 */
export function parseJsonResponse(text: string): any | null {
  if (typeof text !== "string" || !text.trim()) {
    return null;
  }

  const trimmed = text.trim();

  // 1. Try raw JSON parse first
  try {
    return JSON.parse(trimmed);
  } catch {
    // continue to fallbacks
  }

  // 2. Try extracting from markdown code fences (```json ... ``` or ``` ... ```)
  const fenceMatch = trimmed.match(/```(?:json)?\s*\n?([\s\S]*?)```/);
  if (fenceMatch) {
    try {
      return JSON.parse(fenceMatch[1].trim());
    } catch {
      // continue to next fallback
    }
  }

  // 3. Try finding first { or [ and matching to closing bracket
  const startObj = trimmed.indexOf("{");
  const startArr = trimmed.indexOf("[");

  let startIndex: number;
  let openChar: string;
  let closeChar: string;

  if (startObj === -1 && startArr === -1) {
    logger.warn(`parseJsonResponse: no JSON structure found in: ${trimmed}`);
    return null;
  } else if (startArr === -1 || (startObj !== -1 && startObj < startArr)) {
    startIndex = startObj;
    openChar = "{";
    closeChar = "}";
  } else {
    startIndex = startArr;
    openChar = "[";
    closeChar = "]";
  }

  let depth = 0;
  let inString = false;
  let escape = false;

  for (let i = startIndex; i < trimmed.length; i++) {
    const ch = trimmed[i];

    if (escape) {
      escape = false;
      continue;
    }

    if (ch === "\\") {
      if (inString) escape = true;
      continue;
    }

    if (ch === '"') {
      inString = !inString;
      continue;
    }

    if (inString) continue;

    if (ch === openChar) {
      depth++;
    } else if (ch === closeChar) {
      depth--;
      if (depth === 0) {
        try {
          return JSON.parse(trimmed.slice(startIndex, i + 1));
        } catch {
          break;
        }
      }
    }
  }

  logger.warn(
    `parseJsonResponse: failed to parse JSON from LLM response: ${trimmed.slice(0, 200)}`,
  );
  return null;
}
