/**
 * env — shared boolean env-flag parsing (one source for the TRUE/FALSE regexes).
 *
 * Recognized truthy values: 1/true/yes/on; falsy: 0/false/no/off (case-insensitive,
 * surrounding whitespace trimmed). Anything else (incl. unset) is "unrecognized".
 */
export const TRUE_RE = /^(?:1|true|yes|on)$/i
export const FALSE_RE = /^(?:0|false|no|off)$/i

/** Parse a boolean env var; returns `fallback` when unset/unrecognized. */
export function envFlag(value: string | undefined, fallback: boolean): boolean {
  const v = value?.trim() ?? ''
  if (TRUE_RE.test(v)) return true
  if (FALSE_RE.test(v)) return false
  return fallback
}
