import { describe, expect, test } from 'bun:test'

import { envFlag } from '../logic/env.ts'

describe('envFlag', () => {
  test('recognizes truthy values regardless of case/whitespace', () => {
    for (const v of ['1', 'true', 'yes', 'on', 'TRUE', 'Yes', ' on ']) {
      expect(envFlag(v, false)).toBe(true)
    }
  })

  test('recognizes falsy values regardless of case/whitespace', () => {
    for (const v of ['0', 'false', 'no', 'off', 'FALSE', 'No', ' off ']) {
      expect(envFlag(v, true)).toBe(false)
    }
  })

  test('returns fallback when unset', () => {
    expect(envFlag(undefined, true)).toBe(true)
    expect(envFlag(undefined, false)).toBe(false)
    expect(envFlag('', true)).toBe(true)
    expect(envFlag('   ', false)).toBe(false)
  })

  test('returns fallback for unrecognized garbage', () => {
    expect(envFlag('maybe', true)).toBe(true)
    expect(envFlag('maybe', false)).toBe(false)
    expect(envFlag('2', true)).toBe(true)
    expect(envFlag('enabled', false)).toBe(false)
  })
})
