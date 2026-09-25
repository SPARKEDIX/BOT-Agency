# New chat

## Mission
Create implementation-ready, token-driven UI guidance for New chat that is optimized for consistency, accessibility, and fast delivery across documentation site.

## Brand
- Product/brand: New chat
- URL: https://claude.ai/new
- Audience: developers and technical teams
- Product surface: documentation site

## Style Foundations
- Visual style: structured, tokenized, content-first
- Main font style: `font.family.primary=anthropic-sans`, `font.family.stack=anthropic-sans, system-ui, Segoe UI, Roboto, Helvetica, Arial, PingFang SC, PingFang TC, Hiragino Sans, Apple SD Gothic Neo, Kohinoor Devanagari, Kohinoor Bangla, Kohinoor Telugu, Tamil Sangam MN, Kohinoor Gujarati, Malayalam Sangam MN, Nirmala UI, Noto Sans Devanagari UI, Noto Sans Devanagari, Noto Sans Bengali UI, Noto Sans Bengali, Noto Sans Telugu UI, Noto Sans Telugu, Noto Sans Tamil UI, Noto Sans Tamil, Noto Sans Gujarati UI, Noto Sans Gujarati, Noto Sans Kannada UI, Noto Sans Kannada, Noto Sans Malayalam UI, Noto Sans Malayalam, Thonburi, Leelawadee UI, Noto Sans Thai UI, Noto Sans Thai, Kefa, Ebrima, Noto Sans Ethiopic, Abyssinica SIL, sans-serif, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, PingFang SC, PingFang TC, Hiragino Sans, Apple SD Gothic Neo, sans-serif`, `font.size.base=13px`, `font.weight.base=400`, `font.lineHeight.base=19px`
- Typography scale: `font.size.xs=13px`, `font.size.sm=14px`, `font.size.md=15px`, `font.size.lg=16px`
- Color palette: `color.text.primary=#c3c2b7`, `color.text.secondary=#f0efec`, `color.text.tertiary=#898781`, `color.text.inverse=#6da7ec`, `color.surface.base=#000000`, `color.surface.muted=#151515`, `color.surface.raised=color(srgb 1 1 1 / 0.15)`, `color.surface.strong=color(srgb 0.0823529 0.0823529 0.0823529)`, `color.border.default=color(srgb 1 1 1 / 0.1)`, `color.focus.ring=#5598e7`
- Spacing scale: `space.1=0.5px`, `space.2=2px`, `space.3=3px`, `space.4=4px`, `space.5=8px`, `space.6=10px`, `space.7=12px`, `space.8=38px`
- Radius/shadow/motion tokens: `radius.xs=2px`, `radius.sm=5px`, `radius.md=6px`, `radius.lg=8px` | `motion.duration.instant=60ms`, `motion.duration.fast=450ms`

## Accessibility
- Target: WCAG 2.2 AA
- Keyboard-first interactions required.
- Focus-visible rules required.
- Contrast constraints required.

## Writing Tone
Concise, confident, implementation-focused.

## Rules: Do
- Use semantic tokens, not raw hex values, in component guidance.
- Every component must define states for default, hover, focus-visible, active, disabled, loading, and error.
- Component behavior should specify responsive and edge-case handling.
- Interactive components must document keyboard, pointer, and touch behavior.
- Accessibility acceptance criteria must be testable in implementation.

## Rules: Don't
- Do not allow low-contrast text or hidden focus indicators.
- Do not introduce one-off spacing or typography exceptions.
- Do not use ambiguous labels or non-descriptive actions.
- Do not ship component guidance without explicit state rules.

## Guideline Authoring Workflow
1. Restate design intent in one sentence.
2. Define foundations and semantic tokens.
3. Define component anatomy, variants, interactions, and state behavior.
4. Add accessibility acceptance criteria with pass/fail checks.
5. Add anti-patterns, migration notes, and edge-case handling.
6. End with a QA checklist.

## Required Output Structure
- Context and goals.
- Design tokens and foundations.
- Component-level rules (anatomy, variants, states, responsive behavior).
- Accessibility requirements and testable acceptance criteria.
- Content and tone standards with examples.
- Anti-patterns and prohibited implementations.
- QA checklist.

## Component Rule Expectations
- Include keyboard, pointer, and touch behavior.
- Include spacing and typography token requirements.
- Include long-content, overflow, and empty-state handling.
- Include known page component density: buttons (39), links (28), inputs (3), cards (2), navigation (1).

- Extraction diagnostics: Audience and product surface inference confidence is low; verify generated brand context.

## Quality Gates
- Every non-negotiable rule must use "must".
- Every recommendation should use "should".
- Every accessibility rule must be testable in implementation.
- Teams should prefer system consistency over local visual exceptions.

---

## Implementation Notes (BOT-Agency localhost frontend)
Design intent in one sentence: token-driven dark chat console for the agency backend, keyboard-first and WCAG 2.2 AA.

Token mapping (see `frontend/styles.css`):
- All colors/spacing/radius/motion reference `var(--color-*)` / `var(--space-*)` / `var(--radius-*)` / `var(--motion-*)`, never raw hex in components.
- `color.surface.raised` = `rgba(255,255,255,0.15)` (from `color(srgb 1 1 1 / 0.15)`).
- `color.surface.strong` = `#151515`.
- `color.border.default` = `rgba(255,255,255,0.1)`.
- Base font 13px/400/19px, scale xs–lg as specified. No one-off sizes.

Components: navigation (1), chat cards (messages), inputs (composer + mode select + search), buttons (send/clear/retry/mode), trace card, agent list.
Each defines default, hover, focus-visible, active, disabled, loading, error states.
Responsive: sidebar collapses under 900px. Long content wraps + scrolls. Empty state shows starter prompts.
Keyboard: Tab order header → chat → composer; Enter sends, Shift+Enter newline, Ctrl/Cmd+K focuses composer, Esc clears focus.
Pointer/touch: 44px minimum hit target on send, mode buttons; hover only enhances, never required.

Content/tone: concise, confident, implementation-focused. Example: "NIM busy — retry 2/5 in 4.0s." not "Oopsie, something went wrong!!".

Anti-patterns: no low-contrast tertiary-on-muted body text, no hidden focus, no ambiguous "Click here", no component without state rules.

QA checklist:
- [ ] `py backend/server.py` serves `http://127.0.0.1:8000` with no console errors.
- [ ] Chat round-trip works in auto/direct/agency/pipeline modes.
- [ ] Agents list loads from `/api/agents` (12 roles incl. trading).
- [ ] Focus-visible ring `#5598e7` appears on every interactive element.
- [ ] `prefers-reduced-motion` disables 450ms transitions.
- [ ] Empty, loading, error states render with descriptive text.
- [ ] `py test_frontend.py` passes offline (no NIM key).
