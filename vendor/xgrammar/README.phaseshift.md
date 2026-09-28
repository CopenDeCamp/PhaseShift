# PhaseShift vendored XGrammar

Base upstream:
  XGrammar v0.2.5
  commit 2ea71da

PhaseShift patches:
  C++20/libc++ NamedGrammar incomplete-type compatibility backport

Source:
  upstream XGrammar main (`EmptyNamedGrammars()` default-argument fix)

Behavior change:
  none

Reason:
  remove PhaseShift-specific C++17 compile exception and keep native code uniformly C++20

## Patches

- `patches/0001-cxx20-named-grammar-incomplete-type.patch`

`VERSION` stays at the upstream base (`v0.2.5` / `2ea71da`). Patch information is
tracked here rather than in the version string, so the patch can simply be dropped
when upgrading to an upstream release that already contains the fix.

Apply with:

```bash
git apply vendor/xgrammar/patches/0001-cxx20-named-grammar-incomplete-type.patch
```
