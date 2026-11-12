# Change log fragments

Each work package writes `changelog.d/<package>.md` rather than editing shared
release notes. Use optional `## Added`, `## Changed`, `## Fixed`, `## Removed`
and `## Migration` sections, each once, with user-facing bullets.

At release integration, fold fragments into the corresponding `###` sections
under `[Unreleased]` in natural package order, then rename that entry to the
release and date. Carry migration notes with that release. Remove consumed
fragments only after checking the folded text. Keep this README.

This follows pyflightstream's fragment convention, whose retired
`scripts/assemble_changelog.py` and `changelog.d/README.md` are retained in its
history before commit `6cc55dff7`. The first cfdbender release condensed legacy
free-form fragments into honest final capabilities and migration notes.
The changelog is not a verification receipt or a promise that pending SRS
requirements have been accepted.
