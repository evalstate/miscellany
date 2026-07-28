# Presentation publishing

This directory stages and deploys the static Hugging Face Space at
`evalstate/presentations`.

The GitHub repository is the source of truth. The Space is a disposable,
write-only projection of the generated static site and should not contain
remote-only source content.

## Files

- `manifest.json` — selected decks and candidate decks.
- `scripts/build.sh` — build configured decks and assemble `publish/site/` from the manifest.
- `scripts/verify.sh` — check that published entries exist in `publish/site/`.
- `scripts/deploy-space.sh` — upload `publish/site/` to the Space.
- `site/` — generated output; do not edit by hand.

## Build

```bash
publish/scripts/build.sh
publish/scripts/verify.sh
```

Manifest entries can define build commands. These run before staging so ignored
artifacts such as Slidev `dist/` directories are regenerated from tracked source
and lockfiles rather than reused from the working tree.

To include `candidate` manifest entries locally:

```bash
publish/scripts/build.sh --include-candidates
```

## Provenance

Each build writes `publish/site/publish-manifest.json` and links it from the
index footer. It records:

- the current `evalstate/miscellany` git commit;
- whether the workspace had local changes at build time;
- for each published deck, its source path and entry file;
- whether each local source path was dirty.

This keeps published content tied back to its local Git source.

## Deploy

```bash
publish/scripts/deploy-space.sh
```

Deployment performs a fresh build and verification before uploading, so an old
ignored `publish/site/` tree cannot be published accidentally.

Override the target Space or commit message:

```bash
SPACE=evalstate/presentations COMMIT_MESSAGE="Add NYC deck" publish/scripts/deploy-space.sh
```

`deploy-space.sh` uses `--delete '*'`, so the Space repo becomes exactly the
contents of `publish/site/`. This is intentional: do not make source changes
directly in the Space.

## Adding a deck

1. Add or update an entry in `manifest.json`.
2. If the entry publishes generated output, add its reproducible build commands.
3. Set `status` to `published` when it should appear on the public index.
4. Run `publish/scripts/build.sh && publish/scripts/verify.sh`.
5. Run `publish/scripts/deploy-space.sh` when ready.
