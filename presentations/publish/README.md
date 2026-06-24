# Presentation publishing

This directory stages and deploys the static Hugging Face Space at
`evalstate/presentations`.

The bucket plan was intentionally dropped: static Spaces cannot mount buckets,
and bucket `resolve/` URLs need HTML/asset rewriting. The source of truth here is
now a generated static Space tree.

## Files

- `manifest.json` — selected decks and candidate decks.
- `scripts/build.sh` — build `publish/site/` from the manifest.
- `scripts/mirror-space.sh` — download the current live Space into `publish/current-space/`.
- `scripts/verify.sh` — check that published entries exist in `publish/site/`.
- `scripts/deploy-space.sh` — upload `publish/site/` to the Space.
- `site/` — generated output; do not edit by hand.

## First-time / after remote-only content changes

`AI DevCon London` currently exists in the live Space, not as an obvious local
source directory. Mirror the Space before building:

```bash
publish/scripts/mirror-space.sh
```

## Build

```bash
publish/scripts/build.sh
publish/scripts/verify.sh
```

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
- whether each local source path was dirty;
- for mirrored live-Space sources, the Space SHA and last-modified timestamp.

This keeps published content tied back to either a local commit or the live Space
revision it was mirrored from.

## Deploy

```bash
publish/scripts/deploy-space.sh
```

Override the target Space or commit message:

```bash
SPACE=evalstate/presentations COMMIT_MESSAGE="Add NYC deck" publish/scripts/deploy-space.sh
```

`deploy-space.sh` uses `--delete '*'`, so the Space repo becomes exactly the
contents of `publish/site/`.

## Adding a deck

1. Build the deck in its source directory.
2. Add or update an entry in `manifest.json`.
3. Set `status` to `published` when it should appear on the public index.
4. Run `publish/scripts/build.sh && publish/scripts/verify.sh`.
5. Run `publish/scripts/deploy-space.sh` when ready.
