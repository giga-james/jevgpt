# Superseded sequential-draft experiment

The earlier four-consecutive-token draft-and-verify implementation was based on a misunderstanding and has been removed. The intended algorithm drafts four competing candidates for the **same** next-token position. See [parallel stratified samples](parallel.md).

`uv run jevgpt` now uses the corrected algorithm by default. The old `--selection speculative` option remains as an alias for `--selection parallel`; it does not perform sequential drafting.
