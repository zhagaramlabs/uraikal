# Contributing to Uraikal

Uraikal is Zhagaram Labs' benchmark for Tamil models and the Tamil text library it scores with.
Issues and pull requests are welcome, in Tamil or English.

Everyone here follows our [Code of Conduct](CODE_OF_CONDUCT.md). Report security problems privately, as in [SECURITY.md](SECURITY.md), not in a public issue.

## Ways to help

- **Score a model** on a track and send the result file, or report a model we should add.
- **Find a scoring mistake:** a Tamil spelling the normaliser treats wrongly, or a number, date or
  verse it reads wrongly. Give the text, what Uraikal did, and what it should do.
- **Library code:** letters, numbers, prosody (யாப்பு), and the roadmap items in the README.
- **Native speakers and scholars:** check the expected outputs in `tests/`.

## Before a pull request

1. Run the same checks as CI:
   ```bash
   for f in tests/test_*.py; do uv run --extra tokenizers python "$f" || break; done
   uv run python -m doctest src/uraikal/text.py src/uraikal/yaappu.py
   ```
2. Keep the library free of dependencies: `pip install uraikal` must pull in nothing else.
3. Keep the change small and in the style of the code around it. New logic gets a test or a doctest.
4. If a change alters scores, say so: scoring changes need a new `URAIKAL_VERSION` (see Releasing).

## Rules that protect the benchmark

- **Hidden test sets stay hidden.** Never commit, publish or train on test-set items. Only scores
  leave the lab.
- **Every dataset states its source, revision and licence** in the spec.
- **No model outputs from closed APIs on private test sets.**

By contributing you agree your work is released under the Apache 2.0 licence ([LICENSE](LICENSE)).
