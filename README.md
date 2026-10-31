# model-identity-probe

A black-box battery for a simple question: **when a third-party gateway
sells you a model under a famous label, is the thing behind the label
actually that model family?** The toolkit defines the probes, scores
recorded answers offline, and refuses to call a substitution on a
single signal. It runs locally; the analysis code makes no network
requests and ships with no endpoints, no keys and no provider accounts.

## Why this exists

Reseller gateways re-label models. A cheaper (or entirely different)
model behind a premium label is invisible in normal use: the chat
works, the answers look plausible, and the only thing that was stolen
is the difference. Self-identification alone proves nothing either --
a system prompt can teach any model any name. What works is a battery
of *independent* fingerprints that are hard to fake all at once.

## The method

1. **Identity, three ways.** Direct question, a multi-turn repeat (does
   the answer stay consistent?), and a pressure probe whose system
   prompt pre-loads a false identity (does the model adopt it?).
2. **Token burn.** Ask "Hi" and read `prompt_tokens`. A minimal
   greeting costs a handful of tokens; hundreds or thousands mean the
   gateway injects hidden context -- or its usage numbers are fiction.
3. **Tokenizer fingerprint.** Three fixed prompts (English pangram,
   CJK text, code). The reported `prompt_tokens` triple differs
   measurably between model families and is very hard to fake without
   actually using that family's tokenizer.
4. **Capability ladder L1-L6.** Arithmetic, primes, *executable* code
   (fib, bisect -- checked by running the answer, not by string
   match), logic traps, and hard number theory. Tier = highest level
   with every level below it passed.
5. **Cutoff ladder.** Dated public events answered YES/NO/UNKNOWN
   estimate the knowledge cutoff; a YES after a NO is flagged as
   inconsistent rather than smoothed away.
6. **Style blocks.** Refusal patterns, glitch tokens, chat-template
   injection, formatting discipline, classic tokenization trivia
   ("strawberry", 9.11 vs 9.9) and a whorl test: five random passwords
   at temperature 1, whose character distribution is a stable family
   fingerprint.

Full probe definitions live in [`src/battery.py`](src/battery.py);
the scoring rules live in [`src/probe.py`](src/probe.py).

## The verdict rule

| Verdict | Meaning |
|---|---|
| `MATCH` | No independent signal contradicts the claimed family. |
| `MISMATCH` | **At least two independent signals** (e.g. identity *and* tokenizer) contradict the claimed family. |
| `INCONCLUSIVE` | Exactly one signal contradicts the claim, or the record is too thin. One signal is a question, not a proof. |

Token burn is deliberately *not* an identity signal: an inflated burn
indicts the gateway's padding or accounting, not the model's family.

## Quick start

Requires Python 3.11 or newer. No dependencies.

```console
python src/run_demo.py
python src/run_demo.py --out my_report.md
```

The demo scores a fully synthetic two-endpoint file
(`examples/demo_results.json`): `endpoint-alpha` is consistent with
its label (`MATCH`); `endpoint-beta` claims one family while its
identity answers and tokenizer signature both point to another --
`MISMATCH`, plus an inflated burn warning. Generated output:
[`examples/demo_report.md`](examples/demo_report.md).

To probe a real endpoint: send the prompts from `src/battery.py`
through any client you trust, save the answers and usage numbers in
the record schema documented at the top of `src/probe.py`, add
reference tokenizer signatures you measured on known-genuine models,
and score the file with the same two functions the demo uses
(`load_results` + `render_report`). Nothing in this repository needs
to know where the answers came from.

## Failure log (honest, from real use of this method)

- An early battery scored reasoning models with a tiny `max_tokens`.
  They burned the whole budget on hidden reasoning, returned empty
  content, and were logged as "silent / lying". They were neither --
  the probe was broken. Truncation is now a retry signal, never a
  failure, and identity probes use a generous budget with one retry.
- A popular classification engine, run locally without its server-side
  baselines and judge, returned family verdicts at confidence
  0.3-0.55 -- barely better than a guess dressed as a score. Hence
  the two-signal rule instead of trusting any single classifier.
- Whole "line-ups" of differently named models turned out to be one
  profile under many labels: identical tokenizer triples, identical
  burn, identical cutoff, identical tier. Exact versions inside such
  a line-up could not be separated by any probe in this battery --
  only the family could. This toolkit claims family-level verdicts
  for that reason, and says so.
- One gateway rewrote canary strings in responses, which made an
  augmentation detector fire on *every* model behind it. A signal
  that fires on everything indicts the transport, not the models;
  such signals belong in the gateway column of the report.
- Streaming latency (time to first token, tokens per second) varies
  wildly with gateway load -- the same model measured 4 s in the
  morning and 11 s in the evening. Latency profiles routes, not
  identity, and is excluded from verdicts here.

## Limitations

- Verdicts are family-level. Distinguishing two close versions of the
  same family is usually impossible from black-box probes alone.
- A determined gateway can proxy probes to the genuine model and
  traffic to a substitute. No client-side battery fully defeats
  selective proxying; treat a `MATCH` as "no evidence of substitution
  in this sample", not as a guarantee.
- Reference tokenizer signatures must be measured on models you
  already trust. Comparing against signatures copied from the same
  gateway you are testing proves nothing.
- Refusal and style blocks drift as providers retune their models;
  they are supporting evidence, never verdict inputs here.

## Tests

The tests use `unittest` and are also collectable by pytest:

```console
python -m unittest discover -s tests -v
python -m pytest
```

Built with Codex, October-November 2026
