# Model identity probe -- report

Synthetic or self-collected data only; verdicts need two independent signals (see README).

| Endpoint | Claimed | Verdict | Tier | Cutoff (last known) | Burn |
|---|---|---|---|---|---|
| endpoint-alpha | claimed-deepseek-flash | **MATCH** | 5/6 | 2025-02 | NORMAL |
| endpoint-beta | claimed-deepseek-flash | **MISMATCH** | 6/6 | 2025-06 | INFLATED |

## endpoint-alpha -- claimed-deepseek-flash
- [supports / neutral] identity_family: self-identifies as deepseek
- [supports / neutral] tokenizer_family: tokenizer signature is closest to deepseek (distance 3)

## endpoint-beta -- claimed-deepseek-flash
- [against claim] identity_family: self-identifies as openai, label claims deepseek
- [against claim] tokenizer_family: tokenizer signature is closest to openai (distance 2), label claims deepseek
- [supports / neutral] token_burn: hidden-token burn is INFLATED (1240 prompt tokens for a minimal greeting) -- a gateway-padding warning, not an identity signal
