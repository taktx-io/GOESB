# common-voice-nl-elderly

Pilot pack for ADR-0010 (gated-pack credential handling) — Dutch, filtered to
Common Voice's own self-reported `age` field, oldest available bucket(s).
This is the **one** pack authorized to prove out the ADR-0010 mechanism
(runner + schema); no further Common-Voice packs (other languages, other
age slices) should be built until Eric has run this one end-to-end and
signed off.

## Status: authored

Built via:

```bash
python scripts/build_common_voice_nl_elderly_pack.py \
  --dataset-id cmqinokkq00wwnr07hv5oax8l \
  --pack-dir packs/common-voice-nl-elderly
```

against **Common Voice Scripted Speech 26.0 - Dutch**
(`cmqinokkq00wwnr07hv5oax8l` on Mozilla Data Collective; validated split,
104,023 clips total for `nl`).

- **Age buckets included:** `seventies` + `eighties` + `nineties`. The
  oldest bucket alone (`nineties`: 5 validated clips) and even
  `eighties`+`nineties` combined (20 clips, 2 speakers) were too small to
  be a meaningful eval set on their own — see the full validated-split
  distribution below. `seventies` had to be folded in to reach a usable
  size.
- **Clip count:** 40 (all 15 `eighties` clips, all 5 `nineties` clips, 20
  of 83 `seventies` clips).
- **Speaker count:** 6 distinct speakers (Common Voice's own anonymized
  `client_id`, used only for counting — never persisted into
  `manifest.jsonl`).
- **Duration:** 270.481s total (40 clips, ~6.8s average), 32kHz mono MP3.
- **Full validated-split age distribution for `nl`** (for context — most
  of the corpus is younger, and over a third has no self-declared age at
  all):

  | bucket | clips |
  |---|---|
  | teens | 2,037 |
  | twenties | 19,657 |
  | thirties | 13,157 |
  | fourties | 19,500 |
  | fifties | 8,288 |
  | sixties | 1,905 |
  | seventies | 83 |
  | eighties | 15 |
  | nineties | 5 |
  | unlabeled | 39,376 |

**Caveat worth flagging to Eric before this is treated as a solid eval
set:** 6 speakers is thin. `seventies` alone has only 6 distinct speakers
in the entire validated split, so this pack's WER is really "how this
engine does on 6 specific older Dutch voices," not a statistically robust
read on elderly-speaker performance. Fine for proving the ADR-0010
mechanism works end-to-end; treat any WER number from it as anecdotal, not
representative, until/unless a larger elderly-speaker corpus is available.

## Speech style — read, not spontaneous

Like FLEURS, Common Voice is **read-aloud** speech (a contributor reads a
prompted sentence), not spontaneous conversation. Per the earlier finding
in the Cowork thread this ADR came from, read speech is an easier register
than real spontaneous speech and understates real-world WER — don't treat
this pack's numbers as representative of harder, spontaneous-speech
benchmarks (e.g. a future JASMIN-style pack). `metadata.speech_style: read`
reflects this; this note is here so it isn't missed when only the schema
field is checked.

## Credential

`audio.source.credential.env_var` is `MDC_API_KEY`. The wizard prompts for
it once (masked), stores it in `~/.goesb/credentials.json` (mode 0600), and
never sends it to goesb's own servers — see ADR-0010.

## A note on `profile_id`

This pack reuses `profile_id: whisper-medium-nl-batch`, the same profile
`fleurs-nl` already targets — no collision to work around anymore.
When a profile matches more than one pack, `_choose_packs_for_profile()`
now prompts (checkbox, `fleurs-nl` pre-checked as the old default)
instead of silently picking the first alphabetically. Enter with no
changes reproduces today's FLEURS-only behavior exactly; checking this
pack too runs both in the same batch. Explicit
`goesb run whisper-medium-nl-batch common-voice-nl-elderly`
invocations are unaffected either way — no prompt outside the wizard.
