# Coding rubric: what do repeated object labels refer to? (Amendment 5, S23)

Each row of `coding_sheet.csv` is one pair of nearby mentions of the same object label (for example *ball* ... *ball*), at most 100 words apart and in different utterances or sentences. You see the utterance before the pair (`before`), the stretch from the first to the second mention (`span`, utterances separated by `||`) and the utterance after (`after`). The sheet does not show which corpus a pair came from. Code from the words only.

## Column `referent` (one code)

| Code | Meaning | Example |
|---|---|---|
| `same` | Both mentions refer to the same physical object | *Where's the ball? ... Throw me the ball.* |
| `different` | The mentions refer to two or more distinct objects of the kind | *That's a big ball. ... Here's another ball.* / *two balls* |
| `generic` | At least one mention refers to the kind in general, not to a particular object | *Balls are for throwing.* / *Do you want a ball?* (no particular ball) |
| `unclear` | The words do not settle it | |

Use `different` whenever the words show more than one object is involved: *another*, *other*, numbers, plurals that clearly count objects, or a description that cannot fit one object (*the red one ... the blue one*). Use `same` when the definite article or a pronoun clearly tracks one object and nothing suggests a second one.

## Column `property` (all that apply, separated by `;`, or `none`)

Code any property of the object stated in `span`: `shape` (round, square, pointy, long), `colour`, `size` (big, little), `material` (soft, hard, wooden, plastic, fuzzy), `other` (wet, broken, new). Count only properties of the labelled object.

## Notes

Use `notes` for anything that made a decision hard. Code all rows before looking at the key file (`coding_sheet_key.csv`).

## Reliability

A second coder codes a random 25% of rows (`pair_id` chosen with seed 2031) without seeing the first coder's codes. Agreement is reported as Cohen's kappa for `referent` and percentage agreement for `property`.
