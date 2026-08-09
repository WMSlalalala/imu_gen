# ghost-cursor as a baseline on HMOG

**Paper.** ghost-cursor, a Fitts-law cursor path library

**Code.** https://github.com/Xetera/ghost-cursor

## What ran unchanged

- the Bezier path construction, the overshoot rule and its corrective second segment, ported faithfully from `spoof.ts`

## Every deviation

- Two documented port corrections: arc-length sampling in place of parameter-space sampling, and an honoured `spreadOverride`.
- A degenerate-chord guard returning a constant point, recorded as declined-by-degeneracy rather than crashing.

## How it was scored

The generator supplies the touch coordinates. Everything else -- the carrier's update timing, its no-contact sentinel, its clock column, and the genuine events -- is the release's and is left byte-identical, which `verify_harness.py` checks.

Reported on: trajectory. A modality this method does not supply is not reported, because those cells would be running the release's own data.

**Declined actions.** keystroke/keystroke. A method that does not model an action declines it rather than having something its authors never proposed invented for it; declined actions carry no number.

## Result

| Modality | Cells | Mean FAR | Cells >= 0.60 |
|---|---|---|---|
| trajectory | 24 | 0.335 | 6 |

FAR at the development-selected FRR = 5% threshold, against a detector trained on this attack.
