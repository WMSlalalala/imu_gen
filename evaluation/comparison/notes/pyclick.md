# pyclick as a baseline on HMOG

**Paper.** pyclick, a Bezier human-like cursor path library (MIT)

**Code.** https://github.com/patrikoss/pyclick

## What ran unchanged

- `HumanCurve` and its distortion/tweening, unmodified

## Every deviation

- `pyautogui` is imported by the library at module scope but never used on this path; an empty stub module satisfies the import without touching the library.

## How it was scored

The generator supplies the touch coordinates. Everything else -- the carrier's update timing, its no-contact sentinel, its clock column, and the genuine events -- is the release's and is left byte-identical, which `verify_harness.py` checks.

Reported on: trajectory. A modality this method does not supply is not reported, because those cells would be running the release's own data.

**Declined actions.** keystroke/keystroke. A method that does not model an action declines it rather than having something its authors never proposed invented for it; declined actions carry no number.

## Result

| Modality | Cells | Mean FAR | Cells >= 0.60 |
|---|---|---|---|
| trajectory | 24 | 0.294 | 6 |

FAR at the development-selected FRR = 5% threshold, against a detector trained on this attack.
