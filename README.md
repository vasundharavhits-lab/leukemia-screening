# Leukemia Screening Dashboard (research prototype)

**Not a diagnosis tool.** Every result must be confirmed by a qualified hematologist.

A Streamlit app that gives a probability of acute lymphoblastic leukemia (ALL) for a blood smear image, with a Grad-CAM view, a login, and a scan log.

## What it does

- Sign in, enter a patient ID (no names), upload a smear image, and get a result.
- A MobileNetV2 model classifies four classes (Benign, Early, Pre, Pro). P(leukemia) = 1 - P(Benign), decision threshold 0.50.
- An input check rejects images that do not look like the training images (photos, black, white, gray, noise) with "Unable to analyze". Rejected images are not saved.
- Single-cell analysis is switched off: its model has no documented training or validation.

## Known limits

- Trained and tested on one public dataset (Kaggle: mehradaria/leukemia). The test split was random by image, not by patient, so real accuracy is likely lower.
- Test accuracy is about 99% (3 to 4 errors out of 489 images). The errors are mostly Early cells called Benign, which is a missed-leukemia error.
- Confidence is not calibrated: the model can be wrong at 100%.
- Accuracy drops to about 83 to 85% when a third of the image is hidden, so images must be framed like the dataset's.
- Not validated on images from other labs, stains, or microscopes.
- Not cleared for clinical use.

## Setup

1. Python 3.11, then `pip install -r requirements.txt`.
2. Create a user: see the usage at the top of `manage_users.py`.
3. Run `streamlit run app.py`.

## Checks

- `python check_model.py` re-checks the model on the 489 test images (needs the test images in `Benign/`, `Early/`, `Pre/`, `Pro/`, which are not in this repository).
- `validate_model.py` is for validating on new patient-labeled data. Switch analysis on for a new image type only after it passes.

## Data

No images or patient data are stored in this repository. Check the dataset license before sharing a trained model.