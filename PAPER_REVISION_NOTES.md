# Paper Revision Notes

Use these notes to revise the paper before submission.

## Main Claim

Recommended wording:

> The proposed model-guided expansion strategy improved macro recall and macro F1 in the current experiment, suggesting that targeted expansion can improve class-balanced performance under dataset imbalance.

Avoid:

> The proposed method improves all metrics.

The cached result shows macro recall and macro F1 improvement, while accuracy is slightly lower after retraining.

## Result Paragraph

Suggested replacement paragraph:

> The baseline model achieved 87.10% accuracy and 0.8724 macro F1 on the held-out test set. Continued training improved macro F1 to 0.8773, focal loss achieved 0.8742, real-image oversampling achieved the best macro F1 of 0.8806, and GAN-based augmentation achieved 0.8788. Although GAN augmentation did not outperform real oversampling in this run, it improved over the baseline and produced the highest macro recall among the tested methods. This suggests that targeted generative expansion can improve class-balanced behavior, but the result should be interpreted as competitive rather than decisively superior.

## Result Table

| Method | Accuracy | Macro Recall | Macro F1 |
| --- | ---: | ---: | ---: |
| Baseline | 0.8710 | 0.8910 | 0.8724 |
| Continue only | 0.8758 | 0.8950 | 0.8773 |
| Focal loss | 0.8720 | 0.8800 | 0.8742 |
| Real oversampling | 0.8772 | 0.8950 | 0.8806 |
| GAN augmented | 0.8767 | 0.8970 | 0.8788 |

## Limitation Paragraph

Suggested text:

> This study has several limitations. First, the current result is based on a single experimental run, so repeated trials across multiple random seeds are required before making a strong statistical claim. Second, real-image oversampling achieved the best macro F1 in this run, while GAN augmentation achieved the best macro recall. Therefore, the proposed method should be described as competitive and useful for improving class-balanced behavior, not as universally superior. Third, no external dataset validation is included, so clinical generalization remains untested.

## Extra Baseline To Add

The notebook now includes focal-loss support. After running it, add a row like this:

| Method | Accuracy | Macro Recall | Macro F1 |
| --- | ---: | ---: | ---: |
| Baseline | | | |
| Focal loss | | | |
| Real oversampling | | | |
| Model-guided expansion | | | |

## Submission Advice

For a submission in the next few days, target a student conference, department symposium, IEEE student branch event, or workshop. For a journal, first run multiple seeds and add an external validation dataset.
