# Publishing Roadmap

The project is close to a student-workshop or college-symposium submission, but it needs stronger experimental evidence before a serious journal submission.

## Best 30-Minute Upgrade

In a short window, the highest-value improvements are:

1. Make the repository understandable and reproducible.
2. Save results in machine-readable files.
3. Add a stronger imbalance baseline such as focal loss.
4. Prepare the paper to emphasize macro F1 and macro recall rather than only accuracy.
5. Be honest about limitations.

## What To Add Before Submission

### Required

- Run at least three random seeds.
- Report mean and standard deviation.
- Add focal loss or weighted sampler as a baseline.
- Include real oversampling as a control.
- Keep test data untouched until final evaluation.

### Recommended

- Add external validation if time allows.
- Add synthetic image quality analysis.
- Add confidence intervals or a simple statistical test.
- Include a clear limitations section.

## Paper Positioning

Use this as the central contribution:

> A simple model-guided expansion pipeline uses validation feedback to decide which minority class should receive targeted data augmentation.

Avoid claiming clinical readiness. A better claim is:

> The method improves class-balanced metrics in the tested setting and motivates further validation with repeated runs and external data.

## Suitable Venues

Start with:

- university research symposium
- IEEE student branch conference
- student paper competition
- AI/ML workshop
- medical imaging workshop

Avoid paid guaranteed-publication venues.

## Immediate Paper Edits

- Add a reproducibility paragraph with seed, hardware, dataset path, and package versions.
- Add a table for macro recall and macro F1.
- Add a limitation that accuracy dropped slightly in the cached retrained run while macro metrics improved.
- Add a future-work line for multi-seed and external validation.

## Submission Readiness

Current status: good project, not yet journal-ready.

Best next milestone: student conference or workshop after multi-seed experiments and one stronger baseline.
