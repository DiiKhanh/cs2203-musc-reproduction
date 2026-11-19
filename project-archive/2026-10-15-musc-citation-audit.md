# MuSc Final Report Citation Audit

## Scope

- Draft: `2026-10-15-musc-final-report-draft-vi.md`
- Citation style: IEEE numbered references
- Audit date: 2026-10-15
- Verification level: metadata and claim-level checks against primary paper,
  proceedings, publisher, and official repository pages

## Summary

| Metric | Result |
|---|---:|
| In-text citation occurrences | 17 |
| Unique in-text citations | 8 |
| Reference-list entries | 8 |
| Orphan in-text citations | 0 |
| Orphan reference-list entries | 0 |
| First-appearance order violations | 0 |
| Self-citations | 0 |
| Sources published from 2022 onward | 6 of 8 |
| Journal articles requiring retraction screening | 0 |

## Corrections Made

| Location | Issue | Correction |
|---|---|---|
| Related work | Missing comparison with representative normal-only and zero-shot methods | Added PatchCore, WinCLIP, and AnomalyCLIP with protocol differences |
| Reference [1] | MuSc title omitted "the" before "Unlabeled Images" | Restored the official paper title |
| References [2]-[8] | New sources initially broke IEEE order of first appearance | Renumbered citations and reordered the reference list |
| Reference [6] | MVTec AD DOI missing | Added `10.1109/CVPR.2019.00982` |
| Reference [7] | VisA paper DOI and page range missing | Added pages 392-408 and `10.1007/978-3-031-20056-4_23` |
| References [2] and [3] | DOI missing | Added the verified PatchCore and WinCLIP DOI URLs |
| Reference [8] | Repository year was not independently supported in the draft | Confirmed the pinned commit author date as 2026-04-25 from the Git object |

## Source Verification

| Ref. | Verification source | Status |
|---:|---|---|
| [1] | ICLR Proceedings and OpenReview record | Verified |
| [2] | CVPR Open Access and IEEE DOI record | Verified |
| [3] | CVPR Open Access and IEEE DOI record | Verified |
| [4] | ICLR OpenReview paper record | Verified |
| [5] | PMLR proceedings record | Verified |
| [6] | CVPR Open Access and IEEE DOI record | Verified |
| [7] | ECVA proceedings PDF and Springer DOI record | Verified |
| [8] | Official MuSc GitHub repository and pinned Git commit | Verified |

## Claim Checks

- PatchCore is described as using a representative memory bank of nominal
  patch features and nearest-neighbor scoring, matching the CVPR paper.
- WinCLIP is described as using a compositional prompt ensemble and multi-scale
  window, patch, and image features without target-specific tuning in its
  zero-shot setting, matching the CVPR paper.
- AnomalyCLIP is described as learning object-agnostic prompts on auxiliary
  anomaly-detection data and transferring without target-dataset training,
  matching the ICLR paper.
- MuSc is described as prompt-free and training-free on the target data while
  using the unlabeled test set transductively, matching the ICLR paper and the
  reproduced implementation.

## Items Requiring Human Confirmation

1. A formal similarity report from Turnitin or iThenticate was not available.
   The draft contains no direct quotations or copied long passages identified
   during this audit, but this is not a plagiarism certificate.
2. The GitHub source repository is private. The instructor must be granted
   access, or the repository must be made public, before the final link check.

## Verdict

The reference list passes the internal citation-integrity checks: every numbered
citation resolves to one verified entry, every entry is cited, and numbering
follows first appearance. Final originality clearance and access control remain
author-owned submission checks.
