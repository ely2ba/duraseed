# Supplementary frozen summaries

All rates are raw. R-P minus R-S throughout. No fitted decay, smoothing, new windows or resampling settings.

## Retention and starting-score sensitivity

| Arm | Role | Baseline | Raw AUC 0–20 | AUC / own baseline | First half-life | Bracket/status |
| --- | --- | --- | --- | --- | --- | --- |
| R-S | targeted | 0.30729167 | 0.11494141 | 0.37404661 | 4.25000000 | [2, 5] |
| R-S | sentinel | 0.04166667 | 0.03453776 | 0.82890625 | 28.57142857 | [20, 40] |
| R-P | targeted | 0.36979167 | 0.17620443 | 0.47649648 | 1.68918919 | [1, 2] |
| R-P | sentinel | 0.35546875 | 0.18860677 | 0.53058608 | 1.52717391 | [1, 2] |

Normalization is a descriptive starting-score sensitivity; it does not create equal initial functions or eliminate selection conditioning.

## First attainment at fixed absolute new-task thresholds

| Threshold | Role | Arm | Status | Interpolated update | Observed bracket | TCES success | Eligible R-P − R-S |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.06 | targeted | R-S | crossed | 4.33748837 | [2, 5] | 0.15015237 | — |
| 0.06 | targeted | R-P | crossed | 0.66520000 | [0, 1] | 0.29097240 | 0.14082002 |
| 0.06 | sentinel | R-S | crossed | 4.33748837 | [2, 5] | 0.03573135 | — |
| 0.06 | sentinel | R-P | crossed | 0.66520000 | [0, 1] | 0.27924792 | 0.24351657 |
| 0.07 | targeted | R-S | crossed | 9.78333333 | [5, 10] | 0.06888889 | — |
| 0.07 | targeted | R-P | crossed | 12.74736842 | [10, 20] | 0.23402823 | 0.16513935 |
| 0.07 | sentinel | R-S | crossed | 9.78333333 | [5, 10] | 0.03266493 | — |
| 0.07 | sentinel | R-P | crossed | 12.74736842 | [10, 20] | 0.26894052 | 0.23627558 |
| 0.1 | targeted | R-S | crossed | 14.76023392 | [10, 20] | 0.06888554 | — |
| 0.1 | targeted | R-P | crossed | 50.16932515 | [40, 80] | 0.01174080 | -0.05714474 |
| 0.1 | sentinel | R-S | crossed | 14.76023392 | [10, 20] | 0.03441155 | — |
| 0.1 | sentinel | R-P | crossed | 50.16932515 | [40, 80] | 0.00748147 | -0.02693008 |
| 0.2 | targeted | R-S | crossed | 95.02362205 | [80, 160] | 0.00228018 | — |
| 0.2 | targeted | R-P | crossed | 107.10256410 | [80, 160] | 0.01121378 | 0.00893359 |
| 0.2 | sentinel | R-S | crossed | 95.02362205 | [80, 160] | 0.00024453 | — |
| 0.2 | sentinel | R-P | crossed | 107.10256410 | [80, 160] | 0.00474593 | 0.00450140 |

Crossings follow training order, including reversals; both coordinates interpolate along the same observed interval. Baseline-exceeded and not-reached rows are not treated as matched post-training progress.

## Authoritative failure and length decomposition

Codes are mutually exclusive; cap/tag/syntax flags overlap and must not be summed. Conditional accuracy is not a repaired or counterfactual score.

| Arm | Role | Update | N | Raw success | Valid successes/N | Conditional | Cap | Bad tag | Bad syntax | Mean tokens | Median tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R-S | sentinel | 0 | 768 | 0.04166667 | 32/284 | 0.11267606 | 79 | 91 | 484 | 572.34375000 | 111.00000000 |
| R-S | targeted | 0 | 768 | 0.30729167 | 236/426 | 0.55399061 | 76 | 85 | 342 | 537.52864583 | 87.00000000 |
| R-S | sentinel | 1 | 768 | 0.02734375 | 21/342 | 0.06140351 | 50 | 50 | 426 | 419.51822917 | 101.50000000 |
| R-S | targeted | 1 | 768 | 0.30729167 | 236/441 | 0.53514739 | 47 | 48 | 327 | 383.03255208 | 82.00000000 |
| R-S | sentinel | 2 | 768 | 0.03776042 | 29/437 | 0.06636156 | 36 | 36 | 331 | 311.66145833 | 91.00000000 |
| R-S | targeted | 2 | 768 | 0.24348958 | 187/500 | 0.37400000 | 27 | 28 | 268 | 261.48828125 | 81.00000000 |
| R-S | sentinel | 5 | 768 | 0.03515625 | 27/466 | 0.05793991 | 19 | 22 | 302 | 221.15755208 | 83.00000000 |
| R-S | targeted | 5 | 768 | 0.12369792 | 95/496 | 0.19153226 | 36 | 40 | 272 | 283.77864583 | 78.00000000 |
| R-S | sentinel | 10 | 768 | 0.03255208 | 25/443 | 0.05643341 | 50 | 55 | 325 | 375.95182292 | 83.00000000 |
| R-S | targeted | 10 | 768 | 0.06640625 | 51/464 | 0.10991379 | 49 | 53 | 304 | 360.22656250 | 82.00000000 |
| R-S | sentinel | 20 | 768 | 0.03645833 | 28/481 | 0.05821206 | 53 | 56 | 287 | 375.63151042 | 84.00000000 |
| R-S | targeted | 20 | 768 | 0.07161458 | 55/494 | 0.11133603 | 58 | 60 | 274 | 409.36848958 | 82.00000000 |
| R-S | sentinel | 40 | 768 | 0.00000000 | 0/494 | 0.00000000 | 17 | 24 | 274 | 139.79687500 | 20.00000000 |
| R-S | targeted | 40 | 768 | 0.00911458 | 7/531 | 0.01318267 | 15 | 16 | 237 | 113.51953125 | 19.00000000 |
| R-S | stage-b | 0 | 8192 | 0.00000000 | 0/0 | undefined / unavailable | 1743 | 8192 | 8192 | 66.39636230 | 58.00000000 |
| R-S | stage-b | 1 | 8192 | 0.01391602 | 114/3374 | 0.03378779 | 250 | 365 | 4818 | 27.91723633 | 21.00000000 |
| R-S | stage-b | 2 | 8192 | 0.03955078 | 324/7694 | 0.04211074 | 2 | 16 | 498 | 11.79675293 | 13.00000000 |
| R-S | stage-b | 5 | 8192 | 0.06579590 | 539/8171 | 0.06596500 | 0 | 0 | 21 | 12.82812500 | 13.00000000 |
| R-S | stage-b | 10 | 8192 | 0.07019043 | 575/8186 | 0.07024188 | 0 | 0 | 6 | 12.77465820 | 13.00000000 |
| R-S | stage-b | 20 | 8192 | 0.13281250 | 1088/8192 | 0.13281250 | 0 | 0 | 0 | 12.69262695 | 13.00000000 |
| R-S | stage-b | 40 | 8192 | 0.15344238 | 1257/8192 | 0.15344238 | 0 | 0 | 0 | 12.68200684 | 13.00000000 |
| R-P | sentinel | 0 | 768 | 0.35546875 | 273/481 | 0.56756757 | 155 | 207 | 287 | 1361.59244792 | 633.50000000 |
| R-P | targeted | 0 | 768 | 0.36979167 | 284/508 | 0.55905512 | 147 | 195 | 260 | 1349.53255208 | 553.00000000 |
| R-P | sentinel | 1 | 768 | 0.24088542 | 185/537 | 0.34450652 | 95 | 122 | 231 | 898.55859375 | 79.00000000 |
| R-P | targeted | 1 | 768 | 0.25130208 | 193/507 | 0.38067061 | 121 | 150 | 261 | 985.25130208 | 127.50000000 |
| R-P | sentinel | 2 | 768 | 0.12109375 | 93/594 | 0.15656566 | 62 | 72 | 174 | 488.64062500 | 23.00000000 |
| R-P | targeted | 2 | 768 | 0.15494792 | 119/569 | 0.20913884 | 66 | 82 | 199 | 543.07161458 | 23.00000000 |
| R-P | sentinel | 5 | 768 | 0.07552083 | 58/599 | 0.09682805 | 26 | 41 | 169 | 258.14322917 | 23.00000000 |
| R-P | targeted | 5 | 768 | 0.07682292 | 59/586 | 0.10068259 | 31 | 47 | 182 | 277.54036458 | 23.00000000 |
| R-P | sentinel | 10 | 768 | 0.36588542 | 281/329 | 0.85410334 | 118 | 123 | 439 | 1344.81250000 | 738.50000000 |
| R-P | targeted | 10 | 768 | 0.31380208 | 241/288 | 0.83680556 | 144 | 151 | 480 | 1339.52213542 | 570.50000000 |
| R-P | sentinel | 20 | 768 | 0.01302083 | 10/722 | 0.01385042 | 0 | 0 | 46 | 21.73958333 | 22.00000000 |
| R-P | targeted | 20 | 768 | 0.02343750 | 18/717 | 0.02510460 | 0 | 0 | 51 | 21.98828125 | 22.00000000 |
| R-P | sentinel | 40 | 768 | 0.00781250 | 6/725 | 0.00827586 | 1 | 1 | 43 | 27.25781250 | 22.00000000 |
| R-P | targeted | 40 | 768 | 0.01041667 | 8/715 | 0.01118881 | 0 | 0 | 53 | 22.05078125 | 22.50000000 |
| R-P | stage-b | 0 | 8192 | 0.05187988 | 425/7145 | 0.05948216 | 553 | 690 | 1047 | 23.62304688 | 14.00000000 |
| R-P | stage-b | 1 | 8192 | 0.06408691 | 525/8117 | 0.06467907 | 3 | 73 | 75 | 13.56860352 | 13.00000000 |
| R-P | stage-b | 2 | 8192 | 0.06127930 | 502/8173 | 0.06142175 | 0 | 0 | 19 | 12.57263184 | 13.00000000 |
| R-P | stage-b | 5 | 8192 | 0.06286621 | 515/8192 | 0.06286621 | 0 | 0 | 0 | 12.78149414 | 13.00000000 |
| R-P | stage-b | 10 | 8192 | 0.06872559 | 563/8192 | 0.06872559 | 0 | 0 | 0 | 12.76440430 | 13.00000000 |
| R-P | stage-b | 20 | 8192 | 0.07336426 | 601/8192 | 0.07336426 | 0 | 0 | 0 | 12.73950195 | 13.00000000 |
| R-P | stage-b | 40 | 8192 | 0.07470703 | 612/8192 | 0.07470703 | 0 | 0 | 0 | 12.59545898 | 13.00000000 |

| Arm | Role | Update | Disjoint authoritative codes |
| --- | --- | --- | --- |
| R-S | sentinel | 0 | {"ast_limit_exceeded": 342, "correct": 32, "invalid_character": 27, "invalid_syntax": 24, "missing_answer_tag": 91, "operand_multiset_mismatch": 218, "wrong_target": 34} |
| R-S | targeted | 0 | {"ast_limit_exceeded": 224, "correct": 236, "invalid_character": 12, "invalid_syntax": 21, "missing_answer_tag": 85, "operand_multiset_mismatch": 158, "wrong_target": 32} |
| R-S | sentinel | 1 | {"ast_limit_exceeded": 317, "correct": 21, "invalid_character": 35, "invalid_syntax": 24, "missing_answer_tag": 50, "operand_multiset_mismatch": 266, "wrong_target": 55} |
| R-S | targeted | 1 | {"ast_limit_exceeded": 245, "correct": 236, "invalid_character": 8, "invalid_syntax": 26, "missing_answer_tag": 48, "operand_multiset_mismatch": 158, "wrong_target": 47} |
| R-S | sentinel | 2 | {"ast_limit_exceeded": 246, "correct": 29, "invalid_character": 21, "invalid_syntax": 28, "missing_answer_tag": 36, "operand_multiset_mismatch": 287, "wrong_target": 121} |
| R-S | targeted | 2 | {"ast_limit_exceeded": 208, "correct": 187, "invalid_character": 11, "invalid_syntax": 21, "missing_answer_tag": 28, "operand_multiset_mismatch": 215, "wrong_target": 98} |
| R-S | sentinel | 5 | {"ast_limit_exceeded": 159, "correct": 27, "invalid_character": 99, "invalid_syntax": 22, "missing_answer_tag": 20, "multiple_answer_tags": 2, "operand_multiset_mismatch": 309, "wrong_target": 130} |
| R-S | targeted | 5 | {"ast_limit_exceeded": 132, "correct": 95, "invalid_character": 74, "invalid_syntax": 26, "missing_answer_tag": 38, "multiple_answer_tags": 2, "operand_multiset_mismatch": 265, "wrong_target": 136} |
| R-S | sentinel | 10 | {"ast_limit_exceeded": 157, "correct": 25, "invalid_character": 100, "invalid_syntax": 13, "missing_answer_tag": 52, "multiple_answer_tags": 3, "operand_multiset_mismatch": 283, "wrong_target": 135} |
| R-S | targeted | 10 | {"answer_too_long": 1, "ast_limit_exceeded": 145, "correct": 51, "invalid_character": 88, "invalid_syntax": 17, "missing_answer_tag": 51, "multiple_answer_tags": 2, "operand_multiset_mismatch": 253, "wrong_target": 160} |
| R-S | sentinel | 20 | {"ast_limit_exceeded": 119, "correct": 28, "invalid_character": 94, "invalid_syntax": 18, "missing_answer_tag": 53, "multiple_answer_tags": 3, "operand_multiset_mismatch": 324, "wrong_target": 129} |
| R-S | targeted | 20 | {"ast_limit_exceeded": 116, "correct": 55, "division_by_zero": 2, "invalid_character": 86, "invalid_syntax": 12, "missing_answer_tag": 58, "multiple_answer_tags": 2, "operand_multiset_mismatch": 291, "wrong_target": 146} |
| R-S | sentinel | 40 | {"answer_too_long": 2, "ast_limit_exceeded": 16, "invalid_character": 229, "invalid_syntax": 3, "missing_answer_tag": 16, "multiple_answer_tags": 8, "operand_multiset_mismatch": 451, "wrong_target": 43} |
| R-S | targeted | 40 | {"ast_limit_exceeded": 9, "correct": 7, "invalid_character": 210, "invalid_syntax": 2, "missing_answer_tag": 13, "multiple_answer_tags": 3, "operand_multiset_mismatch": 490, "wrong_target": 34} |
| R-S | stage-b | 0 | {"invalid_program": 3110, "missing_answer_tag": 5064, "multiple_answer_tags": 18} |
| R-S | stage-b | 1 | {"correct": 114, "illegal_instruction": 3, "invalid_program": 593, "missing_answer_tag": 251, "multiple_answer_tags": 79, "program_too_long": 5638, "wrong_target": 1514} |
| R-S | stage-b | 2 | {"correct": 324, "illegal_instruction": 23, "invalid_program": 477, "missing_answer_tag": 2, "multiple_answer_tags": 14, "program_too_long": 11, "wrong_target": 7341} |
| R-S | stage-b | 5 | {"correct": 539, "illegal_instruction": 143, "invalid_program": 21, "wrong_target": 7489} |
| R-S | stage-b | 10 | {"correct": 575, "illegal_instruction": 2, "invalid_program": 6, "wrong_target": 7609} |
| R-S | stage-b | 20 | {"correct": 1088, "illegal_instruction": 1006, "wrong_target": 6098} |
| R-S | stage-b | 40 | {"correct": 1257, "illegal_instruction": 1119, "wrong_target": 5816} |
| R-P | sentinel | 0 | {"answer_too_long": 1, "ast_limit_exceeded": 26, "correct": 273, "invalid_character": 50, "invalid_syntax": 3, "missing_answer_tag": 202, "multiple_answer_tags": 5, "operand_multiset_mismatch": 43, "wrong_target": 165} |
| R-P | targeted | 0 | {"ast_limit_exceeded": 12, "correct": 284, "empty_answer": 2, "invalid_character": 46, "invalid_syntax": 5, "missing_answer_tag": 186, "multiple_answer_tags": 9, "operand_multiset_mismatch": 52, "wrong_target": 172} |
| R-P | sentinel | 1 | {"ast_limit_exceeded": 24, "correct": 185, "invalid_character": 83, "invalid_syntax": 2, "missing_answer_tag": 120, "multiple_answer_tags": 2, "operand_multiset_mismatch": 64, "wrong_target": 288} |
| R-P | targeted | 1 | {"ast_limit_exceeded": 20, "correct": 193, "empty_answer": 1, "intermediate_limit_exceeded": 1, "invalid_character": 87, "invalid_syntax": 3, "missing_answer_tag": 149, "multiple_answer_tags": 1, "operand_multiset_mismatch": 65, "wrong_target": 248} |
| R-P | sentinel | 2 | {"ast_limit_exceeded": 26, "correct": 93, "invalid_character": 75, "invalid_syntax": 1, "missing_answer_tag": 71, "multiple_answer_tags": 1, "operand_multiset_mismatch": 80, "wrong_target": 421} |
| R-P | targeted | 2 | {"ast_limit_exceeded": 27, "correct": 119, "empty_answer": 2, "invalid_character": 83, "invalid_syntax": 5, "missing_answer_tag": 80, "multiple_answer_tags": 2, "operand_multiset_mismatch": 91, "wrong_target": 359} |
| R-P | sentinel | 5 | {"ast_limit_exceeded": 29, "correct": 58, "empty_answer": 1, "invalid_character": 96, "invalid_syntax": 2, "missing_answer_tag": 40, "multiple_answer_tags": 1, "operand_multiset_mismatch": 85, "wrong_target": 456} |
| R-P | targeted | 5 | {"ast_limit_exceeded": 32, "correct": 59, "empty_answer": 1, "invalid_character": 100, "invalid_syntax": 2, "missing_answer_tag": 47, "operand_multiset_mismatch": 56, "wrong_target": 471} |
| R-P | sentinel | 10 | {"ast_limit_exceeded": 4, "correct": 281, "invalid_character": 311, "invalid_syntax": 1, "missing_answer_tag": 117, "multiple_answer_tags": 6, "operand_multiset_mismatch": 8, "wrong_target": 40} |
| R-P | targeted | 10 | {"ast_limit_exceeded": 6, "correct": 241, "invalid_character": 323, "missing_answer_tag": 144, "multiple_answer_tags": 7, "operand_multiset_mismatch": 6, "wrong_target": 41} |
| R-P | sentinel | 20 | {"ast_limit_exceeded": 12, "correct": 10, "invalid_syntax": 34, "operand_multiset_mismatch": 63, "wrong_target": 649} |
| R-P | targeted | 20 | {"ast_limit_exceeded": 11, "correct": 18, "invalid_syntax": 40, "operand_multiset_mismatch": 52, "wrong_target": 647} |
| R-P | sentinel | 40 | {"ast_limit_exceeded": 24, "correct": 6, "invalid_character": 2, "invalid_syntax": 16, "missing_answer_tag": 1, "operand_multiset_mismatch": 229, "wrong_target": 490} |
| R-P | targeted | 40 | {"ast_limit_exceeded": 35, "correct": 8, "invalid_character": 1, "invalid_syntax": 17, "operand_multiset_mismatch": 213, "wrong_target": 494} |
| R-P | stage-b | 0 | {"correct": 425, "illegal_instruction": 100, "invalid_program": 190, "missing_answer_tag": 559, "multiple_answer_tags": 4, "program_too_long": 700, "wrong_target": 6214} |
| R-P | stage-b | 1 | {"correct": 525, "illegal_instruction": 2, "invalid_program": 70, "missing_answer_tag": 3, "program_too_long": 5, "wrong_target": 7587} |
| R-P | stage-b | 2 | {"correct": 502, "illegal_instruction": 3, "invalid_program": 19, "wrong_target": 7668} |
| R-P | stage-b | 5 | {"correct": 515, "wrong_target": 7677} |
| R-P | stage-b | 10 | {"correct": 563, "wrong_target": 7629} |
| R-P | stage-b | 20 | {"correct": 601, "wrong_target": 7591} |
| R-P | stage-b | 40 | {"correct": 612, "wrong_target": 7580} |
