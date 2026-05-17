## Evaluation metrics for bifurcation identification

To evaluate the performance of stARC on the bifurcation identification task, we use several standard classification metrics.

- A true positive (TP) denotes a bifurcating sample that is correctly identified as its true bifurcation type.

- A true negative (TN) denotes a non-bifurcating sample that is correctly identified as non-bifurcating.

- A false positive (FP) denotes a non-bifurcating sample that is incorrectly identified as a bifurcation. 
In addition, when a bifurcating sample is assigned to an incorrect bifurcation type, it is counted as a false positive sample.

- A false negative (FN) denotes a bifurcating sample that is incorrectly identified as non-bifurcating.

Based on these quantities, the true positive rate (TPR), is defined as the ratio of correctly classified bifurcating samples to the sum of TP samples and FN samples, i.e.,

$$\mathrm{TPR}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}.$$

The accuracy is defined as the proportion of correctly classified samples among all evaluated samples:

$$\mathrm{Accuracy}=\frac{\mathrm{TP}+\mathrm{TN}}{\mathrm{TP}+\mathrm{TN}+\mathrm{FP}+\mathrm{FN}}.$$

## Rules for bifurcation type classification

In this work, the bifurcation type is determined according to the local behavior of the estimated dominant eigenvalue branches after the early-warning index. 
Each sample is assigned to only one class, following the priority order: Neimark-Sacker, transcritical, period-doubling, and Null. 
Once a sample satisfies a criterion at an earlier step, the subsequent criteria are not evaluated for class assignment.

- A Neimark-Sacker (NS) bifurcation is identified first when two estimated eigenvalue branches approximately form a complex conjugate pair. 
This is characterized by similar real parts, imaginary parts with opposite signs, and a non-negligible imaginary component. 
The imaginary-part threshold is introduced to avoid classifying nearly real eigenvalues as NS.

- A transcritical (Trans) bifurcation is identified when the dominant real component approaches the positive unit-circle boundary.

- A period-doubling (PD) bifurcation is identified when the dominant real component approaches the negative unit-circle boundary.

- Samples that do not satisfy any of the above criteria are assigned to the Null class.


## Possible misclassification between transcritical and Neimark-Sacker bifurcation

According to the rules for bifurcation type classification, the bifurcation type is assigned using a mutually exclusive and sequential decision rule and the NS criterion is evaluated before the Trans criterion. 
As a result, a sample satisfying the empirical NS condition is assigned to the NS class directly, without being further tested against the Trans criterion.

This may occasionally cause a true Trans sample to be classified as NS. 
Although a transcritical bifurcation is theoretically associated with a dominant real eigenvalue approaching the positive unit-circle boundary, 
data-driven eigenvalue estimates may exhibit small nonzero imaginary components due to numerical fluctuations, noise contamination, or the inherent randomness in the construction of reservoir networks. 
In such cases, the sample may satisfy the NS criterion before the Trans criterion is considered.

Therefore, occasional Trans-to-NS misclassification may occur under the adopted classification protocol, especially for samples near the empirical decision boundaries. 
This behavior is associated with the sensitivity of threshold-based eigenvalue classification to small perturbations in the estimated eigenvalue branches near critical transition points.
