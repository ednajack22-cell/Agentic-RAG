# Validation protocol and evaluation rubric
To assess response reliability, the study employed an LLM-as-a-Judge protocol using the RAGAS framework to operationalize faithfulness assessment. Continuous RAGAS outputs (0.0–1.0) were binarized using a predefined threshold of 0.55 to support the paired binary inferential design reported in the main text and to ensure that only sufficiently grounded responses were classified as successful.
Validation rubric for faithfulness assessment:
Claim extraction: Systematically isolate each factual claim contained in the generated response.
Context cross-reference: Search the retrieved context for explicit evidentiary support for each isolated claim.
Score calculation: The final continuous score represents the ratio of explicitly supported claims to the total number of factual claims in the response.
Examples:
Pass (✓): Four of five claims are traced directly to the retrieved context (score = 0.80).
Fail (✗): The model generates a plausible answer, but only two of six claims are supported by the retrieved evidence (score = 0.33).
To validate the automated instrument, a blind human-in-the-loop inter-rater reliability audit was conducted on a stratified random subsample of 150 queries. The automated validator and human experts demonstrated strong agreement (Cohen’s κ = 0.940; 95% CI: [0.839, 1.000], via 10,000 bootstrap resamples), supporting the validity of the evaluation protocol. The CI lower bound of 0.839 falls within the almost perfect agreement range by Landis and Koch (1977) standards. Two false positives were observed (1.3%), both involving borderline multi-hop queries identified consistently across audit stages. Zero false negatives were recorded.