-- Where are observed claim records and costs concentrated? Descriptive, not causal.
SELECT p.Region, COUNT(*) AS claim_records, COUNT(DISTINCT c.IDpol) AS policies,
       AVG(c.ClaimAmount) AS mean_recorded_amount, SUM(c.ClaimAmount) AS total_recorded_amount
FROM claims c JOIN policies p ON c.IDpol = p.IDpol
GROUP BY p.Region ORDER BY claim_records DESC;
