-- Do policy-period reported counts match severity record counts? No target repair.
WITH observed AS (SELECT IDpol, COUNT(*) AS severity_records FROM claims GROUP BY IDpol)
SELECT p.ClaimNb AS reported_claim_count, COALESCE(o.severity_records, 0) AS severity_records,
       COUNT(*) AS policies
FROM policies p LEFT JOIN observed o ON p.IDpol = o.IDpol
GROUP BY p.ClaimNb, COALESCE(o.severity_records, 0)
ORDER BY reported_claim_count, severity_records;
