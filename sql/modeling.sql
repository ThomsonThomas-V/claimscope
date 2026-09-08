-- Preserve one source severity record per row; never sum claims into policy outcomes.
-- Exclude claim count, exposure, bonus-malus and identifiers from model predictors.
SELECT c.claim_row_id, c.IDpol, c.ClaimAmount,
       p.VehPower, p.VehAge, p.DrivAge, p.Density,
       p.Area, p.VehBrand, p.VehGas, p.Region
FROM claims AS c
INNER JOIN policies AS p ON c.IDpol = p.IDpol
ORDER BY c.claim_row_id;
