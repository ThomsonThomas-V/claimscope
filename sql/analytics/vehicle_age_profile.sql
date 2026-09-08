-- Does observed severity vary across broad vehicle-age bands?
SELECT CASE WHEN p.VehAge < 5 THEN '0-4' WHEN p.VehAge < 10 THEN '5-9'
            ELSE '10+' END AS vehicle_age_band,
       COUNT(*) AS claim_records, AVG(c.ClaimAmount) AS mean_recorded_amount
FROM claims c JOIN policies p ON c.IDpol = p.IDpol
GROUP BY vehicle_age_band;
