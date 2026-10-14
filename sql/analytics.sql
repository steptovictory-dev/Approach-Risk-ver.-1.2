-- 1: Inventory by frozen season and split
SELECT d.airport, d.season, d.split, count(*) AS pairs,
       count(DISTINCT e.event_id) AS events
FROM altitude.examples e JOIN altitude.study_days d USING (airport, day)
GROUP BY d.airport, d.season, d.split ORDER BY 1, 2, 3;

-- 2: Each approach contributes equally to the baseline metric
WITH event_errors AS (
    SELECT e.airport, d.split, e.event_id,
           avg(abs(e.target_height_ft - e.baseline_ft)) AS mae_ft
    FROM altitude.examples e JOIN altitude.study_days d USING (airport, day)
    GROUP BY e.airport, d.split, e.event_id
)
SELECT airport, split, avg(mae_ft) AS macro_mae_ft
FROM event_errors GROUP BY airport, split ORDER BY 1, 2;

-- 3: METAR observation age at the forecast origin
SELECT airport, day, avg((origin_time-origin_metar_time)/60) AS mean_age_minutes,
       max((origin_time-origin_metar_time)/60) AS max_age_minutes
FROM altitude.examples GROUP BY airport, day ORDER BY 1, 2;

-- 4: Actual horizons retain their original spread
SELECT airport, horizon_s, count(*) AS pairs
FROM altitude.examples GROUP BY airport, horizon_s ORDER BY 1, 2;

-- 5: Missing chronology is visible through a window function
WITH gaps AS (
    SELECT airport, event_id, origin_time - lag(origin_time) OVER (
        PARTITION BY airport, event_id ORDER BY origin_time) AS gap_s
    FROM altitude.examples
)
SELECT airport, max(gap_s) AS largest_origin_gap_s
FROM gaps GROUP BY airport;

-- 6: Aircraft overlap describes dependence across the frozen splits
SELECT e.airport, e.icao24, count(DISTINCT d.split) AS splits,
       string_agg(DISTINCT d.split, ', ' ORDER BY d.split) AS split_names
FROM altitude.examples e JOIN altitude.study_days d USING (airport, day)
GROUP BY e.airport, e.icao24 HAVING count(DISTINCT d.split) > 1;

-- 7: Check the physics and target construction
SELECT airport,
       max(abs(target_time-origin_time-horizon_s)) AS horizon_difference_s,
       max(abs(height_ft+vertical_rate_fps*horizon_s-baseline_ft)) AS baseline_difference_ft,
       max(abs(target_height_ft-baseline_ft-residual_ft)) AS residual_difference_ft
FROM altitude.examples GROUP BY airport;

-- 8: An event crossing splits would invalidate the experiment
SELECT e.airport, e.event_id, count(DISTINCT d.split) AS splits
FROM altitude.examples e JOIN altitude.study_days d USING (airport, day)
GROUP BY e.airport, e.event_id HAVING count(DISTINCT d.split) > 1;
