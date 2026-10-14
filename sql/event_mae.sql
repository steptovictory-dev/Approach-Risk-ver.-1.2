-- Сначала считаем ошибку каждого захода, затем сравниваем два способа усреднения
WITH event_errors AS (
    SELECT e.airport, d.split, e.event_id,
           count(*) AS pairs,
           avg(abs(e.target_height_ft - e.baseline_ft)) AS event_mae_ft
    FROM altitude.examples e JOIN altitude.study_days d USING (airport, day)
    WHERE e.airport = 'LFPG'
    GROUP BY e.airport, d.split, e.event_id
)
SELECT airport, split, count(*) AS events, sum(pairs) AS pairs,
       avg(event_mae_ft) AS event_macro_mae_ft,
       sum(pairs * event_mae_ft) / sum(pairs) AS row_mae_ft,
       min(pairs) AS min_pairs,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY pairs) AS median_pairs,
       max(pairs) AS max_pairs,
       corr(pairs::double precision, event_mae_ft) AS pairs_mae_correlation
FROM event_errors GROUP BY airport, split ORDER BY split;

-- Распределение количества пар без публикации идентификаторов заходов
WITH event_sizes AS (
    SELECT e.airport, d.split, e.event_id, count(*) AS pairs
    FROM altitude.examples e JOIN altitude.study_days d USING (airport, day)
    WHERE e.airport = 'LFPG'
    GROUP BY e.airport, d.split, e.event_id
)
SELECT airport, split, pairs, count(*) AS events
FROM event_sizes GROUP BY airport, split, pairs ORDER BY split, pairs;
