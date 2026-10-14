-- Проверяем сохранение количества строк и единственность соответствия
SELECT
    (SELECT count(*) FROM altitude.examples WHERE airport = 'LFPG') AS source_rows,
    (SELECT count(*) FROM altitude.examples e
     JOIN altitude.study_days d USING (airport, day)
     WHERE e.airport = 'LFPG') AS joined_rows,
    (SELECT count(*) FROM altitude.examples e
     LEFT JOIN altitude.study_days d USING (airport, day)
     WHERE e.airport = 'LFPG' AND d.airport IS NULL) AS unmatched_rows,
    (SELECT count(*) FROM (
        SELECT e.airport, e.event_id, e.origin_time
        FROM altitude.examples e JOIN altitude.study_days d USING (airport, day)
        WHERE e.airport = 'LFPG'
        GROUP BY e.airport, e.event_id, e.origin_time HAVING count(*) <> 1
    ) conflicts) AS nonunique_matches;

-- Показываем присоединенные метаданные на нескольких настоящих наблюдениях
SELECT e.airport, e.day, e.event_id, e.origin_time, d.season, d.split
FROM altitude.examples e JOIN altitude.study_days d USING (airport, day)
WHERE e.airport = 'LFPG'
ORDER BY e.day, e.event_id, e.origin_time LIMIT 5;

-- Считаем объем выборок после соединения
SELECT d.split, count(*) AS pairs, count(DISTINCT e.event_id) AS events,
       count(DISTINCT e.day) AS days
FROM altitude.examples e JOIN altitude.study_days d USING (airport, day)
WHERE e.airport = 'LFPG'
GROUP BY d.split ORDER BY d.split;
