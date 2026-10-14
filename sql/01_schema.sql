-- Dedicated schema keeps this project's tables together
CREATE SCHEMA IF NOT EXISTS altitude;
CREATE TABLE IF NOT EXISTS altitude.study_days (
    airport text NOT NULL,
    day date NOT NULL,
    season text NOT NULL CHECK (season IN ('spring', 'summer', 'autumn')),
    split text NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    PRIMARY KEY (airport, day)
);
CREATE TABLE IF NOT EXISTS altitude.examples (
    airport text NOT NULL,
    day date NOT NULL,
    event_id text NOT NULL,
    icao24 text NOT NULL,
    trace_id text NOT NULL,
    origin_time double precision NOT NULL,
    target_time double precision NOT NULL,
    height_ft double precision NOT NULL,
    vertical_rate_fps double precision NOT NULL,
    past_height_rate_fps double precision NOT NULL,
    speed_knots double precision NOT NULL,
    distance_m double precision NOT NULL,
    lateral_m double precision NOT NULL,
    heading_error_deg double precision NOT NULL,
    past_gap_s double precision NOT NULL CHECK (past_gap_s BETWEEN 5 AND 30),
    horizon_s double precision NOT NULL CHECK (horizon_s BETWEEN 27 AND 33),
    target_height_ft double precision NOT NULL,
    baseline_ft double precision NOT NULL,
    residual_ft double precision NOT NULL,
    origin_metar_time double precision NOT NULL,
    origin_metar_available_time double precision NOT NULL,
    boundary_time double precision NOT NULL,
    PRIMARY KEY (airport, event_id, origin_time),
    FOREIGN KEY (airport, day) REFERENCES altitude.study_days (airport, day),
    CHECK (target_time > origin_time AND target_time < boundary_time),
    CHECK (origin_metar_time <= origin_metar_available_time),
    CHECK (origin_metar_available_time <= origin_time)
);
CREATE INDEX IF NOT EXISTS examples_day_idx ON altitude.examples (airport, day);
