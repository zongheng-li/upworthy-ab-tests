-- Build the analysis tables from the raw Upworthy exploratory file.
-- Run from Python with src.upworthy.connect(), which fills in {data_path}.
--
-- Output tables
--   packages : raw data, one row per package (= one arm of a test)
--   tests    : one row per test, with quality flags
--   arms     : one row per arm, joined to its test, with CTR

CREATE OR REPLACE TABLE packages AS
SELECT *
FROM read_csv_auto('{data_path}');

-- One row per test.
-- is_aa: every arm shows the same headline AND the same image,
--        so any difference between arms is noise (an "A/A test").
-- in_problem_period: per the archive documentation, tests created
--        between 2013-06-25 and 2014-01-10 may have randomization problems.
CREATE OR REPLACE TABLE tests AS
SELECT
    clickability_test_id                                              AS test_id,
    MIN(created_at)                                                   AS test_start,
    COUNT(*)                                                          AS n_arms,
    COUNT(DISTINCT headline)                                          AS n_headlines,
    COUNT(DISTINCT COALESCE(eyecatcher_id, 'none'))                   AS n_images,
    COUNT(DISTINCT headline || '|' || COALESCE(eyecatcher_id, 'none')) AS n_variants,
    SUM(impressions)                                                  AS impressions,
    SUM(clicks)                                                       AS clicks,
    SUM(clicks) * 1.0 / SUM(impressions)                              AS ctr_pooled,
    COUNT(DISTINCT headline || '|' || COALESCE(eyecatcher_id, 'none')) = 1
                                                                      AS is_aa,
    MIN(created_at) >= '2013-06-25' AND MIN(created_at) < '2014-01-11'
                                                                      AS in_problem_period
FROM packages
GROUP BY clickability_test_id;

-- One row per arm, with its own CTR and its test's pooled CTR.
CREATE OR REPLACE TABLE arms AS
SELECT
    p.clickability_test_id          AS test_id,
    p.headline,
    p.eyecatcher_id,
    p.impressions,
    p.clicks,
    p.clicks * 1.0 / p.impressions  AS ctr,
    p.winner                        AS upworthy_winner,
    t.n_arms,
    t.ctr_pooled,
    t.is_aa,
    t.in_problem_period
FROM packages p
JOIN tests t
  ON p.clickability_test_id = t.test_id;
