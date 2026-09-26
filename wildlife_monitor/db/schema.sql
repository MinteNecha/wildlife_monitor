-- Wildlife Monitor relational schema, third normal form.
--
-- Implements the design in section 5 of the detailed design document. The flat
-- DetectionRecord row is decomposed so that every non-key attribute depends on
-- the key, the whole key, and nothing but the key:
--
--   * camera coordinates and habitat depend on the camera, not the detection
--     -> Camera, Habitat
--   * capture time, file path and ground truth depend on the image, not on
--     which pipeline processed it                     -> Image
--   * model configuration depends on the pipeline      -> Pipeline
--   * species naming and prompt depend on the species  -> Species
--   * the top-5 candidate list is repeating data       -> Prediction
--
-- Deviation from the design document: Detection carries instance_count. The
-- multi-instance counting added for MegaDetector post-dates the schema in the
-- proposal, and the count is a property of one pipeline's reading of one
-- image, so Detection is where it belongs.
--
-- Large binaries (JPEGs, masks, overlays) stay on disk; the database stores
-- paths to them.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS Habitat (
    habitat_type    TEXT PRIMARY KEY,
    description     TEXT
);

CREATE TABLE IF NOT EXISTS Camera (
    camera_id       TEXT PRIMARY KEY,
    site_name       TEXT,
    latitude        REAL NOT NULL,
    longitude       REAL NOT NULL,
    habitat_type    TEXT REFERENCES Habitat(habitat_type),
    season          TEXT
);

CREATE TABLE IF NOT EXISTS Species (
    species_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name  TEXT NOT NULL UNIQUE,
    common_name     TEXT,
    scientific_name TEXT,
    text_prompt     TEXT
);

CREATE TABLE IF NOT EXISTS Image (
    image_id        TEXT PRIMARY KEY,
    camera_id       TEXT NOT NULL REFERENCES Camera(camera_id),
    captured_at     TEXT,
    file_path       TEXT NOT NULL,
    width           INTEGER,
    height          INTEGER,
    ground_truth_id INTEGER REFERENCES Species(species_id)
);

CREATE TABLE IF NOT EXISTS Pipeline (
    pipeline_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    retrieval_model TEXT,
    localiser_model TEXT,
    version         TEXT
);

CREATE TABLE IF NOT EXISTS Detection (
    detection_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id          TEXT NOT NULL REFERENCES Image(image_id) ON DELETE CASCADE,
    pipeline_id       INTEGER NOT NULL REFERENCES Pipeline(pipeline_id),
    species_id        INTEGER NOT NULL REFERENCES Species(species_id),
    confidence        REAL CHECK (confidence BETWEEN 0 AND 1),
    detection_quality REAL,
    location_type     TEXT,
    location          TEXT,
    instance_count    INTEGER DEFAULT 1,
    mask_path         TEXT,
    overlay_path      TEXT,
    created_at        TEXT DEFAULT (datetime('now')),
    UNIQUE (image_id, pipeline_id)
);

CREATE TABLE IF NOT EXISTS Prediction (
    prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id    INTEGER NOT NULL REFERENCES Detection(detection_id) ON DELETE CASCADE,
    species_id      INTEGER NOT NULL REFERENCES Species(species_id),
    rank            INTEGER NOT NULL,
    score           REAL,
    UNIQUE (detection_id, rank)
);

-- ── Pipeline 2 (P3) ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS Sequence (
    sequence_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    species_id      INTEGER NOT NULL REFERENCES Species(species_id),
    camera_id       TEXT NOT NULL REFERENCES Camera(camera_id),
    start_time      TEXT,
    end_time        TEXT,
    detection_count INTEGER,
    peak_month      INTEGER,
    UNIQUE (species_id, camera_id)
);

CREATE TABLE IF NOT EXISTS SequenceMember (
    sequence_id     INTEGER NOT NULL REFERENCES Sequence(sequence_id) ON DELETE CASCADE,
    detection_id    INTEGER NOT NULL REFERENCES Detection(detection_id) ON DELETE CASCADE,
    position        INTEGER NOT NULL,
    PRIMARY KEY (sequence_id, detection_id)
);

CREATE TABLE IF NOT EXISTS BehaviourPattern (
    pattern_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_id         INTEGER NOT NULL REFERENCES Sequence(sequence_id) ON DELETE CASCADE,
    activity_class      TEXT,
    movement_class      TEXT,
    social_class        TEXT,
    activity_confidence REAL,
    movement_confidence REAL,
    model_version       TEXT,
    created_at          TEXT DEFAULT (datetime('now')),
    UNIQUE (sequence_id, model_version)
);

CREATE TABLE IF NOT EXISTS Validation (
    validation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id      INTEGER NOT NULL REFERENCES BehaviourPattern(pattern_id) ON DELETE CASCADE,
    verdict         TEXT NOT NULL CHECK (verdict IN ('validated', 'novel', 'spurious')),
    notes           TEXT,
    validated_at    TEXT DEFAULT (datetime('now'))
);

-- ── Indexes for the queries the dashboard actually runs ──────────────────────

CREATE INDEX IF NOT EXISTS idx_image_camera      ON Image(camera_id);
CREATE INDEX IF NOT EXISTS idx_detection_species ON Detection(species_id);
CREATE INDEX IF NOT EXISTS idx_detection_image   ON Detection(image_id);
CREATE INDEX IF NOT EXISTS idx_detection_pipe    ON Detection(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_pattern_sequence  ON BehaviourPattern(sequence_id);
CREATE INDEX IF NOT EXISTS idx_validation_pat    ON Validation(pattern_id);

-- ── A flat view matching the original CSV layout ─────────────────────────────
-- Pipeline 2 and the dashboard consume detections as a wide row. Expressing
-- that as a view keeps the normalised tables authoritative while giving
-- consumers the shape they already expect.

CREATE VIEW IF NOT EXISTS DetectionFlat AS
SELECT
    d.detection_id                AS detection_id,
    i.image_id                    AS image_id,
    p.name                        AS pipeline,
    i.captured_at                 AS timestamp,
    c.camera_id                   AS camera_id,
    c.latitude                    AS latitude,
    c.longitude                   AS longitude,
    COALESCE(c.habitat_type, 'unknown') AS habitat_type,
    s.canonical_name              AS species,
    d.confidence                  AS confidence,
    d.location_type               AS location_type,
    d.location                    AS location,
    d.detection_quality           AS detection_quality,
    d.instance_count              AS instance_count,
    i.file_path                   AS image_path,
    COALESCE(d.mask_path, '')     AS mask_path,
    COALESCE(d.overlay_path, '')  AS overlay_path,
    COALESCE(g.canonical_name, 'unknown') AS ground_truth_species,
    CASE
        WHEN g.canonical_name IS NULL THEN 'unknown'
        WHEN g.canonical_name = s.canonical_name THEN 'correct'
        ELSE 'incorrect'
    END                           AS correct
FROM Detection d
JOIN Image    i ON i.image_id   = d.image_id
JOIN Camera   c ON c.camera_id  = i.camera_id
JOIN Pipeline p ON p.pipeline_id = d.pipeline_id
JOIN Species  s ON s.species_id = d.species_id
LEFT JOIN Species g ON g.species_id = i.ground_truth_id;
