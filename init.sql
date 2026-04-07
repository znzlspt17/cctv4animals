-- PostgreSQL 초기화 SQL
-- 원격 PostgreSQL 서버에서 superuser 권한으로 1회 실행

-- 1. pgvector 확장 활성화 (PostgreSQL 서버에 vector 모듈이 설치된 경우)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 현재 face_images 테이블의 embedding 컬럼(BYTEA)을 vector 타입으로 추가
--    (기존 BYTEA 컬럼은 호환성 유지를 위해 보존)
ALTER TABLE face_images ADD COLUMN IF NOT EXISTS embedding_vec vector(512);

-- 3. HNSW 인덱스 생성 (cosine distance 기반 ANN 검색)
CREATE INDEX IF NOT EXISTS idx_face_images_embedding_vec
    ON face_images USING hnsw (embedding_vec vector_cosine_ops);

-- 4. 동물 탐지 로그 테이블
CREATE TABLE IF NOT EXISTS animal_detection_logs (
    id          BIGSERIAL PRIMARY KEY,
    source      VARCHAR(50)  NOT NULL DEFAULT 'api',
    class_name  VARCHAR(100) NOT NULL,
    confidence  FLOAT        NOT NULL,
    bbox_x1     FLOAT,
    bbox_y1     FLOAT,
    bbox_x2     FLOAT,
    bbox_y2     FLOAT,
    detected_at TIMESTAMP    NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_animal_logs_class_at ON animal_detection_logs (class_name, detected_at);
CREATE INDEX IF NOT EXISTS ix_animal_logs_detected_at ON animal_detection_logs (detected_at);

-- 5. 식물 탐지 로그 테이블
CREATE TABLE IF NOT EXISTS plant_detection_logs (
    id                  BIGSERIAL PRIMARY KEY,
    source              VARCHAR(50)  NOT NULL DEFAULT 'api',
    class_name          VARCHAR(100) NOT NULL,
    disease_code        INTEGER,
    disease_label       VARCHAR(100),
    confidence          FLOAT        NOT NULL,
    bbox_x1             FLOAT,
    bbox_y1             FLOAT,
    bbox_x2             FLOAT,
    bbox_y2             FLOAT,
    crop_type           INTEGER,
    crop_name           VARCHAR(100),
    shooting_type       INTEGER,
    shooting_type_name  VARCHAR(50),
    grow_stage          INTEGER,
    grow_stage_name     VARCHAR(50),
    area                INTEGER,
    area_name           VARCHAR(50),
    detected_at         TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_plant_logs_class_at ON plant_detection_logs (class_name, detected_at);
CREATE INDEX IF NOT EXISTS ix_plant_logs_detected_at ON plant_detection_logs (detected_at);

-- 6. 탐지 이미지 바이트스트림 컬럼 추가 (기존 DB 마이그레이션용)
ALTER TABLE animal_detection_logs ADD COLUMN IF NOT EXISTS image_data BYTEA;
ALTER TABLE plant_detection_logs  ADD COLUMN IF NOT EXISTS image_data BYTEA;
