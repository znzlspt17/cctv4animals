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

-- 4. 기존 BYTEA 임베딩 → vector 컬럼으로 마이그레이션
--    (Python 측에서 수행 권장 — numpy frombuffer → list 변환 후 UPDATE)
--    아래는 참고용 예시 (실제 데이터 구조에 따라 조정 필요)
-- UPDATE face_images
--    SET embedding_vec = embedding::vector
--  WHERE embedding IS NOT NULL AND embedding_vec IS NULL;
