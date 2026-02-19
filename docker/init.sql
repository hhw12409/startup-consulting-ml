-- 📁 docker/init.sql
-- =====================
-- 창업 컨설턴트 AI — MySQL 스키마

USE startup_consultant;

-- ================================================================
-- 0. 전용 사용자 생성
-- ================================================================
CREATE USER IF NOT EXISTS 'startup'@'%' IDENTIFIED BY 'startup1234';
GRANT ALL PRIVILEGES ON startup_consultant.* TO 'startup'@'%';
FLUSH PRIVILEGES;

-- ================================================================
-- 1. 상가 원본 데이터 (공공데이터 API 수집 결과)
-- ================================================================
CREATE TABLE IF NOT EXISTS stores (
                                      id BIGINT AUTO_INCREMENT PRIMARY KEY,

    -- 사업자 정보
                                      biz_id VARCHAR(20) COMMENT '사업자번호 (bizesId)',
    store_name VARCHAR(200) COMMENT '상호명 (bizesNm)',
    branch_name VARCHAR(100) COMMENT '지점명 (brchNm)',

    -- 업종 분류
    category_large_cd VARCHAR(10) COMMENT '상권업종대분류코드 (indsLclsCd)',
    category_large VARCHAR(50) COMMENT '상권업종대분류명 (indsLclsCdNm)',
    category_mid_cd VARCHAR(10) COMMENT '상권업종중분류코드 (indsMclsCd)',
    category_mid VARCHAR(50) COMMENT '상권업종중분류명 (indsMclsCdNm)',
    category_small_cd VARCHAR(10) COMMENT '상권업종소분류코드 (indsSclsCd)',
    category_small VARCHAR(100) COMMENT '상권업종소분류명 (indsSclsCdNm)',

    -- 표준산업분류
    ksic_cd VARCHAR(10) COMMENT '표준산업분류코드 (ksicCd)',
    ksic_name VARCHAR(100) COMMENT '표준산업분류명 (ksicNm)',

    -- 지역 정보
    sido_cd VARCHAR(5) COMMENT '시도코드',
    sido_name VARCHAR(20) COMMENT '시도명',
    sgg_cd VARCHAR(5) COMMENT '시군구코드',
    sgg_name VARCHAR(20) COMMENT '시군구명',
    adong_cd VARCHAR(10) COMMENT '행정동코드',
    adong_name VARCHAR(30) COMMENT '행정동명',
    ldong_cd VARCHAR(10) COMMENT '법정동코드',
    ldong_name VARCHAR(30) COMMENT '법정동명',

    -- 주소
    lot_address VARCHAR(300) COMMENT '지번주소 (lnoAdr)',
    road_address VARCHAR(300) COMMENT '도로명주소 (rdnmAdr)',
    building_name VARCHAR(100) COMMENT '건물명 (bldNm)',
    zip_code VARCHAR(10) COMMENT '우편번호 (nwZipCd)',

    -- 위치
    longitude DECIMAL(11, 8) COMMENT '경도 (lon)',
    latitude DECIMAL(10, 8) COMMENT '위도 (lat)',

    -- 층/호 정보
    floor_info VARCHAR(20) COMMENT '층정보 (flrNo)',
    unit_info VARCHAR(20) COMMENT '호정보 (hoNo)',

    -- 사업자 상태 (국세청 API)
    biz_status_cd VARCHAR(5) COMMENT '사업자상태코드 (01=계속, 02=휴업, 03=폐업)',
    biz_status VARCHAR(10) COMMENT '사업자상태명',
    closure_date VARCHAR(10) COMMENT '폐업일 (end_dt)',

    -- 메타
    data_ym VARCHAR(6) COMMENT '데이터 기준년월 (stdrYm)',
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '수집 시각',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- 인덱스
    UNIQUE KEY uk_biz_id (biz_id),
    INDEX idx_category (category_large),
    INDEX idx_adong (adong_cd, adong_name),
    INDEX idx_sgg (sgg_cd),
    INDEX idx_status (biz_status_cd),
    INDEX idx_collected (collected_at)

    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    COMMENT='소상공인 상가 원본 데이터';


-- ================================================================
-- 2. 행정동 코드 테이블
-- ================================================================
CREATE TABLE IF NOT EXISTS region_codes (
                                            id INT AUTO_INCREMENT PRIMARY KEY,
                                            region_cd VARCHAR(10) NOT NULL COMMENT '행정동코드 10자리',
    region_cd_8 VARCHAR(8) NOT NULL COMMENT '행정동코드 8자리 (상가 API용)',
    sido_cd VARCHAR(2) COMMENT '시도코드',
    sgg_cd VARCHAR(3) COMMENT '시군구코드',
    dong_cd VARCHAR(3) COMMENT '동코드',
    sido_name VARCHAR(20) COMMENT '시도명',
    sgg_name VARCHAR(20) COMMENT '시군구명',
    dong_name VARCHAR(30) COMMENT '동명',
    full_name VARCHAR(80) COMMENT '전체 명칭',

    UNIQUE KEY uk_region_cd (region_cd),
    INDEX idx_region_cd_8 (region_cd_8),
    INDEX idx_sido (sido_cd)

    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    COMMENT='행정동 코드 마스터';


-- ================================================================
-- 3. 수집 이력 테이블
-- ================================================================
CREATE TABLE IF NOT EXISTS collection_logs (
                                               id BIGINT AUTO_INCREMENT PRIMARY KEY,
                                               dong_cd VARCHAR(10) NOT NULL COMMENT '수집 행정동코드',
    dong_name VARCHAR(30) COMMENT '행정동명',
    store_count INT DEFAULT 0 COMMENT '수집 건수',
    status ENUM('success', 'fail', 'no_data') DEFAULT 'success',
    error_msg TEXT COMMENT '에러 메시지',
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_dong (dong_cd),
    INDEX idx_status (status),
    INDEX idx_collected (collected_at)

    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    COMMENT='데이터 수집 이력';