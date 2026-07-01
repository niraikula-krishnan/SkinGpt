CREATE DATABASE IF NOT EXISTS skingpt CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE skingpt;

CREATE TABLE IF NOT EXISTS user_analytics (
    user_number VARCHAR(32) PRIMARY KEY,
    user_label VARCHAR(80) NOT NULL,
    analytics JSON NOT NULL,
    last_disease VARCHAR(100) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Users are added dynamically from the app during each session.
