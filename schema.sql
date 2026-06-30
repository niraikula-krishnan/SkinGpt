CREATE DATABASE IF NOT EXISTS skingpt CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE skingpt;

CREATE TABLE IF NOT EXISTS user_analytics (
    user_number VARCHAR(20) PRIMARY KEY,
    user_label VARCHAR(50) NOT NULL,
    analytics JSON NOT NULL,
    last_disease VARCHAR(100) DEFAULT NULL
);

INSERT INTO user_analytics (user_number, user_label, analytics) VALUES
    ('user_1', 'User 1', '{}'),
    ('user_2', 'User 2', '{}'),
    ('user_3', 'User 3', '{}')
ON DUPLICATE KEY UPDATE user_label = VALUES(user_label);
