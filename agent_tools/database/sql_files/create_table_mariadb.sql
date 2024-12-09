-- add you database name!
CREATE TABLE IF NOT EXISTS `you database here`.`chrome_bookmarks` (
    `folder_id`VARCHAR(300) NOT NULL COLLATE 'utf8mb4_general_ci',
    `folder_name` VARCHAR(300) NOT NULL COLLATE 'utf8mb4_general_ci' COMMENT 'Name of the folder tab',
    `url_guid` VARCHAR(500) NOT NULL COLLATE 'utf8mb4_general_ci' COMMENT 'Unique UUID of the url address',
	  `url_id` VARCHAR(300) NOT NULL COLLATE 'utf8mb4_general_ci',
	  `url_tab_name` VARCHAR(2000) NULL COLLATE 'utf8mb4_general_ci' COMMENT 'URL tab description',
	  `url` LONGTEXT NOT NULL COLLATE 'utf8mb4_general_ci' COMMENT 'URL address',
    `description` VARCHAR(500) NULL COLLATE 'utf8mb4_general_ci' COMMENT 'Description of the website content',
    `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY `unique_url` (`url`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
