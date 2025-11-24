-- Migration script to add custom_folder_path column to uploads table
-- Run this if you have an existing database

ALTER TABLE uploads ADD COLUMN custom_folder_path VARCHAR(255) NULL AFTER result_path;

-- Verify the change
DESCRIBE uploads;
