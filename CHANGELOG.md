# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Custom folder path selection for storing audio segments and spectrograms
- CSV export functionality for labeled data
- "Download CSV" button on results and labeling pages
- Custom storage path display on results page
- Database migration script for `custom_folder_path` column

### Changed
- Updated `Upload` model to include `custom_folder_path` field
- Modified upload form to include custom folder path input
- Enhanced `upload()` route to handle custom storage locations
- Updated `delete_selected_uploads()` to properly delete custom folder contents

### Technical Details
- New database column: `uploads.custom_folder_path` (VARCHAR(255), nullable)
- New API endpoint: `GET /export_csv/<upload_id>`
- CSV export includes: ID, Audio Filename, Spectrogram Filename, Label
- UTF-8 BOM encoding for Excel compatibility

## [1.0.0] - 2025-11-17

### Added
- Initial release
- Audio upload and processing
- Spectrogram generation (Mel, STFT, DEMON)
- Manual labeling interface
- Automatic labeling with model upload
- YOLOv8 model training integration
- Training report visualization
- Docker containerization
