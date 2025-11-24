# Custom Folder Path Guide

## Overview

This feature allows you to specify a custom location for storing audio segments and spectrograms instead of using the default `static/results` folder.

## Benefits

- **Flexible Storage**: Store files on external drives or network storage
- **Organization**: Keep different projects in separate locations
- **Disk Management**: Use storage locations with more available space

## How to Use

### 1. Using Default Storage

If you leave the "Custom Folder Path" field empty, files will be stored in:
```
app/static/results/<upload_id>/
```

### 2. Using Custom Storage

1. On the upload page, enter an **absolute path** in the "Custom Folder Path" field
   - Example (Linux/Mac): `/data/audio_analysis`
   - Example (Windows): `C:\Data\AudioAnalysis`

2. The system will create a subdirectory using the upload ID:
   ```
   /your/custom/path/<upload_id>/
   ```

3. Inside this folder, you'll find:
   - `audio_segments/` - Audio clips
   - `spectrograms/` - Spectrogram images
   - `spectrograms_training/` - Training-ready spectrograms

## Important Notes

### Permissions

Ensure the application has write permissions to the custom folder:

```bash
# Linux/Mac
sudo chmod 755 /your/custom/path
sudo chown -R <app_user>:<app_group> /your/custom/path

# Or for Docker
sudo chown -R 1000:1000 /your/custom/path
```

### Docker Volume Mounting

If using Docker, you must mount the custom path as a volume in `docker-compose.yml`:

```yaml
services:
  web:
    volumes:
      - ./static:/app/static
      - /your/custom/path:/your/custom/path  # Add this line
  
  worker:
    volumes:
      - ./static:/app/static
      - /your/custom/path:/your/custom/path  # Add this line
```

Restart the containers:
```bash
docker-compose down
docker-compose up -d
```

### Path Requirements

- Must be an **absolute path** (not relative)
- Directory must exist or be creatable
- Application needs write permissions
- Path should not contain special characters

## Examples

### Example 1: Local SSD
```
Custom Folder Path: /mnt/ssd/audio_data
Result: /mnt/ssd/audio_data/1/
```

### Example 2: Network Storage
```
Custom Folder Path: /mnt/nas/projects/audio
Result: /mnt/nas/projects/audio/1/
```

### Example 3: External Drive
```
Custom Folder Path: /media/external_drive/audio_analysis
Result: /media/external_drive/audio_analysis/1/
```

## Troubleshooting

### Permission Denied Error

**Problem**: Cannot create folder or write files

**Solution**:
1. Check folder permissions
2. Verify the path exists
3. Ensure Docker volume is mounted (if using Docker)

### Path Not Found

**Problem**: Directory does not exist

**Solution**:
1. Create the directory first:
   ```bash
   mkdir -p /your/custom/path
   ```
2. Set correct permissions

### Files Not Accessible from Web

**Problem**: Custom folder files cannot be viewed in browser

**Solution**: Custom folders are stored outside the web-accessible `static/` directory. This is by design for security. Files can still be:
- Accessed directly via filesystem
- Exported via CSV
- Used for training

## Deleting Records

When you delete an upload record:
- Files in custom folders will be automatically removed
- The subdirectory for that upload will be deleted
- The parent custom folder will remain

## Best Practices

1. **Use descriptive paths**: Make paths meaningful to your project
2. **Keep backups**: Regularly backup important analysis results
3. **Monitor disk space**: Ensure adequate space in custom locations
4. **Document paths**: Keep a record of which projects use which paths
5. **Test first**: Try with a small file before processing large datasets
