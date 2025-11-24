# CSV Export Guide

## Overview

The CSV export feature allows you to download all labeling data in a structured spreadsheet format, making it easy to:
- Review labeled data
- Share results with team members
- Import into other analysis tools
- Create reports

## How to Export CSV

### Method 1: From Results Page

1. Navigate to **History** → Select an upload → **View Results**
2. Click the **"Download CSV"** button
3. The CSV file will download automatically

### Method 2: From Labeling Page

1. Navigate to **History** → Select an upload → **Labeling Mode**
2. Click the **"Download CSV"** button
3. The CSV file will download automatically

## CSV File Format

### File Name Convention
```
labels_<original_filename>_<upload_id>.csv
```

Example: `labels_audio_sample.wav_42.csv`

### Column Structure

| Column | Description | Example |
|--------|-------------|----------|
| ID | Unique result identifier | 123 |
| Audio Filename | Name of audio segment file | `segment_000.wav` |
| Spectrogram Filename | Name of spectrogram image | `spec_000.png` |
| Label | Assigned label name | `Bird Song` |

### Example CSV Content

```csv
ID,Audio Filename,Spectrogram Filename,Label
1,segment_000.wav,spec_000.png,Bird Song
2,segment_001.wav,spec_001.png,Wind Noise
3,segment_002.wav,spec_002.png,No Label
4,segment_003.wav,spec_003.png,Bird Song
```

## Features

### UTF-8 BOM Encoding

CSV files use UTF-8 with BOM (Byte Order Mark) encoding:
- **Benefit**: Opens correctly in Excel without encoding issues
- **Compatible with**: Excel, Google Sheets, LibreOffice
- **Special characters**: Fully supported (Chinese, Japanese, etc.)

### No Label Handling

Items without labels show:
```
No Label
```

This makes it easy to identify unlabeled items for review.

## Using the CSV File

### In Excel

1. Double-click the CSV file
2. Excel will open it automatically with correct formatting
3. All special characters will display correctly

### In Google Sheets

1. Go to Google Sheets
2. **File** → **Import** → Upload CSV file
3. Choose "UTF-8" encoding
4. Click **Import**

### In Python (pandas)

```python
import pandas as pd

df = pd.read_csv('labels_audio_sample.wav_42.csv', encoding='utf-8-sig')
print(df.head())

# Count labels
label_counts = df['Label'].value_counts()
print(label_counts)

# Filter by label
bird_songs = df[df['Label'] == 'Bird Song']
```

### In R

```r
library(readr)

data <- read_csv('labels_audio_sample.wav_42.csv')
head(data)

# Summary statistics
table(data$Label)
```

## Advanced Usage

### Filtering Data

You can filter the CSV to find specific patterns:

```python
import pandas as pd

df = pd.read_csv('labels.csv', encoding='utf-8-sig')

# Find all bird songs
birds = df[df['Label'] == 'Bird Song']

# Find unlabeled items
unlabeled = df[df['Label'] == 'No Label']

# Export filtered data
birds.to_csv('bird_songs_only.csv', index=False)
```

### Generating Statistics

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('labels.csv', encoding='utf-8-sig')

# Count each label
label_counts = df['Label'].value_counts()

# Plot distribution
label_counts.plot(kind='bar')
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('label_distribution.png')
```

### Merging Multiple Exports

```python
import pandas as pd
import glob

# Read all CSV files
csv_files = glob.glob('labels_*.csv')
dfs = [pd.read_csv(f, encoding='utf-8-sig') for f in csv_files]

# Merge into single dataframe
combined = pd.concat(dfs, ignore_index=True)

# Save combined data
combined.to_csv('all_labels_combined.csv', index=False, encoding='utf-8-sig')
```

## Troubleshooting

### Excel Shows Garbled Characters

**Problem**: Special characters display incorrectly

**Solution**: The file should open correctly by default. If not:
1. Open Excel
2. Go to **Data** → **From Text/CSV**
3. Select the CSV file
4. Choose **UTF-8** encoding
5. Click **Load**

### Empty Label Column

**Problem**: All labels show as "No Label"

**Solution**: This means the items haven't been labeled yet. Go to the labeling page to assign labels.

### File Won't Download

**Problem**: Clicking "Download CSV" does nothing

**Solution**:
1. Check browser console for errors (F12)
2. Ensure the upload has completed processing
3. Try a different browser
4. Check that the upload_id exists

## Best Practices

1. **Export regularly**: Save labeled data periodically to avoid data loss
2. **Version control**: Use descriptive filenames or add timestamps
3. **Backup**: Keep CSV backups separate from the application
4. **Documentation**: Note what each label represents
5. **Review**: Use CSV to review labeling consistency

## Integration Examples

### Creating Training Reports

```python
import pandas as pd

df = pd.read_csv('labels.csv', encoding='utf-8-sig')

report = f"""
=== Labeling Report ===
Total Items: {len(df)}
Labeled: {len(df[df['Label'] != 'No Label'])}
Unlabeled: {len(df[df['Label'] == 'No Label'])}

Label Distribution:
{df['Label'].value_counts().to_string()}
"""

print(report)

# Save report
with open('labeling_report.txt', 'w') as f:
    f.write(report)
```

### Quality Control Check

```python
import pandas as pd

df = pd.read_csv('labels.csv', encoding='utf-8-sig')

# Find potential issues
print("Unlabeled items:", len(df[df['Label'] == 'No Label']))
print("Total unique labels:", df['Label'].nunique())
print("Items per label:")
print(df['Label'].value_counts())

# Flag underrepresented labels
min_samples = 10
small_classes = df['Label'].value_counts()[df['Label'].value_counts() < min_samples]
if len(small_classes) > 0:
    print("\nWarning: These labels have fewer than", min_samples, "samples:")
    print(small_classes)
```
