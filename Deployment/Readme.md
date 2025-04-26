# Travel Data Database Seeder

A utility script for seeding MongoDB with travel-related data (attractions, hotels, etc.) from CSV files. This tool prepares and imports data into your MongoDB database for use in travel recommendation applications.

## Overview

The database seeder script loads data from CSV files, processes it to ensure compatibility with MongoDB, and creates appropriate indexes for optimal query performance. It's designed to work with the travel recommendation application's data structure.

## Prerequisites

- Python 3.6+
- MongoDB (local or remote instance)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure your MongoDB connection details are properly configured in `config.py`

## Configuration

The script uses a `config.py` file in the parent directory. Make sure it contains the following settings:

```python
class Config:
    MONGO_URI = "mongodb://localhost:27017"
    MONGO_DB_NAME = "travel_db"
    MONGO_COLLECTION = "travel_data"
    # Other application settings...
```

Adjust these settings according to your MongoDB setup.

## Usage

### Basic Usage

Run the script with a path to your CSV file:

```
python seed_database.py path/to/your/travel_data.csv
```

### Options

- `--verify`: Verify the database contents after seeding
  ```
  python seed_database.py path/to/your/travel_data.csv --verify
  ```

### Example

```
python seed_database.py data/attractions_and_hotels.csv --verify
```

## CSV Data Format

Your CSV file should include the following columns:

### Required Columns
- `name`: Name of the attraction or hotel
- `type`: Type of record (e.g., "attraction", "hotel")
- `city`: City location
- `country`: Country location

### Recommended Columns
- `description`: Description text
- `rating`: Numerical rating (e.g., 4.5)
- `numberOfReviews`: Number of reviews
- `subcategories`: Category information
- `webUrl`: Website URL

### Hotel-Specific Columns
- `LowerPrice`: Minimum price
- `UpperPrice`: Maximum price
- `amenities`: List of amenities (stored as a string representation of a list)

## Features

- Automated data cleaning and preparation
- Conversion of string-format lists to actual MongoDB lists
- Handling of NaN values
- Creation of indexes for performance optimization
- Metadata tracking of database seeding operation
- Verification of imported data

## Database Structure

The script creates:
1. A main collection (as specified in `Config.MONGO_COLLECTION`)
2. A metadata collection that tracks:
   - When the database was last seeded
   - How many records were imported
   - Which source file was used

## Indexes

The following indexes are created for performance:
- `type`: For filtering by attraction/hotel type
- `country`: For country-based queries
- `city`: For city-based queries
- Text indexes on `name` and `description` for text search
- `rating`: For sorting by rating
- `subcategories`: For category filtering
- Compound index on `city` and `type`: For location-specific type searches

## Troubleshooting

### Connection Issues
If you're having trouble connecting to MongoDB:
- Verify your MongoDB service is running
- Check your connection string in `config.py`
- Ensure network access if using a remote database

### Data Import Problems
If your data isn't importing correctly:
- Check your CSV format and encoding
- Ensure column names match expected names
- Look for data type issues in your source CSV


## Contributors

[Harshitha C]