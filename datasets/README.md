# datasets/

Store datasets used by the Data Science Lab here.

Supported formats include CSV, Excel, JSON, and Parquet. Large datasets are
git-ignored by default (see `.gitignore`), so either commit small sample
datasets explicitly or document where to download them in a `README.md` next
to the data.

## Bundled sample datasets (`samples/`)

The Dataset Explorer loads these from the `samples/` subfolder:

- `student_grades.csv` - 122 student exam records with missing values and
  duplicate rows, for practicing inspection and data-quality checks.
- `sales_records.xlsx` - 152 store sales records demonstrating an Excel
  workbook with categorical, numeric, and datetime columns.

These files intentionally contain quality issues so students can practice
detecting missing values, duplicates, and constant columns.
