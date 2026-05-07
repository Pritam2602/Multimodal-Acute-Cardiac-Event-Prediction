from pathlib import Path

import boto3


PROJECT_ROOT = Path(__file__).resolve().parent
DESTINATION = PROJECT_ROOT / "mimic_data" / "machine_measurements.csv"

BUCKET = "arn:aws:s3:us-east-1:724665945834:accesspoint/mimic-iv-ecg-v1-0-01"
KEY = "mimic-iv-ecg/1.0/machine_measurements.csv"


def load_dotenv(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)

    env = load_dotenv(PROJECT_ROOT / ".env")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=env.get("AWS_ACCESS_KEY"),
        aws_secret_access_key=env.get("AWS_SECRET_KEY"),
        region_name=env.get("AWS_REGION", "us-east-1"),
    )
    s3.download_file(BUCKET, KEY, str(DESTINATION))

    print(f"Downloaded successfully: {DESTINATION}")


if __name__ == "__main__":
    main()
