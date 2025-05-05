import pandas as pd
import os
import re
from tqdm import tqdm

def extract_sections(report_text):

    text = report_text.replace('\n', ' ').replace('\r', ' ').lower()

    findings_match = re.search(r'findings:(.*?)(impression:|$)', text)
    impression_match = re.search(r'impression:(.*)', text)

    findings = findings_match.group(1).strip() if findings_match else ""
    impression = impression_match.group(1).strip() if impression_match else ""

    return findings, impression

def extract_reports(report_csv_path, reports_root_dir, output_path):

    metadata = pd.read_csv(report_csv_path)
    extracted = []

    print(f"Loaded metadata with {len(metadata)} entries")

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        file_path = os.path.join(reports_root_dir, row['path'])

        if not os.path.exists(file_path):
            print(f"{file_path} does not exist!")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Could not read {file_path}: {e}")
            continue

        findings, impression = extract_sections(text)

        extracted.append({
            "subject_id": row['subject_id'],
            "study_id": row['study_id'],
            "findings": findings,
            "impression": impression,
            "full_text": text.strip()
        })

    print(f"Extracted {len(extracted)} reports. Saving to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(extracted).to_csv(output_path, index=False)

if __name__ == "__main__":
    extract_reports(
        report_csv_path="../mimic-cxr-reports/cxr-study-list.csv",
        reports_root_dir="../mimic-cxr-reports",
        output_path="../data/radiology_reports.csv"
    )
