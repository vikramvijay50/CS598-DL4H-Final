import os

def run_pipeline():
    print("\nStep 1: Extract radiology reports from MIMIC-CXR")
    os.system("python extract_radiology_reports.py")

    print("\nStep 2: Preprocess reports (split into sentences)")
    os.system("python preprocess.py")

    print("\nStep 3: Annotate sentences with pseudo-teacher labels")
    os.system("python annotate_teacher.py")

    print("\nStep 4: Train student model with distillation (classification)")
    os.system("python train_student.py")

    print("\nStep 5: Evaluate student model performance")
    os.system("python evaluate.py")

    print("\nStep 5: Export student model")
    os.system("python export_model.py")

    print("\nAll steps completed!")

if __name__ == "__main__":
    run_pipeline()
