import os
import sys
import argparse
import random
from pathlib import Path

# Load Dataset Folder
Dataset_Path = "../../../../../scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet"

# get bmp files according to the langauge, DPI and style
def get_files(language, dpi, style):
    # Get lists
    language_dir_paths = []
    DPI_list = []
    style_list = []

    if language == "Thai" or language == "English":
        language_dir_paths.append(os.path.join(Dataset_Path, language))
    else:
        language_dir_paths.append(os.path.join(Dataset_Path, "Thai"))
        language_dir_paths.append(os.path.join(Dataset_Path, "English"))
    
    if dpi == "all":
        DPI_list.append("200")
        DPI_list.append("300")
        DPI_list.append("400")
    else:
        DPI_list.append(dpi)

    if style == "all":
        style_list.append("bold")
        style_list.append("bold_italic")
        style_list.append("italic")
        style_list.append("normal")
    else:
        style_list.append(style)

    # Get files paths
    all_files_paths = []
    for language_dir_path in language_dir_paths:
        for letter_dir in os.listdir(language_dir_path):
            letter_dir_path = os.path.join(language_dir_path, letter_dir)
            if os.path.isdir(letter_dir_path):
                for dpi_dir in os.listdir(letter_dir_path):
                    if dpi_dir in DPI_list:
                        dpi_dir_path = os.path.join(letter_dir_path, dpi_dir)
                        for style_dir in os.listdir(dpi_dir_path):
                            if style_dir in style_list:
                                style_dir_path = os.path.join(dpi_dir_path, style_dir)
                                for file in os.listdir(style_dir_path):
                                    if file.endswith('.bmp'):
                                        all_files_paths.append(os.path.join(style_dir_path, file))
    
    return all_files_paths


def split_data_files(selected_files, train_ratio, test_ratio, output_dir):
    random.shuffle(selected_files)
    total = len(selected_files)
    train_split = int(total * train_ratio)
    test_split = int(total * (train_ratio + test_ratio))

    train_files = selected_files[:train_split]
    test_files = selected_files[train_split:test_split]
    valid_files = selected_files[test_split:]

    print(f"Total files: {total}")
    print(f"Training files: {len(train_files)}")
    print(f"Testing files: {len(test_files)}")
    print(f"Validation files: {len(valid_files)}")

    # Save the split files
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, 'train.txt'), 'w') as train_file:
        for path in train_files:
            train_file.write(f"{path}\n")

    with open(os.path.join(output_dir, 'test.txt'), 'w') as test_file:
        for path in test_files:
            test_file.write(f"{path}\n")

    with open(os.path.join(output_dir, 'valid.txt'), 'w') as valid_file:
        for path in valid_files:
            valid_file.write(f"{path}\n")

    print("Data split complete. Files saved in:", output_dir)


def main():
    parser = argparse.ArgumentParser("Load data into training, texting, validation sets based on language, DPI and style.")
    parser.add_argument('--language', type=str, choices=["Thai","English","Both"], help="choose language: Thai, English, or Both")
    parser.add_argument('--dpi', type=str, choices=["200", "300", "400","all"], help="choose dpi: 200, 300, 400, or all")
    parser.add_argument('--style', type=str, choices=["bold","bold_italic","italic","normal", "all"], help="choose style: bold, bold_italic, italic, normal, or all")
    parser.add_argument('--training-ratio', type=float, default=0.8, help="default 0.8")
    parser.add_argument('--testing-ratio', type=float, default=0.1, help="default 0.1")
    # validation default 0.1
    parser.add_argument('--output-dir', type=str, default="./")
    args = parser.parse_args()

    selected_files = get_files(args.language, args.dpi, args.style)
    split_data_files(selected_files, args.training_ratio, args.testing_ratio, args.output_dir)
    print(f"Dataset (Training, Test, Validation) generated. Saved at: {args.output_dir}")

if __name__ == '__main__':
    main()