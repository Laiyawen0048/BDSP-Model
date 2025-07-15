import os
import shutil


def export_visualization_files(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                src_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, src_folder)
                dest_file_folder = os.path.join(dest_folder, relative_path)

                if not os.path.exists(dest_file_folder):
                    os.makedirs(dest_file_folder)

                dest_file_path = os.path.join(dest_file_folder, file)
                shutil.copy2(src_file_path, dest_file_path)
                print(f'已复制: {src_file_path} 到 {dest_file_path}')


source_directory = r'C:\Users\沐阳\PycharmProjects\pythonProject3\BDSP Model\data_processing'
desktop_directory = os.path.join(os.path.expanduser("~"), "Desktop")
target_directory = os.path.join(desktop_directory, "visualization_results")

export_visualization_files(source_directory, target_directory)