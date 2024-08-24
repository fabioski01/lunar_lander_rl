import os

FOLDER_NAME = 'torch_model'
os.makedirs(FOLDER_NAME, exist_ok=True)
os.makedirs('plots', exist_ok=True)

try:
    with open(f'{FOLDER_NAME}/test_file.txt', 'w') as f:
        f.write('This is a test file.')
    print("Test file saved successfully.")
except Exception as e:
    print(f"Error saving test file: {e}")
