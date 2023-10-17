import os


path = os.path.dirname(os.path.abspath(__file__))
cutIdx = path.rfind(os.path.sep)
workspacePath = path[:cutIdx]
eeg_path = workspacePath[:workspacePath.rfind(
    os.path.sep)] + os.path.sep + "Project_Files" + os.path.sep + "PatientFilesMicromed" + os.path.sep + "AllPatients" + os.path.sep
data_path = workspacePath + os.path.sep + 'Data' + os.path.sep
images_path = workspacePath + os.path.sep + 'Data' + os.path.sep

os.makedirs(data_path, exist_ok=True)
os.makedirs(images_path, exist_ok=True)
