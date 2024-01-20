import pickle

pickle_path = r"C:\Users\keela\Documents\Models\Basic_CCE\initial_detections.pkl"

# Load data from the pickle file
with open(pickle_path, 'rb') as pkl:
    detections_dict = pickle.load(pkl)

# Print the loaded data
print(detections_dict)
