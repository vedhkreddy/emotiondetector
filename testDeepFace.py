from deepface import DeepFace
import os

objs = DeepFace.analyze(img_path="googledrive-archive/DST_002_F_ANG.jpg", actions=['emotion'])
# Directory containing the images
image_dir = "googledrive-archive"

# Counters for correct and total predictions
correct_predictions = 0
total_images = 0

i = 0

# Iterate through all files in the directory
for filename in os.listdir(image_dir):
	if i == 100:  # Limit to 10 images
		break
	i += 1
	if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image files
		total_images += 1
		img_path = os.path.join(image_dir, filename)
		
		# Analyze the image
		objs = DeepFace.analyze(img_path=img_path, actions=['emotion'])
		emotion = objs[0]['dominant_emotion']
		print(emotion)
		
		# Map prefixes to emotions
		emotion_map = {
			"HPY": "happy",
			"DIS": "disgust",
			"NUT": "neutral",
			"SAD": "sad",
			"SUR": "surprise",
			"ANG": "angry",
			"FER": "fear", 
		}
		
		# Find the expected emotion based on the prefix in the filename
		expected_emotion = None
		for prefix, emotion_name in emotion_map.items():
			if prefix in filename:
				expected_emotion = emotion_name
				break
		
		# Skip if no expected emotion is found
		if expected_emotion is None:
			continue
		
		# Check if the prediction matches the expected emotion
		if emotion.lower() == expected_emotion:
			print("yay")
			correct_predictions += 1

# Calculate and print the accuracy
accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
print(f"Accuracy: {accuracy:.2f}%")