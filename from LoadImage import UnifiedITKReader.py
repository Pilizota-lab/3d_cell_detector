from LoadImage import UnifiedITKReader

reader = UnifiedITKReader()
# Adjust the path to an actual file and use proper backslashes or forward slashes
image_path = r'\\wsl.localhost\Ubuntu\home\urte\MEDIAR-main\test_images\23_21_31 _16.tiff'  
image_data = reader.read(image_path)

print("Image Data Shape:", image_data.shape)
print("Data Type:", type(image_data))
print("Max, Min:", image_data.max(), image_data.min())  
