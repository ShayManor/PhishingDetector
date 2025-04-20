import kagglehub

# Download latest version
path = kagglehub.dataset_download("ethancratchley/email-phishing-dataset")

print("Path to dataset files:", path)