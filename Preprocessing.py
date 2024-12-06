import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from urllib.parse import urlparse, parse_qs
from pycaret.classification import ClassificationExperiment, load_model, predict_model

# ----------------------------#
#        MODEL TRAINING       #
# ----------------------------#

def train_and_save_model(dataset_path="D:/Machine learning/dataset_full.csv", model_name='tuned_phishing_model'):
    print("\n==== Starting Model Training ====\n")
    
    # 1. Importing dataset
    try:
        df = pd.read_csv(dataset_path)
        print(f"Dataset '{dataset_path}' loaded successfully.\n")
    except FileNotFoundError:
        print(f"Error: The file '{dataset_path}' was not found.")
        return None

    # 2. Summary statistics of the dataset
    print("Dataset Description:")
    print(df.describe())
    print("\n")

    # 3. Visualizing the distribution of the dataset
    plt.figure(figsize=(6,4))
    sns.countplot(x='phishing', data=df)
    plt.title('Distribution of Phishing vs. Not Phishing')
    plt.xlabel('Phishing (1 = Yes, 0 = No)')
    plt.ylabel('Count')
    plt.show()

    # 4. Dropping unnecessary columns
    cols_to_drop = [
        'url_google_index',
        'domain_google_index',
        'qty_vowels_domain',
        'server_client_domain',
        'tld_present_params',
        'time_response', 
        'domain_spf', 
        'qty_ip_resolved', 
        'qty_nameservers', 
        'qty_mx_servers', 
        'ttl_hostname', 
        'url_shortened'
    ]
    df = df.drop(columns=cols_to_drop, axis=1, errors='ignore')
    print(f"Dropped columns: {cols_to_drop}\n")

    # 5. Feature Engineering
    def extract_features(df):
        rows, columns = df.shape
        original_features = list(df.columns)
        dataset_array = np.array(df)
        
        features_indices = []
        attributes = ['url', 'domain', 'directory', 'file', 'params']
        
        new_dataset = {}
        
        for index, name in enumerate(original_features):
            if 'qty' in name and name.split('_')[-1] in attributes:
                features_indices.append([index, name.split('_')[-1]])
            else:
                new_dataset[name] = dataset_array[:, index]
        
        for index, attribute in features_indices:
            if attribute == 'domain':
                if f"qty_char_{attribute}" not in new_dataset.keys():
                    new_dataset[f"qty_char_{attribute}"] = np.zeros(rows)
                
                new_dataset[f"qty_char_{attribute}"] += dataset_array[:, index]
            # Add similar aggregations for other attributes if needed
        
        df_features = pd.DataFrame(new_dataset).astype(int)
        df_features[df_features < -1] = -1  # Handle any specific data cleaning as needed
        
        return df_features

    df_features = extract_features(df)
    print(df_features.head())
    print(df_features.columns)

    print("Feature engineering completed.\n")

    # 6. Correlation Matrix
    corr = df_features.corr()
    plt.figure(figsize=(20, 20))  # Adjusted for better visibility
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # 7. Summary statistics of the engineered features
    print("Engineered Features Description:")
    print(df_features.describe())
    print("\n")

    # 8. Setting up the data for modeling
    print("Setting up the data for modeling...")
    exp = ClassificationExperiment()
    exp.setup(data=df_features, target='phishing', session_id=123, verbose=False)

    # 9. Comparing and selecting the best model
    print("Comparing models to find the best one...")
    best_model = exp.compare_models()

    # 10. Tuning the hyperparameters of the best performing model
    print("Tuning the best model...")
    tuned_model = exp.tune_model(best_model, n_iter=1, optimize='F1')  # Increased iterations for better tuning

    # 11. Finalizing and saving the model
    print("Finalizing the model...")
    final_model = exp.finalize_model(tuned_model)
    exp.save_model(final_model, model_name)  # Corrected line
    print(f"Model trained and saved as '{model_name}.pkl'.\n")

    print("==== Model Training Completed ====\n")
    return final_model

# ----------------------------#
#      URL CLASSIFICATION     #
# ----------------------------#

def extract_features_from_url(url):
    features = {}
    
    parsed = urlparse(url)

    # Features based on the provided columns
    features['length_url'] = len(url)
    features['domain_length'] = len(parsed.netloc)
    features['domain_in_ip'] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', parsed.netloc) else 0
    features['directory_length'] = len(parsed.path)
    features['file_length'] = len(os.path.splitext(parsed.path)[1])  # Length of the file extension
    features['params_length'] = len(parsed.query)
    
    # Additional features
    features['email_in_url'] = 1 if re.search(r'[\w\.-]+@[\w\.-]+', url) else 0
    features['asn_ip'] = 0  # Placeholder, requires an external lookup for the ASN (if necessary)
    features['time_domain_activation'] = -1  # Placeholder, needs actual data
    features['time_domain_expiration'] = -1  # Placeholder, needs actual data
    features['tls_ssl_certificate'] = 0  # Placeholder, requires SSL check
    features['qty_redirects'] = 0  # Placeholder, would need to analyze the URL
    
    # Domain related features
    domain = parsed.netloc
    features['qty_char_domain'] = sum(1 for char in domain if char.isalnum())  # Count alphanumeric characters in domain

    # Ensure all required features are present
    required_features = [
        'length_url',
        'domain_length',
        'domain_in_ip',
        'directory_length',
        'file_length',
        'params_length',
        'email_in_url',
        'asn_ip',
        'time_domain_activation',
        'time_domain_expiration',
        'tls_ssl_certificate',
        'qty_redirects',
        'qty_char_domain'
    ]
    
    for feat in required_features:
        if feat not in features:
            features[feat] = 0  # Assign default value if feature is missing
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])
    
    return features_df
def classify_url(url, model_path='tuned_phishing_model'):
    # Step 1: Extract features from the URL
    features_df = extract_features_from_url(url)
    
    # Step 2: Load the trained model
    if not os.path.exists(f"{model_path}.pkl"):
        print(f"Model file '{model_path}.pkl' not found. Please train and save the model first.")
        return None, None
    
    model = load_model(model_path)
    
    # Step 3: Make prediction
    prediction = predict_model(model, data=features_df)
    
    # Print the prediction DataFrame to check its structure
    print("Prediction Output:")
    print(prediction)  # This will help debug the actual structure
    
    # Step 4: Interpret the prediction
    # Check the actual names of the columns in the prediction output
    if 'prediction_label' in prediction.columns and 'prediction_score' in prediction.columns:
        label = prediction['prediction_label'][0]
        score = prediction['prediction_score'][0]  
    else:
        print("Prediction output does not contain expected columns. Please check the model output.")
        return None, None
    
    result = "Phishing" if label == 1 else "Not Phishing"
    
    prediction_details = {
        'Label': result,
        'Score': score
    }
    
    return result, prediction_details


# ----------------------------#
#         MAIN FUNCTION       #
# ----------------------------#

def main():
    print("==== Phishing URL Detection System ====\n")
    
    # Check if the model already exists
    model_filename = 'tuned_phishing_model'
    if not os.path.exists(f"{model_filename}.pkl"):
        print("Trained model not found. Initiating training process...")
        final_model = train_and_save_model(dataset_path="D:/Machine learning/dataset_full.csv", model_name=model_filename)
        if final_model is None:
            print("Model training failed. Exiting the program.")
            return
    else:
        print(f"Trained model '{model_filename}.pkl' found. Skipping training.\n")
    
    while True:
        url = input("Enter a URL to classify (or type 'exit' to quit): ").strip()
        
        if url.lower() == 'exit':
            print("Exiting the classifier. Goodbye!")
            break
        
        # Basic URL validation
        if not re.match(r'^https?://', url):
            print("Invalid URL format. Please include the protocol (e.g., http:// or https://).\n")
            continue
        
        # Classify the URL
        result, details = classify_url(url, model_path=model_filename)
        
        if result:
            print(f"\nClassification Result: {result}")
            print(f"Confidence Score: {details['Score']:.4f}\n")
        else:
            print("Classification failed due to an error.\n")

if __name__ == "__main__":
    main()
