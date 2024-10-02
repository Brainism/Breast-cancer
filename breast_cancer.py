import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

root_dir = r'C:/Breast-cancer/training_set'
benign_dir = os.path.join(root_dir, 'benign')
malignant_dir = os.path.join(root_dir, 'malignant')

def analyze_image_data(dataset_path):
    image_data = {'width': [], 'height': [], 'label': [], 'num_masks': [], 'image_path': []}
    
    for label in ['benign', 'malignant']:
        class_dir = os.path.join(dataset_path, label)
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.png') and '_mask' not in file_name:
                img_path = os.path.join(class_dir, file_name)
                img = Image.open(img_path)
                width, height = img.size
                study_id = file_name.split('.')[0]
                mask_count = sum(1 for f in os.listdir(class_dir) if f.startswith(study_id) and '_mask' in f)
                image_data['width'].append(width)
                image_data['height'].append(height)
                image_data['label'].append(label)
                image_data['num_masks'].append(mask_count)
                image_data['image_path'].append(img_path)
    
    return pd.DataFrame(image_data)

image_df = analyze_image_data(root_dir)

X = image_df[['width', 'height', 'num_masks']]
y = image_df['label'].map({'benign': 0, 'malignant': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

classification_rep = classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'])
print("Classification Report:\n", classification_rep)

conf_matrix = confusion_matrix(y_test, y_pred)

roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(2, 2, 2)
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()

plt.subplot(2, 2, 3)
sns.countplot(data=image_df, x='label', palette='Set1')
plt.title("Distribution of Benign and Malignant Images")
plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.subplot(2, 2, 4)
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.sort_values().plot(kind='barh', color='skyblue')
plt.title('Feature Importance')
plt.xlabel('Importance Score')

plt.tight_layout()
plt.show()