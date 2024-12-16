import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import cv2  # OpenCV 임포트

root_dir = r'C:\Breast-cancer'
dataset_dirs = ['training_set']

def analyze_image_data(dataset_paths):
    image_data = {
        'width': [],
        'height': [],
        'label': [],
        'num_masks': [],
        'image_path': []
    }
    
    for dataset_path in dataset_paths:
        for label in ['benign', 'malignant']:
            class_dir = os.path.join(dataset_path, label)
            
            if not os.path.isdir(class_dir):
                print(f"디렉토리가 존재하지 않습니다: {class_dir}")
                continue
            
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.png') and '_mask' not in file_name:
                    img_path = os.path.join(class_dir, file_name)
                    try:
                        img = Image.open(img_path)
                        width, height = img.size
                    except Exception as e:
                        print(f"이미지를 열 수 없습니다: {img_path}. 오류: {e}")
                        continue
                    study_id = file_name.split('.')[0]
                    mask_count = sum(1 for f in os.listdir(class_dir) if f.startswith(study_id) and '_mask' in f)
                    
                    image_data['width'].append(width)
                    image_data['height'].append(height)
                    image_data['label'].append(label)
                    image_data['num_masks'].append(mask_count)
                    image_data['image_path'].append(img_path)
    
    return pd.DataFrame(image_data)

def plot_class_distribution(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    
    ax = sns.countplot(data=df, x='label', palette='Set2')
    
    plt.title("Distribution of Benign and Malignant Images", fontsize=16)
    plt.xlabel("Class", fontsize=14)
    plt.ylabel("Number of Images", fontsize=14)
    
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    plt.show()

def plot_image_size_distribution(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['width'], color='skyblue', kde=True, label='Width', bins=50, stat="density")
    sns.histplot(df['height'], color='salmon', kde=True, label='Height', bins=50, stat="density")
    plt.title("Distribution of Image Dimensions (Width & Height)", fontsize=16)
    plt.xlabel("Pixels", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend()
    plt.show()

def plot_num_masks_distribution(df):

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='num_masks', palette='Greens_d')
    plt.title("Distribution of Number of Masks per Image", fontsize=16)
    plt.xlabel("Number of Masks", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.show()

def show_image_with_masks(image_path):

    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"이미지를 열 수 없습니다: {image_path}. 오류: {e}")
        return
    study_id = os.path.basename(image_path).split('.')[0]
    class_dir = os.path.dirname(image_path)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Original Image: {study_id}", fontsize=16)
    plt.axis('off')
    plt.show()
    
    mask_files = [f for f in os.listdir(class_dir) if f.startswith(study_id) and '_mask' in f]
    
    for mask_file in mask_files:
        mask_path = os.path.join(class_dir, mask_file)
        try:
            mask_img = Image.open(mask_path)
        except Exception as e:
            print(f"마스크 이미지를 열 수 없습니다: {mask_path}. 오류: {e}")
            continue
        
        plt.figure(figsize=(6, 6))
        plt.imshow(mask_img, cmap='gray')
        plt.title(f"Mask: {mask_file}", fontsize=16)
        plt.axis('off')
        plt.show()

def load_and_preprocess_image_cv2(image_path, img_size=(128, 128)):

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("이미지를 로드할 수 없습니다.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
    except Exception as e:
        print(f"이미지를 열 수 없습니다: {image_path}. 오류: {e}")
        return None
    
    try:
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        print(f"이미지를 리사이즈할 수 없습니다: {image_path}. 오류: {e}")
        return None
    
    img_array = img / 255.0  # 픽셀 값을 [0,1] 범위로 정규화
    return img_array

def prepare_dataset(df, img_size=(128, 128)):

    images = []
    labels = []
    failed_images = []

    print("\nPreparing dataset...")
    for index, row in df.iterrows():
        img_array = load_and_preprocess_image_cv2(row['image_path'], img_size)
        if img_array is not None:
            images.append(img_array)
            labels.append(row['label'])
        else:
            failed_images.append(row['image_path'])

    images = np.array(images)
    labels = np.array(labels)

    if failed_images:
        print("\n로딩에 실패한 이미지 목록:")
        for img_path in failed_images:
            print(f" - {img_path}")

    if len(labels) == 0:
        raise ValueError("모든 이미지 로딩에 실패했습니다. 데이터셋을 확인하세요.")

    le = LabelEncoder()
    integer_labels = le.fit_transform(labels)

    lb = LabelBinarizer()
    labels = lb.fit_transform(integer_labels)

    if labels.shape[1] == 1:
        labels = np.hstack((1 - labels, labels))

    print(f"Dataset prepared with {len(images)} images and {len(labels)} labels.")
    return images, labels

def build_cnn_model(input_shape):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history):
    sns.set(style="whitegrid")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6)) # 하나의 axes 객체 생성

    # Accuracy (왼쪽 y축)
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color=color)
    sns.lineplot(x=epochs_range, y=acc, label='Training Accuracy', marker='o', color=color, ax=ax1)
    sns.lineplot(x=epochs_range, y=val_acc, label='Validation Accuracy', marker='o', linestyle='--', color=color, ax=ax1)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 1]) # Accuracy y축 범위 설정

    # Loss (오른쪽 y축)
    ax2 = ax1.twinx() # 두 번째 y축 생성
    color = 'tab:orange'
    ax2.set_ylabel('Loss', color=color)
    sns.lineplot(x=epochs_range, y=loss, label='Training Loss', marker='o', color=color, ax=ax2)
    sns.lineplot(x=epochs_range, y=val_loss, label='Validation Loss', marker='o', linestyle='--', color=color, ax=ax2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, max(max(loss), max(val_loss)) * 1.1]) # Loss y축 범위 설정

    # Legend 설정 (두 axes의 label을 합쳐서 하나의 legend 표시)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")  # or loc="center right"

    fig.tight_layout() # tight_layout 적용
    plt.title("Training History") # 전체 그래프 제목 추가
    plt.show()

def plot_confusion_matrix(cm, X_test, y_test, classes=['Benign', 'Malignant'], title='Confusion Matrix'):
    """
    혼동 행렬을 시각화 합니다.
    """
    num_samples = min(5, len(X_test))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.show()

    plot_sample_predictions(cnn_model, X_test, y_test, num_samples=num_samples)


def plot_sample_predictions(model, X_test, y_test, class_names=['Benign', 'Malignant'], num_samples=5):
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        img = X_test[idx]
        true_label = np.argmax(y_test[idx])
        pred_probs = model.predict(np.expand_dims(img, axis=0))[0]
        pred_label = np.argmax(pred_probs)
        confidence = pred_probs[pred_label]

        # 각 샘플마다 새로운 Figure 생성
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True, squeeze=False)

        axes[0, 0].imshow(img, aspect='auto')
        axes[0, 0].axis('off')
        axes[0, 0].set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]} ({confidence*100: .2f}%)", fontsize=5)

        sns.barplot(x=class_names, y=pred_probs, hue=class_names, palette= 'viridis', legend=False, ax=axes[0, 1])
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_title(f"True: {class_names[true_label]}\mPred: {class_names[pred_label]} ({confidence*100:.2f}%)", fontsize=5)
        axes[0, 1].tick_params(labelsize=8)

        plt.savefig(f"sample_prediction_{i}.png", bbox_inches='tight')
        plt.show()

def plot_misclassified_samples(model, X_test, y_test, class_names=['Benign', 'Malignant'], num_samples=5):
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    misclassified_indices = np.where(y_pred_classes != y_true)[0]
    
    if len(misclassified_indices) == 0:
        print("잘못 분류된 샘플이 없습니다.")
        return

    num_samples = min(num_samples, len(misclassified_indices))
    selected_indices = np.random.choice(misclassified_indices, num_samples, replace=False)
    
    for i, idx in enumerate(selected_indices):
        img = X_test[idx]
        true_label = y_true[idx]
        pred_label = y_pred_classes[idx]
        confidence = y_pred[idx][pred_label]

        fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)

        axes[0].imshow(img, aspect='auto')
        axes[0].axis('off')
        axes[0].set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]} ({confidence*100:.2f}%)", fontsize=5)
        
        
        # 막대 그래프 형태로 예측 확률 표시
        sns.barplot(x=class_names, y=y_pred[idx], hue=class_names, palette='magma', legend=False, ax=axes[1])
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Prediction Probabilities", fontsize=5)
        axes[1].tick_params(labelsize=8)

    plt.savefig("misclassified_samples.png", bbox_inches='tight')
    plt.show()

from sklearn.metrics import roc_curve, auc

def plot_roc_curves_comparison(cnn_model, rf_model, X_test_cnn, y_test_cnn, X_test_rf, y_test_rf):
    """
    CNN과 Random Forest 모델의 ROC 커브를 하나의 그래프에 표시하여 비교합니다.
    """

    # CNN 모델 예측 확률
    y_pred_cnn = cnn_model.predict(X_test_cnn)
    y_true_cnn = np.argmax(y_test_cnn, axis=1)

    # Random Forest 모델 예측 확률
    y_pred_rf = rf_model.predict_proba(X_test_rf)[:, 1]  # 악성 확률만 사용
    y_true_rf = y_test_rf

    plt.figure(figsize=(8, 6))

    # CNN ROC 커브
    fpr_cnn, tpr_cnn, _ = roc_curve(y_true_cnn, y_pred_cnn[:, 1])
    roc_auc_cnn = auc(fpr_cnn, tpr_cnn)
    plt.plot(fpr_cnn, tpr_cnn, color='darkorange', lw=2, 
label=f'CNN (AUC = {roc_auc_cnn:.2f})')

    # Random Forest ROC 커브
    fpr_rf, tpr_rf, _ = roc_curve(y_true_rf, y_pred_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, color='navy', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # 대각선 점선
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves Comparison', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.show()

def predict_image(image_path, model, img_size=(128, 128)):
    """
    학습된 모델을 사용하여 단일 이미지의 클래스를 예측합니다.
    """
    img_array = load_and_preprocess_image_cv2(image_path, img_size)
    if img_array is None:
        return "Invalid Image"
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_label = 'benign' if class_index == 0 else 'malignant'
    return class_label

def display_sample_images(image_df, num_samples=2):
    """
    양성 및 악성 이미지를 2x2 그리드 형태로 표시합니다.
    """
    benign_paths = image_df[image_df['label'] == 'benign'] ['image_path'].sample(num_samples, random_state=42).tolist()
    malignant_paths = image_df[image_df['label'] == 'malignant']['image_path'].sample(num_samples, random_state=42).tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 그리드 생성

    image_paths = benign_paths + malignant_paths
    labels = ['Benign'] * num_samples + ['Malignant'] * num_samples


    for i, (path, label) in enumerate(zip(image_paths, labels)):
        img = Image.open(path)
        row = i // 2
        col = i % 2
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{label} Image #{i+1}", fontsize=10)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

# --- 메인 실행 흐름 ---.

if __name__ == "__main__":
    # 루트 디렉토리 설정 (데이터셋 상위 디렉토리)
    root_dir = r'C:\Breast-cancer'  
    # 데이터셋 하위 폴더
    dataset_dirs = ['training_set']
    # 각 데이터셋의 전체 경로 생성
    full_dataset_paths = [os.path.join(root_dir, ds) for ds in dataset_dirs]
    
    # 디렉토리 경로 확인
    print("Generated dataset paths:")
    for path in full_dataset_paths:
        print(path)
        if os.path.exists(path):
            print(f"Path exists: {path}")
        else:
            print(f"Error: Path does not exist - {path}")

    # 이미지 데이터를 분석하여 DataFrame 생성
    print("Analyzing image data...")
    image_df = analyze_image_data(full_dataset_paths)
    print("Image data analysis completed.")

    # 샘플 이미지 표시 (여기에 추가)
    print("\nDisplaying sample images...")
    display_sample_images(image_df)
    print("Sample images displayed.")
    
    # 데이터셋의 통계 요약 정보 출력
    print("\n데이터셋 통계 요약:")
    print(image_df.describe())
    
    # 클래스 분포 시각화
    print("\nPlotting class distribution...")
    plot_class_distribution(image_df)
    print("Class distribution plotted.")
    
    # 이미지 크기 분포 시각화
    print("\nPlotting image size distribution...")
    plot_image_size_distribution(image_df)
    print("Image size distribution plotted.")
    
    # 마스크 수 분포 시각화
    print("\nPlotting number of masks distribution...")
    plot_num_masks_distribution(image_df)
    print("Number of masks distribution plotted.")
    
    # 데이터셋의 첫 번째 이미지와 관련 마스크 표시
    if not image_df.empty:
        sample_image_path = image_df.loc[0, 'image_path']
        print(f"\nDisplaying sample image and its masks: {sample_image_path}")
        show_image_with_masks(sample_image_path)
    else:
        print("데이터셋이 비어 있습니다.")
    
    # 데이터셋 준비
    try:
        images, labels = prepare_dataset(image_df, img_size=(128, 128))
    except ValueError as ve:
        print(f"데이터셋 준비 중 오류 발생: {ve}")
        exit(1)
    
    # 데이터가 충분한지 확인
    print(f"\n전체 이미지 수: {len(images)}")
    print(f"레이블 분포:\n{pd.Series(labels[:,1]).value_counts()}")  # Assuming labels[:,1] is the malignant class
    
    # 학습용과 테스트용 데이터로 분할 (stratify를 사용하여 클래스 비율 유지)
    print("\nSplitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Random Forest 모델 학습을 위한 데이터 준비
    X_rf = image_df [['width', 'height', 'num_masks']]
    y_rf = image_df ['label'].map({'benign': 0, 'malignant': 1})
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X_rf, y_rf, test_size=0.2, random_state=42,
stratify=y_rf  
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # CNN 모델 생성
    print("\nBuilding CNN model...")
    input_shape = (128, 128, 3)
    cnn_model = build_cnn_model(input_shape)
    print("CNN model built.")

    # Random Forest 모델 생성 및 학습
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_rf, y_train_rf)

    # ROC 커브 비교
    plot_roc_curves_comparison(cnn_model, rf_model, X_test, y_test, X_test_rf, y_test_rf)
    
    # 모델 요약 출력
    cnn_model.summary()
    
    # 모델 학습: 10 에포크, 배치 사이즈 16, 검증 데이터 사용
    print("\nStarting training...")
    history = cnn_model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_test, y_test),
        batch_size=16
    )
    print("Training completed.")
    
    # 학습 과정 시각화
    print("\nPlotting training history...")
    plot_training_history(history)
    print("Training history plotted.")
    
    # 테스트 데이터셋에 대한 모델 평가
    print("\nEvaluating model on test data...")
    loss, accuracy = cnn_model.evaluate(X_test, y_test)
    print(f"모델 정확도: {accuracy * 100:.2f}%")
    print(f"모델 손실: {loss:.4f}")
    
    # 예측 및 혼동 행렬 생성
    print("\nGenerating predictions and confusion matrix...")
    y_pred = cnn_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred_classes)
    print("혼동 행렬:")
    print(cm)
    
    # 혼동 행렬 시각화
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(cm, X_test, y_test)
    print("Confusion matrix plotted.")
    
    # 분류 보고서 출력
    print("\nGenerating classification report...")
    print("분류 보고서:")
    print(classification_report(y_true, y_pred_classes, target_names=['Benign', 'Malignant']))
    
    # 샘플 예측 결과 시각화
    print("\nPlotting sample predictions...")
    num_samples = min(5, len(X_test))
    plot_sample_predictions(cnn_model, X_test, y_test, num_samples=num_samples)
    print("Sample predictions plotted.")
    
    # 잘못 분류된 샘플 시각화
    print("\nPlotting misclassified samples...")
    plot_misclassified_samples(cnn_model, X_test, y_test)
    print("Misclassified samples plotted.")
    
    # 예측을 위한 샘플 이미지 경로
    if not image_df.empty:
        sample_image_path = image_df.loc[0, 'image_path']
        print(f"\nPredicting class for sample image: {sample_image_path}")
        prediction = predict_image(sample_image_path, cnn_model)
        print(f"예측 결과: {prediction}")
    else:
        print("데이터셋이 비어 있어 예측을 수행할 수 없습니다.")
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
