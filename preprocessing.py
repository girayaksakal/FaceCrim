import cv2
import os
from PIL import Image

# Haarcascade yüz tanıma modelini yükle
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Fotoğraf işleme fonksiyonu
def preprocess_image(image_path, output_path):
    # Fotoğrafı yükle
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return False
    
    # Siyah beyaz dönüşümü
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        print(f"Error: Unable to convert image {image_path} to grayscale. {e}")
        print(str(e))
        return False

    # Yüz tespiti
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    if len(faces) == 0:
        print(f"No face detected in {image_path}. Skipping...")
        return False

    # Fotoğrafı sadece yüz bölgesi gözükecek şekilde kırp
    x, y, w, h = faces[0]
    cropped_face = gray_image[y:y+h, x:x+w]

    # Yeniden boyutlandırma
    resized_face = cv2.resize(cropped_face, (224, 224))

    # Fotoğrafı kaydet
    Image.fromarray(resized_face).save(output_path)
    return True

# İşlemi bütün veri seti için uygula ve yeni bir veri seti oluştur
def preprocess_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        for image_name in os.listdir(class_path):
            input_image_path = os.path.join(class_path, image_name)
            output_image_path = os.path.join(output_class_path, image_name)

            # Fotoğrafı işleyemezse atla
            success = preprocess_image(input_image_path, output_image_path)
            if success:
                print(f"Processed {input_image_path} -> {output_image_path}")
            else:
                print(f"Skipped {input_image_path}")

# İşlenecek ve işlenmiş veri seti dizinleri
input_directory = "data_samples_raw"  # İşlenmemiş veri seti
output_directory = "data_samples_processed"  # İşlenmiş veri seti

preprocess_dataset(input_directory, output_directory)

# İşlem sonucunda hatalı fotoğraflar olabilir. Bunları manuel olarak düzeltmek gerekebilir.