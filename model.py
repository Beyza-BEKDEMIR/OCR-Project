import cv2
import numpy as np
import zipfile as zf
import pytesseract as pt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator

# Bu dosyada makine öğrenmesi modeli oluşturdum.
file_path = "C:\\Users\\şehitler ölmez\\Desktop\\Python Temelleri\\Bolum 1\\dataset.zip"

try:
    with zf.ZipFile(file_path, "r") as zip_ref:
        # Dosyayı açmak için yapılacak işlemler
        print("Dosya başarıyla açıldı.")
except FileNotFoundError:
    print("Dosya mevcut değil.")
except PermissionError:
    print("Dosyaya erişim izni yok.")

train_images = []
test_images = []

with zf.ZipFile(file_path, "r") as zip_ref:
    file_list = zip_ref.namelist()
    for file_path in file_list:
        if file_path.startswith("data/training_data") or file_path.startswith("data2/training_data"):
            with zip_ref.open(file_path) as image_file:
                img_data = image_file.read()
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                train_images.append(img)
        elif file_path.startswith("data/testing_data") or file_path.startswith("data2/testing_data"): 
            with zip_ref.open(file_path) as image_file:
                img_data = image_file.read()
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                test_images.append(img)

print("Training Images Count:", len(train_images))
print("Testing Images Count:", len(test_images))
resized_images = []
for image in train_images:
    resized_image = cv2.resize(image, (30, 27))
    resized_images.append(resized_image)

train_images = np.array(resized_images)

resized_test_images = []
for image in test_images:
    resized_image = cv2.resize(image, (30, 27))
    resized_test_images.append(resized_image)

test_images = np.array(resized_test_images)

train_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
                "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

unique_labels = np.unique(train_labels)
train_labels_encoded = np.searchsorted(unique_labels, train_labels)
print("resimlerin boyutu: ", train_images.shape)  # train_images dizisinin boyutu
print("etiketlerin sayısı: ", train_labels_encoded.shape)  # train_labels_encoded dizisinin boyutu


# Görüntülerin boyutunu yeniden şekillendirme
train_images = np.expand_dims(train_images, axis=-1)
train_images = train_images / 255.0  # Normalizasyon    

# Test verilerinin boyutunu yeniden şekillendirme
test_images = np.expand_dims(test_images, axis=-1)
test_images = test_images / 255.0  # Normalizasyon

new_train_labels_encoded = []

for label_index in range(36):
    if label_index >= 10:
        label = chr(ord('A') + label_index - 10)  # Eğer etiketler harf ise (10'dan büyükse) sayıları harfe çevirecek.
    else:
        label = str(label_index)  # Eğer etiketler sayı ise (0-9 arası)

    new_train_labels_encoded.extend([label] * 573)
    
   
train_labels_encoded = new_train_labels_encoded

encoder = LabelEncoder()
train_labels_encoded = encoder.fit_transform(np.repeat(train_labels, 573))
train_labels_encoded = np.tile(train_labels_encoded, 2)

# Test verileri etiketlerini tanımlama
test_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
                "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

test_labels_encoded = np.array([str(i) if i < 10 else chr(ord('A') + i - 10) for i in range(36)])

# Test verilerinin etiketlerini kodlama
test_labels_encoded = encoder.transform(np.repeat(test_labels, 28))
test_labels_encoded = np.tile(test_labels_encoded, 2)


print("resimlerin boyutu: ", train_images.shape)  # train_images dizisinin boyutu
print("etiketlerin sayısı: ", train_labels_encoded.shape) 

# Veri artırma için ImageDataGenerator nesnesi oluşturma
datagen = ImageDataGenerator(
    rotation_range=45,  # Rastgele döndürme
    width_shift_range=0.2,  # Genişlik kaydırma
    height_shift_range=0.2,  # Yükseklik kaydırma
    #shear_range=0.2,  # Kesme dönüşümü
    zoom_range=0.2,  # Yakınlaştırma
    #horizontal_flip=True,  # Yatay çevirme
)

# Veri artırma işlemini uygulama
datagen.fit(train_images)

# CNN modelini oluşturma
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(27, 30, 1))) #Resmin boyutları ve renk kanalları yazılır. Siyah-beyaz olduğu için tek kanal kullanılır.
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))  # L2 regularizasyonu uygulama
model.add(Dense(36, activation='softmax'))

# Modeli derleme
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(train_images, train_labels_encoded, epochs=10, batch_size=32, validation_split=0.2)

# Doğruluk ve kayıp değerleri alınır
accuracy = history.history['accuracy']
loss = history.history['loss']

"""# Burda bütün test verilerini tahmin eder.
predictions = model.predict(test_images)
predicted_labels = [test_labels[np.argmax(prediction)] for prediction in predictions]

print("Tahmin Sonuçları: ", predicted_labels)
#predicted_labels = encoder.inverse_transform(np.argmax(predictions, axis=1)) """

# Doğruluk ve kayıp değerlerini görselleştirin
plt.plot(accuracy, label='Train Accuracy')
plt.plot(loss, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()

# Test verileri üzerinde modelin performansını değerlendirme
test_loss, test_accuracy = model.evaluate(test_images, test_labels_encoded)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

# Eğitilmiş modeli kaydetme
model.save('ocr_model.keras')