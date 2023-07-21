import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
import zipfile as zf
import requests
from io import BytesIO
from sklearn.preprocessing import LabelEncoder

# Oluşturduğum modeli kullanarak modelin doğru sonuç verip vermediğini test ettim.
loaded_model = load_model('ocr_model.keras')

file_path = "C:\\Users\\şehitler ölmez\\Desktop\\Python Temelleri\\Bolum 1\\dataset.zip"
image_file_path_1 = "data/testing_data/A/28320.png"  
image_file_path_2 = "data/testing_data/3/28313.png" 
image_file_path_3 = "data2/training_data/J/56879.png"
image_file_path_4 = "data2/testing_data/7/41299.png"
image_file_path_5 = "data/testing_data/C/28898.png"
image_file_path_6 = "data/training_data/B/20135.png"
image_file_path_7 = "data/testing_data/0/29210.png"
image_file_path_8 = "data/testing_data/O/28442.png"
image_file_path_9 = "data/training_data/K/18380.png"
image_file_path_10 = "data/training_data/Q/18206.png"

# Tahmin sonuçlarını etiketlere dönüştürme
encoder = LabelEncoder()
encoder.classes_ = np.array([str(i) if i < 10 else chr(ord('A') + i - 10) for i in range(36)])

# Bu kısımda veri setindeki verileri okutma işlemi yaptım.
try:
    with zf.ZipFile(file_path, "r") as zip_ref:
        # Dosyayı açmak için yapılacak işlemler
        print("Dosya başarıyla açıldı.")
        
        # Resmi okuma işlemleri
        with zip_ref.open(image_file_path_1) as resim:
            resim_data = resim.read()
            nparr = np.frombuffer(resim_data, np.uint8)
            rsm = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            cv2.imshow("Image 1", rsm)
            cv2.waitKey(0) # Sayfanın kapanıp kodların devam etmesi için herhangi bir tuşa basmak gerekir.
            cv2.destroyAllWindows()

            # Test verilerini numpy dizisine dönüştürme
            resized_img = cv2.resize(rsm, (30, 27))
            input_image = np.expand_dims(resized_img, axis=-1) / 255.0
            input_image = np.expand_dims(input_image, axis=0)

            # Tahmin yapma
            predictions = loaded_model.predict(input_image)
            predicted_labels = encoder.inverse_transform(np.argmax(predictions, axis=1))
            print("Resimdeki metin: ", predicted_labels[0])

        with zip_ref.open(image_file_path_2) as resim:
            resim_data = resim.read()
            nparr = np.frombuffer(resim_data, np.uint8)
            rsm2 = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            cv2.imshow("Image 2", rsm2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()  
            
            resized_img_2 = cv2.resize(rsm2, (30, 27))
            input_image_2 = np.expand_dims(resized_img_2, axis=-1) / 255.0
            input_image_2 = np.expand_dims(input_image_2, axis=0)    

            predictions2 = loaded_model.predict(input_image_2)
            predicted_labels_2 = encoder.inverse_transform(np.argmax(predictions2, axis=1))
            print("Resimdeki metin: ", predicted_labels_2[0])

        with zip_ref.open(image_file_path_3) as resim:
            resim_data = resim.read()
            nparr = np.frombuffer(resim_data, np.uint8)
            rsm3 = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            cv2.imshow("Image 3", rsm3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()  
            
            resized_img_3 = cv2.resize(rsm3, (30, 27))
            input_image_3 = np.expand_dims(resized_img_3, axis=-1) / 255.0
            input_image_3 = np.expand_dims(input_image_3, axis=0)    

            predictions3 = loaded_model.predict(input_image_3)
            predicted_labels_3 = encoder.inverse_transform(np.argmax(predictions3, axis=1))
            print("Resimdeki metin: ", predicted_labels_3[0])   

        with zip_ref.open(image_file_path_4) as resim:
            resim_data = resim.read()
            nparr = np.frombuffer(resim_data, np.uint8)
            rsm4 = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            cv2.imshow("Image 4", rsm4)
            cv2.waitKey(0)
            cv2.destroyAllWindows()  
            
            resized_img_4 = cv2.resize(rsm4, (30, 27))
            input_image_4 = np.expand_dims(resized_img_4, axis=-1) / 255.0
            input_image_4 = np.expand_dims(input_image_4, axis=0)    

            predictions4 = loaded_model.predict(input_image_4)
            predicted_labels_4 = encoder.inverse_transform(np.argmax(predictions4, axis=1))
            print("Resimdeki metin: ", predicted_labels_4[0])

        with zip_ref.open(image_file_path_5) as resim:
            resim_data = resim.read()
            nparr = np.frombuffer(resim_data, np.uint8)
            rsm5 = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            cv2.imshow("Image 5", rsm5)
            cv2.waitKey(0)
            cv2.destroyAllWindows()  
            
            resized_img_5 = cv2.resize(rsm5, (30, 27))
            input_image_5 = np.expand_dims(resized_img_5, axis=-1) / 255.0
            input_image_5 = np.expand_dims(input_image_5, axis=0)    

            predictions5 = loaded_model.predict(input_image_5)
            predicted_labels_5 = encoder.inverse_transform(np.argmax(predictions5, axis=1))
            print("Resimdeki metin: ", predicted_labels_5[0])

        with zip_ref.open(image_file_path_6) as resim:
            resim_data = resim.read()
            nparr = np.frombuffer(resim_data, np.uint8)
            rsm6 = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            cv2.imshow("Image 6", rsm6)
            cv2.waitKey(0)
            cv2.destroyAllWindows()  
            
            resized_img_6 = cv2.resize(rsm6, (30, 27))
            input_image_6 = np.expand_dims(resized_img_6, axis=-1) / 255.0
            input_image_6 = np.expand_dims(input_image_6, axis=0)    

            predictions6 = loaded_model.predict(input_image_6)
            predicted_labels_6 = encoder.inverse_transform(np.argmax(predictions6, axis=1))
            print("Resimdeki metin: ", predicted_labels_6[0])

        with zip_ref.open(image_file_path_7) as resim:
            resim_data = resim.read()
            nparr = np.frombuffer(resim_data, np.uint8)
            rsm7 = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            cv2.imshow("Image 7", rsm7)
            cv2.waitKey(0)
            cv2.destroyAllWindows()  
            
            resized_img_7 = cv2.resize(rsm7, (30, 27))
            input_image_7 = np.expand_dims(resized_img_7, axis=-1) / 255.0
            input_image_7 = np.expand_dims(input_image_7, axis=0)    

            predictions7 = loaded_model.predict(input_image_7)
            predicted_labels_7 = encoder.inverse_transform(np.argmax(predictions7, axis=1))
            print("Resimdeki metin: ", predicted_labels_7[0])   

        with zip_ref.open(image_file_path_8) as resim:
            resim_data = resim.read()
            nparr = np.frombuffer(resim_data, np.uint8)
            rsm8 = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            cv2.imshow("Image 8", rsm8)
            cv2.waitKey(0)
            cv2.destroyAllWindows()  
            
            resized_img_8 = cv2.resize(rsm8, (30, 27))
            input_image_8 = np.expand_dims(resized_img_8, axis=-1) / 255.0
            input_image_8 = np.expand_dims(input_image_8, axis=0)    

            predictions8 = loaded_model.predict(input_image_8)
            predicted_labels_8 = encoder.inverse_transform(np.argmax(predictions8, axis=1))
            print("Resimdeki metin: ", predicted_labels_8[0])    

        with zip_ref.open(image_file_path_9) as resim:
            resim_data = resim.read()
            nparr = np.frombuffer(resim_data, np.uint8)
            rsm9 = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            cv2.imshow("Image 9", rsm9)
            cv2.waitKey(0)
            cv2.destroyAllWindows()  
            
            resized_img_9 = cv2.resize(rsm9, (30, 27))
            input_image_9 = np.expand_dims(resized_img_9, axis=-1) / 255.0
            input_image_9 = np.expand_dims(input_image_9, axis=0)    

            predictions9 = loaded_model.predict(input_image_9)
            predicted_labels_9 = encoder.inverse_transform(np.argmax(predictions9, axis=1))
            print("Resimdeki metin: ", predicted_labels_9[0])   

        with zip_ref.open(image_file_path_10) as resim:
            resim_data = resim.read()
            nparr = np.frombuffer(resim_data, np.uint8)
            rsm10 = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            cv2.imshow("Image 10", rsm10)
            cv2.waitKey(0)
            cv2.destroyAllWindows()  
            
            resized_img_10 = cv2.resize(rsm10, (30, 27))
            input_image_10 = np.expand_dims(resized_img_10, axis=-1) / 255.0
            input_image_10 = np.expand_dims(input_image_10, axis=0)    

            predictions10 = loaded_model.predict(input_image_10)
            predicted_labels_10 = encoder.inverse_transform(np.argmax(predictions10, axis=1))
            print("Resimdeki metin: ", predicted_labels_10[0])     
except FileNotFoundError:
    print("Dosya mevcut değil.")
except PermissionError:
    print("Dosyaya erişim izni yok.")

# Bu kısımda ise veri seti dışındaki resimler üzerinden okutma işlemi yaptım.
image_urls = [
    ('https://www.woodykidstore.com.tr/dekoratif-harfler-flamalar-1521-66-B.jpg', 'Resim 1'),
    ('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSZEFLvhjtZsapZlghQHnQ9bqsjd_CkKBs7hRDFtR4T5V9zZr3FeJQlfg83sH6XgSjCOkg&usqp=CAU', 'Resim 2'),
    ('https://i.pinimg.com/236x/13/5d/19/135d19079622a7a91730ea619753761b.jpg', 'Resim 3'),
    ('https://2.bp.blogspot.com/-4YoGk5wOBzg/XIZ14DXH65I/AAAAAAAAxAg/6Vqvup2MqL4j2TIAN5-9EeEphNVMBygMgCLcBGAs/s1600/Y_harfi_buyuk.jpg', 'Resim 4'),
    ('https://i.ebayimg.com/00/s/NjAwWDQ2Mw/u003d/u003d/z/ovkAAOxyyjpRt3UI/$(KGrHqJHJFMFG)LnodcYBRt3UIJQ)w~~60_1.JPG', 'Resim 5')
]

for image_url, image_name in image_urls:
    response = requests.get(image_url)
    image_bytes = BytesIO(response.content)
    img = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    # Resmi ekranda gösterme
    cv2.imshow(image_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Resmi yeniden boyutlandırma ve tahmin etmek için modele uygun hale getirme
    resized_img = cv2.resize(img, (30, 27))
    input_image = np.expand_dims(resized_img, axis=-1) / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    
    # Tahmin yapma
    predictions = loaded_model.predict(input_image)
    predicted_label = np.argmax(predictions[0])
    harf_etiket = encoder.inverse_transform([predicted_label])[0]
    
    print("Tahmin Sonucu ({}): {}".format(image_name, harf_etiket))
