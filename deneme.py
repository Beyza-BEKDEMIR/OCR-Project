import cv2
import numpy as np
import zipfile as zf
import pytesseract as pt 

#bu dosyada veri setindeki verileri kullanarak tesseract kütüphanesi ile okutma yaptım.
file_path = "C:\\Users\\şehitler ölmez\\Desktop\\Python Temelleri\\Bolum 1\\dataset.zip" # dosyanın yolunu tamamen girmeyince çalışmıyordu.

try:
    with zf.ZipFile(file_path, "r") as zip_ref:
        # Dosyayı açmak için yapılacak işlemler
        print("Dosya başarıyla açıldı.")
except FileNotFoundError:
    print("Dosya mevcut değil.")
except PermissionError:
    print("Dosyaya erişim izni yok.")

# Bütün verileri yazdırır.
"""with zf.ZipFile("C:\\Users\\şehitler ölmez\\Desktop\\Python Temelleri\\Bolum 1\\archive.zip", "r") as zip_ref:
    # PNG dosyalarını okuma
    for file_name in zip_ref.namelist():
        if file_name.lower().endswith(".png"):
           with zip_ref.open(file_name) as image_file:
                img_data = image_file.read()
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                cv2.imshow("Image", img)
                cv2.waitKey(0) #herhangi bir tuşa basınca resimler ilerler.
                cv2.destroyAllWindows()"""
   

pt.pytesseract.tesseract_cmd = "C:\\Users\\şehitler ölmez\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe" # tesseract kullanabilmek için gerekli

with zf.ZipFile("C:\\Users\\şehitler ölmez\\Desktop\\Python Temelleri\\Bolum 1\\dataset.zip", "r") as zip_ref:
    with zip_ref.open("data2/training_data/Z/56859.png") as image_file:
        img_data = image_file.read()
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        text = pt.image_to_string(img, lang='eng')
        print(text)
        image_with_text = img.copy()
        cv2.putText(image_with_text, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Resmi gösterme
        cv2.imshow("Image with Text", image_with_text)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    with zip_ref.open("data/training_data/B/20135.png")  as resim:
         resim_data=resim.read()
         nparr = np.frombuffer(resim_data, np.uint8)
         rsm= cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
         frame =  pt.image_to_string(rsm, lang='eng', config='--oem 3 --psm 6')
         print("Resimdeki Metin:")
         print(frame)
         cv2.imshow("deneme", rsm)
         cv2.waitKey(0)
         cv2.destroyAllWindows()

    with zip_ref.open("data/training_data/E/338.png")  as resim:
        resim_data=resim.read()
        nparr = np.frombuffer(resim_data, np.uint8)
        rsm= cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        frame =  pt.image_to_string(rsm, lang='eng', config='--oem 3 --psm 6')
        print("Resimdeki Metin:")
        print(frame)
        cv2.imshow("deneme", rsm)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    with zip_ref.open("data/training_data/M/382.png")  as resim:
        resim_data=resim.read()
        nparr = np.frombuffer(resim_data, np.uint8)
        rsm= cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        frame =  pt.image_to_string(rsm, lang='eng', config='--oem 3 --psm 6')
        print("Resimdeki Metin:")
        print(frame)
        cv2.imshow("deneme", rsm)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




