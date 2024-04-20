# Tugas Besar 2 IF3270 Pembelajaran Mesin Bagian A

## Deskripsi Tugas:
Rancanglah program dalam bahasa Python untuk membuat jaringan saraf tiruan pada bagian
feed forward neural network (FFNN).
### Spesifikasi
1. Pada bagian A, program berfungsi sebagai neural network yang dapat melakukan feed forward dari input yang diberikan.
2. Program dapat menerima masukan dari file JSON.
3. Implementasi pada fungsi aktivasi berikut:
- Linear
- ReLU
- Sigmoid
- Softmax
4. Program dapat menyimpan bobot (weights) dan struktur model.
5. Implementasi forward propagation untuk FFNN dengan kemampuan:
a. Menampilkan model berupa struktur jaringan dan bobotnya, formatnya bebas.
b. Memberikan output untuk input 1 instance.
c. Memberikan output untuk input berupa batch.
6. Direkomendasikan membuat layer dan fungsi aktivasi secara modular.

## How to run:
1. Pergi ke src directory
```
> cd src/api
```
2. Install all requirements environment to run all of them
```
> pip install -r requirements.txt
```
3. Untuk menjalankan program dapat pada file main.py dengan cara mengganti path dari test case yang ingin ditest pada line 6 di file main.py. Kemudian dapat langsung menjalankannya diterminal dengan cara mengetik command berikut.
```
> python main.py
```
atau dengan cara menjalankannya langsung dari file .ipynb untuk dapat memperoleh visualisasi untuk setiap test case.