
# Metode Numerik UTS Tugas 1 2 3
# Perbandingan metode Newton Raphson dengan Biseksi dengan Matplotlib ( Soal No 1 )

import numpy as np
import matplotlib.pyplot as plt

# Konstanta
L = 0.5  # Induktansi dalam Henry
C = 10e-6  # Kapasitansi dalam Farad
target_f = 1000  # Frekuensi target dalam Hz
tolerance = 0.1  # Toleransi error dalam Ohm

# Fungsi untuk menghitung f(R)
def f_R(R):
    term_inside_sqrt = 1 / (L * C) - (R**2) / (4 * L**2)
    if term_inside_sqrt <= 0:
        return None  # Invalid jika akar kuadrat negatif
    return (1 / (2 * np.pi)) * np.sqrt(term_inside_sqrt)

# Turunan f'(R) untuk metode Newton-Raphson
def f_prime_R(R):
    term_inside_sqrt = 1 / (L * C) - (R**2) / (4 * L**2)
    if term_inside_sqrt <= 0:
        return None  # Invalid jika akar kuadrat negatif
    sqrt_term = np.sqrt(term_inside_sqrt)
    return -R / (4 * np.pi * L**2 * sqrt_term)

# Metode Newton-Raphson untuk mencari R
def newton_raphson_method(initial_guess, tolerance):
    R = initial_guess
    while True:
        f_val = f_R(R)
        if f_val is None:
            return None  # Invalid jika akar kuadrat negatif
        f_value = f_val - target_f
        f_prime_value = f_prime_R(R)
        if f_prime_value is None:
            return None  # Invalid jika turunan tidak terdefinisi
        # Update R menggunakan rumus Newton-Raphson
        new_R = R - f_value / f_prime_value
        if abs(new_R - R) < tolerance:
            return new_R
        R = new_R

# Metode Biseksi untuk mencari R
def bisection_method(a, b, tolerance):
    while (b - a) / 2 > tolerance:
        mid = (a + b) / 2
        f_mid = f_R(mid) - target_f
        if f_mid is None:
            return None  # Invalid jika akar kuadrat negatif
        if abs(f_mid) < tolerance:
            return mid
        if (f_R(a) - target_f) * f_mid < 0:
            b = mid
        else:
            a = mid
    return (a + b) / 2

# Menjalankan kedua metode dengan hasil akhir
initial_guess = 50  # Tebakan awal untuk metode Newton-Raphson
interval_a, interval_b = 0, 100  # Interval untuk metode Bisection

# Hasil metode Newton-Raphson
R_newton = newton_raphson_method(initial_guess, tolerance)
f_newton = f_R(R_newton) if R_newton is not None else "Tidak ditemukan"

# Hasil metode Biseksi
R_bisection = bisection_method(interval_a, interval_b, tolerance)
f_bisection = f_R(R_bisection) if R_bisection is not None else "Tidak ditemukan"

# Menampilkan hasil metode Newton Raphson
print("Metode Newton-Raphson:")
print(f"Nilai R: {R_newton} ohm, Frekuensi resonansi: {f_newton} Hz")

# Print nilai hasil metode biseksi
print("\nMetode Bisection:")
print(f"Nilai R: {R_bisection} ohm, Frekuensi resonansi: {f_bisection} Hz")

plt.figure(figsize=(10, 5))
plt.axhline(target_f, color="red", linestyle="--", label="Target Frekuensi 1000 Hz")

# Plot hasil metode Newton-Raphson jika valid
if R_newton is not None:
    plt.scatter(R_newton, f_newton, color="blue", label="Newton-Raphson", zorder=5)
    plt.text(R_newton, f_newton + 30, f"NR: R={R_newton:.2f}, f={f_newton:.2f} Hz", color="blue")

# Plot hasil metode Biseksi jika valid
if R_bisection is not None:
    plt.scatter(R_bisection, f_bisection, color="green", label="Bisection", zorder=5)
    plt.text(R_bisection, f_bisection + 30, f"Bisection: R={R_bisection:.2f}, f={f_bisection:.2f} Hz", color="green")

# Mengatur label dan title
plt.xlabel("Nilai R (Ohm)")
plt.ylabel("Frekuensi Resonansi f(R) (Hz)")
plt.title("Perbandingan Hasil Metode Newton-Raphson dan Bisection")
plt.legend()
plt.grid(True)
plt.show()


# MetodeGaus ( Soal No 2 )

import numpy as np

# Membuat Matriks koefisien dan vektor konstanta
A = np.array([[1, 1, 1],
              [1, 2, -1],
              [2, 1, 2]], dtype=float)

b = np.array([6, 2, 10], dtype=float)

# Membuat sistem metode eliminasi Gauss
def gauss_elimination(A, b):
    n = len(b)
    # menggabungkan A dan b ke dalam satu matriks
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Melakukan proses eliminasi
    for i in range(n):
        # Buat elemen diagonal menjadi 1 dan mengeliminasi dibawahnya
        for j in range(i + 1, n):
            ratio = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= ratio * Ab[i, i:]

    # Proses Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]

    return x

# Eliminasi Gauss-Jordan
def gauss_jordan(A, b):
    n = len(b)
    # Gabungkan A dan b ke dalam satu matriks
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Proses eliminasi
    for i in range(n):
        # Buat elemen diagonal menjadi 1
        Ab[i] = Ab[i] / Ab[i, i]
        for j in range(n):
            if i != j:
                Ab[j] -= Ab[i] * Ab[j, i]

    # Solusi
    return Ab[:, -1]

# Menjalankan kedua metode Gauss
solution_gauss = gauss_elimination(A, b)
solution_gauss_jordan = gauss_jordan(A, b)

# Menampilkan hasil dari kedua metode Eliminasi
print("Solusi dengan Eliminasi Gauss:")
print(f"x1 = {solution_gauss[0]}, x2 = {solution_gauss[1]}, x3 = {solution_gauss[2]}")

# Menampilkan hasil dari kedua metode Eliminasi Gauss Jordan
print("\nSolusi dengan Eliminasi Gauss-Jordan:")
print(f"x1 = {solution_gauss_jordan[0]}, x2 = {solution_gauss_jordan[1]}, x3 = {solution_gauss_jordan[2]}")

# Perbandingan selisih menggunakan 4 metode ( Soal No 3 )

import numpy as np

# Menjalankan fungsi untuk menghitung R(T)
def R(T):
    return 5000 * np.exp(3500 * (1/T - 1/298))

# Fungsi untuk menghitung turunan numerik

# Metode selisih maju
def forwarddifference(T, h):
    return (R(T + h) - R(T)) / h

# Metode selisih mundur
def backwarddifference(T, h):
    return (R(T) - R(T - h)) / h

# Metode nilai tengah
def centraldifference(T, h):
    return (R(T + h) - R(T - h)) / (2 * h)

# Menghitung nilai turunan Eksak
def exactderivative(T):
    return 5000 * np.exp(3500 * (1/T - 1/298)) * (-3500 / T**2)

# Rentang temperatur dan interval
temperatures = np.arange(250, 351, 10)
h = 1e-3  # Interval kecil untuk perbedaan hingga

# Menyimpan hasil untuk setiap metode
results = {
    "Temperature (K)": temperatures,
    "Forward Difference": [forwarddifference(T, h) for T in temperatures],
    "Backward Difference": [backwarddifference(T, h) for T in temperatures],
    "Central Difference": [centraldifference(T, h) for T in temperatures],
    "Exact Derivative": [exactderivative(T) for T in temperatures],
}

import matplotlib.pyplot as plt

# Menghitung error relatif
errors = {
    "Forward Difference Error": np.abs((np.array(results["Forward Difference"]) - np.array(results["Exact Derivative"])) / np.array(results["Exact Derivative"])) * 100,
    "Backward Difference Error": np.abs((np.array(results["Backward Difference"]) - np.array(results["Exact Derivative"])) / np.array(results["Exact Derivative"])) * 100,
    "Central Difference Error": np.abs((np.array(results["Central Difference"]) - np.array(results["Exact Derivative"])) / np.array(results["Exact Derivative"])) * 100,
}

# Plotting error relatif
plt.figure(figsize=(10, 6))
plt.plot(temperatures, errors["Forward Difference Error"], label="Forward Difference Error", marker='o')
plt.plot(temperatures, errors["Backward Difference Error"], label="Backward Difference Error", marker='s')
plt.plot(temperatures, errors["Central Difference Error"], label="Central Difference Error", marker='^')
plt.xlabel("Temperature (K)")
plt.ylabel("Relative Error (%)")
plt.legend()
plt.title("Relative Error of Numerical Derivatives Compared to Exact Derivative")
plt.grid()
plt.show()

# Menjalankan Metode Richardson 
def richardson_extrapolation(T, h):
    f_h2 = centraldifference(T, h / 2)
    f_h = centraldifference(T, h)
    return (4 * f_h2 - f_h) / 3

# Menghitung Richardson extrapolation untuk setiap temperatur
richardson_results = [richardson_extrapolation(T, h) for T in temperatures]

# Menghitung error relatif untuk metode Richardson
richardson_error = np.abs((np.array(richardson_results) - np.array(results["Exact Derivative"])) / np.array(results["Exact Derivative"])) * 100

# Menambahkan hasil Richardson ke plot error
plt.figure(figsize=(10, 6))
plt.plot(temperatures, errors["Forward Difference Error"], label="Forward Difference Error", marker='o')
plt.plot(temperatures, errors["Backward Difference Error"], label="Backward Difference Error", marker='s')
plt.plot(temperatures, errors["Central Difference Error"], label="Central Difference Error", marker='^')
plt.plot(temperatures, richardson_error, label="Richardson Extrapolation Error", marker='x')
plt.xlabel("Temperature (K)")
plt.ylabel("Relative Error (%)")
plt.legend()
plt.title("Relative Error Comparison of Numerical Methods")
plt.grid()
plt.show()
