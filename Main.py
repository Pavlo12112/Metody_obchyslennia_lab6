"""Лабораторна робота. LU-розклад. Ітераційний метод уточнення розв'язку СЛАР."""

import random

N = 100
X_EXACT = 2.5
SEED = 33
EPS0 = 1e-14
MAX_ITER = 100


# ---------- Крок 1: генерація та запис у файли ----------
def generate_matrix(n: int, rng: random.Random) -> list[list[float]]:
    """Випадкова матриця зі строгим діагональним переважанням
    (гарантує існування LU-розкладу без перестановок)."""
    a = [[rng.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        off_diag = sum(abs(a[i][j]) for j in range(n) if j != i)
        a[i][i] = off_diag * rng.uniform(1.05, 1.3)
    return a


def save_matrix(path: str, a: list[list[float]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in a:
            f.write(" ".join(f"{v:.17g}" for v in row) + "\n")


def save_vector(path: str, v: list[float]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for x in v:
            f.write(f"{x:.17g}\n")


def generate_system() -> None:
    rng = random.Random(SEED)
    a = generate_matrix(N, rng)
    b = [sum(a[i][j] * X_EXACT for j in range(N)) for i in range(N)]
    save_matrix("matrix_A.txt", a)
    save_vector("vector_B.txt", b)
    print(f"Згенеровано матрицю A ({N}x{N}) -> matrix_A.txt")
    print(f"Згенеровано вектор B ({N}) -> vector_B.txt\n")


# ---------- Крок 2: зчитування, лінійна алгебра ----------
def read_matrix(path: str) -> list[list[float]]:
    with open(path, encoding="utf-8") as f:
        return [[float(v) for v in line.split()] for line in f if line.strip()]


def read_vector(path: str) -> list[float]:
    with open(path, encoding="utf-8") as f:
        return [float(line) for line in f if line.strip()]


def mat_vec(a: list[list[float]], x: list[float]) -> list[float]:
    """Добуток матриці на вектор."""
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in a]


def vector_norm(v: list[float]) -> float:
    """Нескінченна норма вектора: max |v_i|."""
    return max(abs(x) for x in v)


def residual(a, x, b) -> list[float]:
    """R = B - A X."""
    return [bi - axi for bi, axi in zip(b, mat_vec(a, x))]


# ---------- LU-розклад ----------
def lu_decomposition(a: list[list[float]]):
    """A = L U, L - нижня трикутна, U - верхня з одиничною діагоналлю.
    l_ik = a_ik - sum_{j<k} l_ij u_jk,  i >= k
    u_ki = (a_ki - sum_{j<k} l_kj u_ji) / l_kk,  i > k
    """
    n = len(a)
    l = [[0.0] * n for _ in range(n)]
    u = [[0.0] * n for _ in range(n)]
    for i in range(n):
        u[i][i] = 1.0
    for k in range(n):
        for i in range(k, n):                       # k-й стовпець L
            l[i][k] = a[i][k] - sum(l[i][j] * u[j][k] for j in range(k))
        if abs(l[k][k]) < 1e-300:
            raise ZeroDivisionError("Нульовий ведучий елемент: LU-розклад неможливий")
        for i in range(k + 1, n):                   # k-й рядок U
            u[k][i] = (a[k][i] - sum(l[k][j] * u[j][i] for j in range(k))) / l[k][k]
    return l, u


def save_lu(l, u, path: str = "LU_decomposition.txt") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Матриця L\n")
        for row in l:
            f.write(" ".join(f"{v:.17g}" for v in row) + "\n")
        f.write("Матриця U\n")
        for row in u:
            f.write(" ".join(f"{v:.17g}" for v in row) + "\n")
    print(f"LU-розклад записано -> {path}")


def solve_lu(l, u, b: list[float]) -> list[float]:
    """Розв'язок AX = B: спочатку LZ = B, потім UX = Z."""
    n = len(b)
    z = [0.0] * n
    z[0] = b[0] / l[0][0]
    for k in range(1, n):
        z[k] = (b[k] - sum(l[k][j] * z[j] for j in range(k))) / l[k][k]
    x = [0.0] * n
    x[n - 1] = z[n - 1]
    for k in range(n - 2, -1, -1):
        x[k] = z[k] - sum(u[k][j] * x[j] for j in range(k + 1, n))
    return x


# ---------- Ітераційне уточнення ----------
def refine(a, b, l, u, x0, eps=EPS0, max_iter=MAX_ITER):
    """A dX = R, R = B - A X(0);  X = X(0) + dX.
    Зупинка: ||dX|| <= eps і ||A X - B|| <= eps (або ||dX|| <= eps,
    якщо нев'язка досягла межі машинної точності)."""
    x = x0[:]
    history = []
    prev_res = float("inf")
    for k in range(1, max_iter + 1):
        r = residual(a, x, b)
        dx = solve_lu(l, u, r)
        x = [xi + di for xi, di in zip(x, dx)]
        dnorm = vector_norm(dx)
        rnorm = vector_norm(residual(a, x, b))
        history.append((dnorm, rnorm))
        if dnorm <= eps and (rnorm <= eps or rnorm >= prev_res):
            break
        prev_res = rnorm
    return x, len(history), history


def main() -> None:
    generate_system()                                   # крок 1
    a = read_matrix("matrix_A.txt")                     # крок 2
    b = read_vector("vector_B.txt")
    n = len(a)
    exact = [X_EXACT] * n

    l, u = lu_decomposition(a)
    save_lu(l, u)
    lu_err = max(abs(sum(l[i][k] * u[k][j] for k in range(n)) - a[i][j])
                 for i in range(n) for j in range(n))
    print(f"max|LU - A| = {lu_err:.3e}\n")

    x0 = solve_lu(l, u, b)                              # крок 3
    eps = vector_norm(residual(a, x0, b))               # крок 4
    print("Розв'язок за допомогою LU-розкладу:")
    print(f"  x_1 = {x0[0]:.15f}, x_100 = {x0[-1]:.15f}")
    print(f"  eps = max|sum(a_ij x_j) - b_i| = {eps:.3e}")
    print(f"  max|x - 2.5| = {vector_norm([xi - X_EXACT for xi in x0]):.3e}\n")

    x, k, hist = refine(a, b, l, u, x0)                 # крок 5
    print(f"Уточнення розв'язку (eps0 = {EPS0:g}):")
    for i, (dn, rn) in enumerate(hist, 1):
        print(f"  ітерація {i}: ||dX|| = {dn:.3e}, ||AX - B|| = {rn:.3e}")
    print(f"  потрібно ітерацій: {k}")
    print(f"  max|x - 2.5| = {vector_norm([xi - X_EXACT for xi in x]):.3e}")


if __name__ == "__main__":
    main()