from typing import Any, Literal


class SimplexTable:
    __slots__ = (
        "core_height",
        "core_width",
        "cb",
        "cj",
        "nonbasic_names",
        "basic_names",
        "extended_core",
        "task_type",
    )

    def __init__(
        self,
        *,
        core: list[list[float]],
        cb: list[float],
        cj: list[float],
        a0: list[float],
        nonbasic_names: list[str],
        basic_names: list[str],
        task_type: Literal["minimization", "maximization"] = "maximization",
    ) -> None:
        assert len(core) == len(cb)
        assert len(core) == len(a0)
        assert len(core[0]) == len(cj)
        assert len(nonbasic_names) == len(cj)
        assert len(basic_names) == len(cb)
        assert task_type in ("minimization", "maximization")

        self.core_height = len(core)
        self.core_width = len(core[0])

        self.cb = cb
        self.cj = cj
        self.nonbasic_names = nonbasic_names
        self.basic_names = basic_names
        self.task_type = task_type

        self.extended_core = []

        for index, row in enumerate(core):
            self.extended_core.append([*row, a0[index]])

        self.extended_core.append(self._init_f_string())

    def _delta_i(self, i: int) -> float:
        result = 0.0
        for j in range(self.core_height):
            result += self.cb[j] * self.extended_core[j][i]
        return result - self.cj[i]

    def _q(self) -> float:
        result = 0.0
        for i in range(self.core_height):
            result += self.cb[i] * self.extended_core[i][-1]
        return result

    def _init_f_string(self) -> list[float]:
        result = []
        for i in range(self.core_width):
            result.append(self._delta_i(i))
        result.append(self._q())
        return result

    def report_matrix(self) -> list[list[Any]]:
        result = [
            ["" for _ in range(self.core_width + 3)]
            for _ in range(self.core_height + 4)
        ]

        result[0][1] = "Cj"
        result[1][0] = "Cb"
        result[1][-1] = "A0"
        result[-2][1] = "f"
        result[-1][-1] = "Q"

        for col_num in range(self.core_width):
            result[-1][col_num + 2] = f"Δ{col_num + 1}"
            result[1][col_num + 2] = self.nonbasic_names[col_num]
            result[0][col_num + 2] = self.cj[col_num]  # type: ignore

        for row_num in range(self.core_height):
            result[row_num + 2][1] = self.basic_names[row_num]
            result[row_num + 2][0] = self.cb[row_num]  # type: ignore

        r, s = self.rs_position()

        for row_num, row in enumerate(self.extended_core):
            for col_num, elem in enumerate(row):
                if row_num == r and col_num == s:
                    result[row_num + 2][col_num + 2] = f"[{elem:.2f}]"
                    continue
                result[row_num + 2][col_num + 2] = f"{elem:.2f}"

        return result

    def deep_copy(self) -> "SimplexTable":
        core_copy = [row[:-1].copy() for row in self.extended_core][:-1]
        a0_copy = []

        for row in self.extended_core:
            a0_copy.append(row[-1])

        return SimplexTable(
            core=core_copy,
            cb=self.cb.copy(),
            cj=self.cj.copy(),
            a0=a0_copy[:-1],
            nonbasic_names=self.nonbasic_names.copy(),
            basic_names=self.basic_names.copy(),
            task_type=self.task_type,
        )

    def is_optimal(self) -> bool:
        no_rs_col = self.rs_column_index() == -1
        if self.task_type == "maximization":
            return (
                all(x >= 0 for x in self.extended_core[-1][:-1]) or no_rs_col
            )
        return all(x <= 0 for x in self.extended_core[-1][:-1]) or no_rs_col

    # Если для некоторой симплекс-таблицы все элементы разрешающего
    # столбца отрицательны, то целевая функция неограниченна сверху
    # (при нахождений максимума) или снизу (при нахождении минимума)
    # и поставленная задача решения не имеет.
    def is_unbounded(self) -> bool:
        rs_col = self.rs_column_index()
        for row_num in range(self.core_height):
            if self.extended_core[row_num][rs_col] >= 0:
                return False
        return True

    # Для того, чтобы решить симплекс-методом задачу минимизации,
    # необходимо изменить правило выбора разрешающего столбца
    # (выбирать столбец s, для которого Δs >= 0).
    def can_be_rs_column(self, col_index: int) -> bool:
        assert 0 <= col_index < self.core_width
        delta = self.extended_core[-1][col_index]
        if self.task_type == "maximization":
            return delta < 0
        return delta >= 0

    def rs_column_index(self) -> int:
        current_abs_max = -1.0
        current_abs_max_col = -1
        for col_num in range(self.core_width):
            if self.can_be_rs_column(col_num):
                delta = self.extended_core[-1][col_num]
                if abs(delta) > current_abs_max:
                    current_abs_max = abs(delta)
                    current_abs_max_col = col_num
        return current_abs_max_col

    def rs_row_index(self) -> int:
        current_min = float("inf")
        current_min_row = -1
        rs_col = self.rs_column_index()
        for row_num in range(self.core_height):
            elem = self.extended_core[row_num][rs_col]
            if elem > 0:
                ratio = self.extended_core[row_num][-1] / elem
                if ratio < current_min:
                    current_min = ratio
                    current_min_row = row_num
        return current_min_row

    def rs_position(self) -> tuple[int, int]:
        return self.rs_row_index(), self.rs_column_index()

    def step(self) -> "SimplexTable":
        result = self.deep_copy()

        r, s = self.rs_position()

        # Переменные xr и xs меняются местами
        result.basic_names[r], result.nonbasic_names[s] = (
            result.nonbasic_names[s],
            result.basic_names[r],
        )
        result.cb[r], result.cj[s] = result.cj[s], result.cb[r]

        # Разрешающий элемент заменяется на обратный
        result.extended_core[r][s] = 1 / result.extended_core[r][s]

        # Элементы разрешающей строки делятся на разрешающий элемент
        for col_num in range(self.core_width + 1):
            if col_num != s:
                result.extended_core[r][col_num] /= self.extended_core[r][s]

        # Элементы разрешающего столбца делятся
        # на разрешающий элемент и меняют знак
        for i in range(self.core_height + 1):
            if self.extended_core[i][s] == 0:
                continue
            if i != r:
                result.extended_core[i][s] /= -self.extended_core[r][s]

        # Остальные элементы пересчитываются по «правилу прямоугольника»
        for i in range(self.core_height + 1):
            for col_num in range(self.core_width + 1):
                if i == r or col_num == s:
                    continue

                result.extended_core[i][col_num] = (
                    self.extended_core[i][col_num] * self.extended_core[r][s]
                    - self.extended_core[i][s] * self.extended_core[r][col_num]
                ) / self.extended_core[r][s]

        return result
