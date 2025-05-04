from tabulate import tabulate

from simplex import SimplexTable

table = SimplexTable(
    core=[[0.2, 0.75, 0.15, 0.15], [0.1, 0.2, 0.2, 0.25]],
    cb=[0, 0],
    cj=[100, 150, 100, 150],
    a0=[1.4, 0.9],
    nonbasic_names=["X1", "X2", "X3", "X4"],
    basic_names=["X5", "X6"],
)


print(tabulate(table.report_matrix(), tablefmt="fancy_grid"))

while not table.is_optimal():
    table = table.step()
    print(tabulate(table.report_matrix(), tablefmt="fancy_grid"))
    if table.is_unbounded():
        break
