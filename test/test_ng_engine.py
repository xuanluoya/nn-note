from nanograd.engine import Value

# Value测试
a = Value(2, label="a")
b = Value(-3, label="b")
c = Value(10.0, label="c")


def test_base():
    print(f"{a}, {b}")
    print(a + b)
    print(a * b)


def test_other():
    print(a + 1)
    print(a * 1)
    print(1 + a)
    print(1 * a)
    print(a - 1)
    print(a / 2)
    print(1 - a)
    print(2 / a)
    print(a * -1)
    print(a.exp())
    print(a.tanh())


def test_backward():
    # 实际上，当a=2时，太大的值会出现消失梯度问题。
    # 这是双曲正切函数的问题，如果出现全是0的情况，请降低数值。
    a = Value(0.5, label="a")

    # 表达式: td = tanh(exp(a**2))
    tb = a**2  # tb.data = 4.0, tb._prev = {a}
    tc = tb.exp()  # tc.data = e^4 ≈ 54.5982, tc._prev = {tb}
    td = tc.tanh()  # td.data = tanh(e^4) ≈ 1.0, td._prev = {tc}

    td.backward()

    print("--- 结果 ---")
    print(f"a: {a}")
    print(f"tb: {tb}")
    print(f"tc: {tc}")
    print(f"td: {td}")

    print("\n--- 梯度检查 ---")
    print(f"a.grad: {a.grad:.10f}")
    print(f"tb.grad: {tb.grad:.10f}")
    print(f"tc.grad: {tc.grad:.10f}")


print("=== Test Nanograd ===")
print("=== base ===")
test_base()
print("=== other ===")
test_other()
print("=== backward ===")
test_backward()
