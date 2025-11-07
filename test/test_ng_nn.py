from nanograd.nn import MLP

# 数据集
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # 期望的目标

n = MLP(3, [4, 4, 1])

for epoch in range(100):  # 迭代100次
    # 前向传播
    ypred = [n(x) for x in xs]

    # 计算损失
    loss = sum([(yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)])  # pyright: ignore[reportOperatorIssue]

    # 清零梯度（重要！）
    # 如果不清理梯度，我们参数更新的就没意义了。
    for p in n.parameters():
        p.grad = 0.0

    # 反向传播
    loss.backward()  # pyright: ignore[reportAttributeAccessIssue]

    # 参数更新
    for p in n.parameters():
        # 梯度 ∂L/∂w 告诉我们：当前参数下，损失函数对参数的变化率
        # 它指向损失增加最快的方向，所以我们需要反向更新参数
        #  - 如果 p.grad > 0，增加这个参数会让损失变大
        #  - 如果 p.grad < 0，增加这个参数会让损失变小
        # 正是因为我们想要减少损失，所以我们需要向梯度的反方向更新参数：
        p.data -= 0.1 * p.grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")  # pyright: ignore[reportAttributeAccessIssue]
