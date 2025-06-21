import torch
import torch.nn as nn

# 1. 创建一个简单的模型
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# 2. 创建一个优化器
optimizer = torch.optim.Adam(model.parameters())

# 3. 从模型创建 id -> name 的映射
param_id_to_name = {id(p): name for name, p in model.named_parameters()}

print("--- 验证开始 ---")
print(f"从 model.named_parameters() 中建立了 {len(param_id_to_name)} 个参数的映射。\n")

# 4. 遍历 optimizer.param_groups
found_in_optimizer = 0
for group in optimizer.param_groups:
    for p in group['params']:
        param_id = id(p)
        
        # 检查从优化器中获取的参数ID是否存在于我们之前创建的映射中
        if param_id in param_id_to_name:
            found_in_optimizer += 1
            param_name = param_id_to_name[param_id]
            print(f"✅ 成功: 在 optimizer 中找到参数 id: {param_id}, 它对应模型中的参数名: '{param_name}'")
        else:
            print(f"❌ 失败: 在 optimizer 中找到的参数 id: {param_id} 不在映射中！")

print("\n--- 验证结束 ---")
print(f"总共在 optimizer 中找到了 {found_in_optimizer} 个参数，与映射中的数量一致。")