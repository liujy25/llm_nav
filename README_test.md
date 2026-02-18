# Prompt模式对比测试工具

这个工具用于测试和比较关键帧prompt和非关键帧prompt在相同图片和目标下的LLM输出差异。

## 功能特点

- **自动标注**：使用LLM分析标注图片，自动生成JSON标注文件
- 支持批量测试多张图片和多个目标
- 自动调用LLM获取两种prompt模式的响应
- 提取和比较action决策
- 生成详细的统计报告
- 记录推理时间对比

## 完整工作流程

### 步骤1: 自动生成标注文件

使用 [`auto_annotate.py`](auto_annotate.py:1) 让LLM分析带waypoint标记的图片，自动生成JSON标注：

```bash
python auto_annotate.py \
  --image_dir /path/to/annotated_images \
  --output_dir ./annotations
```

LLM会分析图片并输出：
1. 场景描述
2. 标注的action数量（waypoint数量）
3. 可见物体列表
4. 可能存在的物体列表

### 步骤2: 运行对比测试

使用生成的标注文件进行prompt模式对比测试：

```bash
python test_prompt_comparison.py \
  --data_dir ./annotations \
  --output_dir ./test_results
```

## 自动标注工具 (auto_annotate.py)

### 功能说明

[`auto_annotate.py`](auto_annotate.py:1) 使用LLM自动分析带waypoint标记的图片，生成测试所需的JSON标注文件。

### 标注流程

LLM会分析图片并回答以下问题：
1. **场景描述**：描述看到的场景类型、家具、布局等
2. **Action数量**：统计图片上标注的waypoint数量（1, 2, 3...）
3. **可见物体**：列出场景中可以看到的物体
4. **潜在物体**：列出场景中看不到但这个房间可能存在的物体

### 使用方法

```bash
# 基本用法
python auto_annotate.py --image_dir /path/to/annotated_images --output_dir ./annotations

# 完整参数
python auto_annotate.py \
  --image_dir /path/to/annotated_images \
  --output_dir ./annotations \
  --model /data/sea_disk0/liujy/models/Qwen/Qwen3-VL-8B-Instruct/ \
  --api_key EMPTY \
  --base_url http://10.15.89.71:34134/v1/ \
  --timeout 30 \
  --no_raw  # 不保存原始VLM响应
```

### 输出格式

生成的JSON文件格式：
```json
{
  "scene_description": "This is a living room with a sofa, coffee table...",
  "num_actions": 5,
  "targets": ["chair", "table", "bottle", "lamp", "tv"],
  "visible_objects": ["chair", "table", "bottle"],
  "potential_objects": ["lamp", "tv"]
}
```

### 输出文件

- `<image_name>.json`: 标注数据（用于测试）
- `<image_name>_raw.txt`: 原始VLM响应（用于调试）
- `annotation_summary.json`: 批量处理摘要

## 测试数据格式

### 目录结构
```
test_data/
├── image_001.jpg
├── image_001.json
├── image_002.jpg
├── image_002.json
└── ...
```

### JSON标注格式
每张图片对应一个JSON文件，格式如下：
```json
{
  "targets": ["chair", "table", "bottle"],
  "num_actions": 5
}
```

字段说明：
- `targets`: 目标列表，每个目标会分别测试
- `num_actions`: 可用的waypoint数量（默认5）

## 使用方法

### 完整工作流程示例

```bash
# 方式1: 使用一键脚本（推荐）
chmod +x run_full_test.sh
./run_full_test.sh ./my_annotated_images ./output

# 方式2: 分步执行
# 步骤1: 自动生成标注（LLM分析图片）
python auto_annotate.py \
  --image_dir ./my_annotated_images \
  --output_dir ./annotations

# 步骤2: 运行prompt对比测试
python test_prompt_comparison.py \
  --data_dir ./annotations \
  --output_dir ./test_results

# 查看结果
cat ./test_results/report_*.txt
```

### 基本用法
```bash
python test_prompt_comparison.py \
  --data_dir /path/to/test_data \
  --output_dir ./test_results
```

### 完整参数
```bash
python test_prompt_comparison.py \
  --data_dir /path/to/test_data \
  --output_dir ./test_results \
  --model /data/sea_disk0/liujy/models/Qwen/Qwen3-VL-8B-Instruct/ \
  --api_key EMPTY \
  --base_url http://10.15.89.71:34134/v1/ \
  --timeout 30
```

### 参数说明
- `--data_dir`: 测试数据目录（必需）
- `--output_dir`: 输出结果目录（默认：./test_results）
- `--model`: VLM模型路径
- `--api_key`: API密钥
- `--base_url`: API基础URL
- `--timeout`: API超时时间（秒）

## 输出结果

测试完成后会生成两个文件：

### 1. 详细结果JSON (`comparison_results_YYYYMMDD_HHMMSS.json`)
包含所有测试的完整数据：
```json
{
  "results": [
    {
      "image": "image_001.jpg",
      "num_actions": 5,
      "targets": [
        {
          "target": "chair",
          "keyframe": {
            "prompt": "...",
            "response": "...",
            "action": 3,
            "inference_time": 2.5
          },
          "non_keyframe": {
            "prompt": "...",
            "response": "...",
            "action": 3,
            "inference_time": 1.8
          },
          "action_match": true,
          "action_diff": null
        }
      ]
    }
  ],
  "statistics": {
    "total_tests": 10,
    "action_matches": 7,
    "action_diffs": 3,
    "total_kf_time": 25.0,
    "total_nkf_time": 18.0
  }
}
```

### 2. 统计报告 (`report_YYYYMMDD_HHMMSS.txt`)
包含：
- 总体统计（一致率、平均推理时间等）
- Action不同的详细案例列表

示例：
```
================================================================================
Prompt模式对比测试报告
================================================================================

总体统计:
  总测试数: 10
  Action一致: 7 (70.0%)
  Action不同: 3 (30.0%)
  关键帧平均推理时间: 2.50s
  非关键帧平均推理时间: 1.80s

Action不同的案例:
--------------------------------------------------------------------------------

图片: image_001.jpg
目标: chair
关键帧Action: 3
非关键帧Action: 2
关键帧响应摘要: Question 1: This is the first keyframe...
非关键帧响应摘要: Based on the current observation...
--------------------------------------------------------------------------------
```

## 两种Prompt模式的区别

### 关键帧Prompt（深度推理版）
- 包含5个引导性问题
- 要求历史回顾和战略规划
- 需要详细的推理过程
- 适合战略决策点

### 非关键帧Prompt（快速执行版）
- 简化的上下文回顾
- 直接要求action选择
- 更快的推理速度
- 适合执行既定策略

## 测试流程

1. 扫描测试数据目录，找到所有图片和JSON文件
2. 对每张图片的每个目标：
   - 构建关键帧prompt并调用LLM
   - 构建非关键帧prompt并调用LLM
   - 提取两个响应中的action
   - 比较action是否一致
3. 汇总统计结果
4. 生成报告文件

## 注意事项

1. 确保VLM服务正常运行
2. 图片和JSON文件名需要匹配（除扩展名外）
3. JSON文件必须包含`targets`字段
4. 测试过程中会实时打印进度
5. 每个目标会调用2次LLM，请注意API配额

## 示例：准备测试数据

```python
import json

# 创建测试标注
annotation = {
    "targets": ["chair", "table", "bottle"],
    "num_actions": 5
}

with open("test_data/image_001.json", "w") as f:
    json.dump(annotation, f, indent=2)
```

## 扩展功能

如需添加更多分析功能，可以修改[`test_prompt_comparison.py`](test_prompt_comparison.py:1)中的：
- [`construct_keyframe_prompt()`](test_prompt_comparison.py:52): 修改关键帧prompt
- [`construct_nonkeyframe_prompt()`](test_prompt_comparison.py:93): 修改非关键帧prompt
- [`extract_action()`](test_prompt_comparison.py:169): 改进action提取逻辑
- [`generate_report()`](test_prompt_comparison.py:283): 自定义报告格式
